"""Fine-tune only existing FP16 codebook entries in a packed Stage 2 policy/value ONNX.

The integer weight assignments, float32 biases, tensor shapes and decoder graph
stay fixed. The training model reconstructs kernels using the same rounded
centroids as the serialization. Run with python -m training.refine_packed.
"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .fit_search_student import load_positions, load_weights
from .pack_weights import byte_lookup
from .pipeline import _symmetry_maps
from .stage2_student import Stage2Net


class Codebook(nn.Module):
    def __init__(self, centers, codes):
        super().__init__()
        self.centers = nn.Parameter(torch.from_numpy(centers.astype(np.float32)))
        self.register_buffer("codes", torch.from_numpy(codes.astype(np.int64)))

    def forward(self, original):
        # Autograd propagates through both casts. The optimizer keeps FP32
        # master values while the forward pass sees the exact serialized FP16.
        rounded = self.centers.half().float()
        dense = rounded.gather(1, self.codes).flatten()
        return dense[:original.numel()].reshape_as(original)


def bind_codebooks(packed, student):
    tables = {t.name: t for t in packed.graph.initializer}
    destinations = {n.input[1]: n.output[0] for n in packed.graph.node
                    if n.op_type == "Reshape" and len(n.input) == 2 and len(n.output) == 1}
    modules = dict(student.named_modules())
    student.requires_grad_(False)
    bindings = {}
    for name, table in tables.items():
        match = re.fullmatch(r"(_*wpack/)(\d+)/centers", name)
        if not match:
            continue
        prefix = match[1] + match[2] + "/"
        parameter = destinations.get(prefix + "shape", "")
        module_name, _, field = parameter.rpartition(".")
        layer = modules.get(module_name)
        if field != "weight" or not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            raise ValueError(f"{name} does not map to a Stage 2 convolution weight: {parameter}")
        shape = tuple(numpy_helper.to_array(tables[prefix + "shape"]).tolist())
        centers = numpy_helper.to_array(table)
        octets = numpy_helper.to_array(tables[prefix + "bytes"])
        rows, width = numpy_helper.to_array(tables[prefix + "groups"]).tolist()
        if (shape != tuple(layer.weight.shape) or centers.dtype != np.float16 or centers.ndim != 2
                or centers.shape[0] != rows or centers.shape[1] not in (4, 16)
                or octets.dtype != np.uint8 or not table.raw_data or not np.isfinite(centers).all()):
            raise ValueError(f"Invalid dimensions, data type or centroid values for {parameter}")
        bits = 2 if centers.shape[1] == 4 else 4
        lookup = byte_lookup(bits)
        if not np.array_equal(lookup, numpy_helper.to_array(tables[match[1] + "lookup"])):
            raise ValueError(f"Unknown byte decoding order for {parameter}")
        n = math.prod(shape)
        if rows <= 0 or width <= 0 or (rows - 1) * width >= n or n > rows * width:
            raise ValueError(f"Invalid group padding for {parameter}")
        codes = lookup[octets].reshape(rows, width)
        if parametrize.is_parametrized(layer, "weight"):
            raise ValueError(f"Multiple codebooks target {parameter}")
        book = Codebook(centers, codes)
        parametrize.register_parametrization(layer, "weight", book)
        bindings[name] = book
    if not bindings:
        raise ValueError("No supported packed Stage 2 weights found")
    return bindings


def layout_hash(model, names):
    """Digest everything except permitted centroid bytes, including every integer assignment."""
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    for tensor in clone.graph.initializer:
        if tensor.name in names:
            tensor.raw_data = bytes(len(tensor.raw_data))
    return hashlib.sha256(clone.SerializeToString()).hexdigest()


def update_centroids(model, bindings):
    changed = 0
    for tensor in model.graph.initializer:
        if tensor.name not in bindings:
            continue
        table = bindings[tensor.name].centers.detach().cpu().numpy().astype("<f2")
        if not np.isfinite(table).all() or tuple(tensor.dims) != table.shape:
            raise ValueError(f"Invalid refined table {tensor.name}")
        changed += int(np.count_nonzero(table != numpy_helper.to_array(tensor)))
        tensor.raw_data = table.tobytes()
    return changed


@torch.inference_mode()
def check_parity(payload, student, states, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(payload, options, providers=["CPUExecutionProvider"])
    if [i.name for i in session.get_inputs()] != ["state"]:
        raise ValueError("Expected a project state input for the packed graph")
    student.cpu().eval()
    errors = {}
    for count in (1, 12):
        x = states[:count]
        actual = session.run(["policy", "value"], {"state": x.numpy()})
        reference = student(x)
        for a, b, width in zip(actual, reference, (81, 1)):
            if a.shape != (count, width) or b.shape != (count, width) or not np.isfinite(a).all():
                raise ValueError("Invalid policy/value shape or nonfinite output")
            np.testing.assert_allclose(a, b.numpy(), atol=4e-4, rtol=1e-4, equal_nan=False)
        errors[count] = [float(np.abs(a - b.numpy()).max()) for a, b in zip(actual, reference)]
    return errors


def refine(student, teacher, books, positions, args, device):
    states, policies, values = positions
    maps = _symmetry_maps()
    x_map, p_map = [torch.as_tensor(np.argsort(np.stack(axis), axis=1), device=device)
                    for axis in zip(*maps)]
    params = [book.centers for book in books.values()]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps,
                                                           eta_min=args.lr / 10)
    for step in range(1, args.steps + 1):
        rows = torch.randint(len(states), (args.batch_size,), device=device)
        symmetry = torch.randint(8, (args.batch_size,), device=device)
        x = states[rows].gather(1, x_map[symmetry]).float()
        target = policies[rows].gather(1, p_map[symmetry]).float()
        target = target / target.sum(1, keepdim=True)
        illegal = x[:, 162:243] == 0
        with torch.no_grad():
            reference, reference_value = teacher(x)
            reference = (reference / args.temperature).masked_fill(illegal, -1e9).softmax(1)
            target_value = torch.lerp(values[rows], reference_value.flatten(), args.teacher_mix)
        logits, value = student(x)
        search_logp = logits.masked_fill(illegal, -1e9).log_softmax(1)
        soft_logp = (logits / args.temperature).masked_fill(illegal, -1e9).log_softmax(1)
        search_loss = -(target * search_logp).sum(1).mean()
        kd_loss = -(reference * soft_logp).sum(1).mean() * args.temperature ** 2
        value_loss = F.mse_loss(value.flatten(), target_value)
        loss = torch.lerp(search_loss, kd_loss, args.teacher_mix) + args.value_weight * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps}: search CE {search_loss.item():.4f}, "
                  f"teacher CE×T² {kd_loss.item():.4f}, value MSE {value_loss.item():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("stage3_teacher", "stage2_moderate", "stage2_student"),
                        required=True)
    parser.add_argument("--base", type=Path, required=True,
                        help="Source .pt for the packed graph, including all frozen biases and offsets")
    parser.add_argument("--teacher", type=Path, required=True, help="Dense stage3_teacher .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Directory of compact search shards")
    parser.add_argument("--max-bytes", type=int, default=1_492_760)
    parser.add_argument("--max-positions", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher-mix", type=float, default=0.8)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("parity.json"))
    args = parser.parse_args()
    if (args.output.suffix != ".onnx" or args.output.exists()
            or args.output.resolve() == args.input.resolve()):
        parser.error("Output must be a fresh .onnx path")
    if not 0 <= args.teacher_mix <= 1 or not math.isfinite(args.value_weight) or args.value_weight < 0:
        parser.error("Invalid mix or value weight")
    if (not math.isfinite(args.lr) or not math.isfinite(args.temperature)
            or min(args.lr, args.temperature, args.steps, args.batch_size, args.threads, args.max_positions) <= 0):
        parser.error("Positive finite training settings required")
    payload = args.input.read_bytes()
    if len(payload) > args.max_bytes:
        parser.error(f"Input is {len(payload):,} bytes and already exceeds the cap")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    packed = onnx.load_model_from_string(payload)
    onnx.checker.check_model(packed)
    student = Stage2Net(compact=args.tier != "stage3_teacher", moderate=args.tier == "stage2_moderate")
    load_weights(student, args.base)
    books = bind_codebooks(packed, student)
    fingerprint = layout_hash(packed, books)
    cases = json.loads(args.fixtures.read_text())["cases"]
    if len(cases) < 12:
        parser.error("At least 12 parity positions are required")
    fixtures = torch.tensor([case["encoded"] for case in cases[:12]], dtype=torch.float32)
    print(json.dumps({"beforeParityMaxAbs": check_parity(payload, student, fixtures, args.threads),
                      "codebooks": len(books), "trainableCentroids": sum(b.centers.numel() for b in books.values()),
                      "layoutSha256": fingerprint}), flush=True)
    teacher = Stage2Net(compact=False)
    load_weights(teacher, args.teacher)
    device = torch.device(args.device)
    teacher.requires_grad_(False).eval().to(device)
    positions = load_positions(args.data, args.max_positions, np.random.default_rng(args.seed), device)
    refine(student.to(device), teacher, books, positions, args, device)
    changed = update_centroids(packed, books)
    result = packed.SerializeToString()
    if len(result) != len(payload) or layout_hash(packed, books) != fingerprint:
        raise ValueError("Packed graph layout or integer assignments changed")
    parity = check_parity(result, student, fixtures, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "candidate.onnx"
        staged.write_bytes(result)
        os.link(staged, args.output)
    print(json.dumps({"output": str(args.output), "bytes": len(result), "changedCentroids": changed,
                      "sourceSha256": hashlib.sha256(payload).hexdigest(),
                      "outputSha256": hashlib.sha256(result).hexdigest(), "afterParityMaxAbs": parity}),
          flush=True)


if __name__ == "__main__":
    main()
