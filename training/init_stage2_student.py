"""Import the Apache-2.0 uttt.ai stage-2 weights and initialize compact affine factors.

See stage2_student.py for the upstream revision and project modifications.
The original ONNX is used offline. Only the initialized student travels into
the compact inference graph. Both checkpoints can be trained on new targets.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from .fit_search_student import export, load_positions
from .stage2_student import FactoredLift, FactoredPoint, Stage2Net
from .wrap_utttai import reference_planes


UPSTREAM = "https://uttt.ai/policy_value_net_stage2.onnx"
SHA256 = "81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2"
# The published ONNX graph has already folded batch norms. Numbered weights
# identify those fused operations. The fixed digest pins their identities.
PARAMETERS = {
    "stem": "conv_block_1.0", "early": ("163", "164"), "trunk": "conv_block_2.3",
    "gate0": ("166", "167"), "pool0": ("169", "170"), "lift0": "conv_block_3a.6",
    "gate1": ("172", "173"), "pool1": ("175", "176"), "lift1": "conv_block_3b.6",
    "gate2": ("178", "179"), "pool2": ("181", "182"), "lift2": "conv_block_3c.6",
    "policy": ("184", "185"), "upsample": "policy_head_upsampling.0",
    "logits": "policy_head_logits.0", "logit_out": "policy_head_logits.2",
    "v1": "value_head_conv1.0", "vpool1": ("187", "188"),
    "v2": "value_head_conv2.0", "vpool2": ("190", "191"),
    "value": "value_head_fc.0", "value_out": "value_head_fc.2",
}


def import_teacher(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Expected the original stage-2 graph from {UPSTREAM}, SHA-256 {SHA256}")
    teacher = Stage2Net(compact=False).eval()
    tensors = {t.name: numpy_helper.to_array(t) for t in onnx.load(source).graph.initializer}
    for name, layer in teacher.layers.items():
        names = PARAMETERS[name]
        weight, bias = names if isinstance(names, tuple) else (names + ".weight", names + ".bias")
        kernel = tensors[weight]
        if name in {"value", "value_out"}:
            kernel = kernel[:, :, None, None]  # Gemm at 1x1 is a pointwise convolution.
        if kernel.shape != tuple(layer.weight.shape) or tensors[bias].shape != tuple(layer.bias.shape):
            raise ValueError(f"Unexpected affine dimensions for {name}")
        layer.weight.copy_(torch.from_numpy(kernel.copy()))
        layer.bias.copy_(torch.from_numpy(tensors[bias].copy()))
    return teacher


def check_parity(teacher, source, fixtures, threads):
    cases = json.loads(fixtures.read_text())["cases"]
    states = torch.tensor([case["encoded"] for case in cases], dtype=torch.float32)
    planes = np.stack([reference_planes(case["state"]) for case in cases])
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    runtime = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    # The published graph may pin its batch dimension to one. Run that exact
    # graph on each fixture and test the imported network at both batch sizes.
    reference = [runtime.run(["policy_logits", "state_value"], {"input": row[None]})
                 for row in planes]
    errors = []
    for n in (1, len(cases)):
        p, v = (np.concatenate([result[i] for result in reference[:n]]) for i in (0, 1))
        expected = (p.reshape(n, 81)[:, teacher.order.numpy()], v[:, None])
        actual = teacher(states[:n])
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a.numpy(), b, atol=4e-4, rtol=1e-4)
        errors.append([float(np.max(np.abs(a.numpy() - b))) for a, b in zip(actual, expected)])
    return {"positions": len(cases), "batchSizes": [1, len(cases)], "maxAbsPolicyValueErrors": errors}


def covariance_factors(teacher, student, states, batch_size, device):
    """Collect input means and second moments without retaining activation maps."""
    moments = {}
    hooks = []

    def capture(name):
        def hook(_layer, args):
            x = args[0].movedim(1, -1).reshape(-1, args[0].size(1)).double()
            sums, gram, count = moments[name]
            sums.add_(x.sum(0))
            gram.add_(x.T @ x)
            moments[name][2] = count + len(x)
        return hook

    for name, layer in student.layers.items():
        if isinstance(layer, (FactoredPoint, FactoredLift)):
            width = teacher.layers[name].in_channels
            moments[name] = [torch.zeros(width, device=device, dtype=torch.float64),
                             torch.zeros(width, width, device=device, dtype=torch.float64), 0]
            hooks.append(teacher.layers[name].register_forward_pre_hook(capture(name)))
    try:
        for batch in states.split(batch_size):
            teacher(batch.to(device).float())
    finally:
        for hook in hooks:
            hook.remove()
    result = {}
    for name, (sums, gram, count) in moments.items():
        mean = sums.cpu() / count
        cov = gram.cpu() / count - mean[:, None] * mean[None, :]
        # A ridge maintains invertibility for nearly constant feature channels.
        ridge = 1e-4 * cov.diag().mean().clamp_min(1e-6)
        cov.diagonal().add_(ridge)
        result[name] = mean, torch.linalg.cholesky(cov)
    return result


def top_modes(gram, rank):
    return torch.linalg.eigh(gram).eigenvectors[:, -rank:]


def initialize_factors(source, target, mean, chol):
    """Minimize calibrated output error subject to input/output rank restrictions."""
    if isinstance(target, FactoredPoint):
        w = source.weight[:, :, 0, 0].T.double()  # [input, output]
        weighted = chol.T @ w
        basis = top_modes(weighted.T @ weighted, target.reduce.out_channels)
        reduced = w @ basis
        target.reduce.weight.copy_(reduced.T[:, :, None, None])
        target.expand.weight.copy_(basis[:, :, None, None])
        target.expand.bias.copy_(source.bias + mean @ w - (mean @ reduced) @ basis.T)
        return

    # The stride-3 decoder has nonoverlapping tiles. Each transposed kernel,
    # including the 1x1 -> 3x3 trunk lifts, maps one input vector to 9 outputs.
    n, m = source.in_channels, source.out_channels
    w = source.weight.permute(0, 2, 3, 1).reshape(n, 9 * m).double()
    weighted = chol.T @ w
    left = top_modes(weighted @ weighted.T, target.reduce.out_channels)
    core = (left.T @ weighted).reshape(-1, 9, m)
    rows = core.reshape(-1, m)
    right = top_modes(rows.T @ rows, target.expand.in_channels)
    core = core @ right
    # Change basis to keep the trainable input projection orthonormal.
    q, r = torch.linalg.qr(torch.linalg.solve_triangular(chol.T, left, upper=True))
    core = (r @ core.flatten(1)).reshape_as(core)
    target.reduce.weight.copy_(q.T[:, :, None, None])
    target.lift.weight.copy_(core.permute(0, 2, 1).reshape_as(target.lift.weight))
    target.expand.weight.copy_(right[:, :, None, None])
    predicted = ((mean @ q) @ core.flatten(1)).reshape(9, -1) @ right.T
    offset = (mean @ w).reshape(9, m) - predicted + source.bias
    target.offset.copy_(offset.T.reshape(1, m, 3, 3))


def agreement(teacher, student, states, batch_size):
    sums = torch.zeros(3, dtype=torch.float64, device=states.device)
    count = 0
    for x in states.split(batch_size):
        legal = x[:, 162:243].bool()
        x = x[legal.any(1)].float()
        if len(x) == 0:
            continue
        legal = x[:, 162:243].bool()
        a, v = teacher(x)
        b, w = student(x)
        a = a.masked_fill(~legal, -1e9).log_softmax(1)
        b = b.masked_fill(~legal, -1e9).log_softmax(1)
        sums += torch.stack(((a.exp() * (a - b)).sum(), (a.argmax(1) == b.argmax(1)).sum(),
                             (v - w).square().sum()))
        count += len(x)
    if not count:
        raise ValueError("No live positions for measuring student agreement")
    kl, top1, mse = (sums / count).tolist()
    return {"positions": count, "legalPolicyKL": kl, "topLegalMoveAgreement": top1, "valueMSE": mse}


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / "weights.pt"
        torch.save(model.cpu().state_dict(), staged)
        os.link(staged, path)
    print(f"Saved {path}", flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original public ONNX with four input planes")
    parser.add_argument("--full-output", type=Path, help="Optional fresh .pt for the stage3_teacher tier")
    parser.add_argument("--output", type=Path, help="Fresh .pt for the selected student tier")
    parser.add_argument("--student-tier", choices=("stage2_student", "stage2_moderate"),
                        default="stage2_student", help="Rank profile of the calibrated student")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Also export and parity-check a native float32 ONNX beside the student .pt")
    parser.add_argument("--data", type=Path, help="NPZ directory for activation calibration")
    parser.add_argument("--validation", type=Path, help="Independent NPZ directory for measuring agreement")
    parser.add_argument("--fixtures", type=Path, default=Path("training/parity.json"))
    parser.add_argument("--positions", type=int, default=4096)
    parser.add_argument("--validation-positions", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.full_output, args.output) if p]
    if (not outputs or len(set(outputs)) < len(outputs)
            or any(p.suffix != ".pt" or p.exists() for p in outputs)):
        parser.error("Choose distinct, fresh .pt output paths")
    if args.output and not args.data:
        parser.error("--output needs calibration shards in --data")
    if args.export_onnx and (not args.output or args.output.with_suffix(".onnx").exists()):
        parser.error("--export-onnx requires a student --output with a fresh sibling .onnx path")
    if min(args.positions, args.validation_positions, args.batch_size, args.threads) < 1:
        parser.error("sample, batch, and thread counts must be positive")
    if args.data and args.validation and args.data.resolve() == args.validation.resolve():
        parser.error("use separate calibration and validation directories")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    teacher = import_teacher(args.source)
    print(json.dumps({"sourceSha256": SHA256,
                      "fullImportParity": check_parity(teacher, args.source, args.fixtures, args.threads)}),
          flush=True)
    if args.full_output:
        save_checkpoint(teacher, args.full_output)
    if not args.output:
        return
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    states = load_positions(args.data, args.positions, rng, device)[0]
    student = Stage2Net(moderate=args.student_tier == "stage2_moderate").eval()
    print(f"Calibrating {args.student_tier}: {sum(p.numel() for p in student.parameters()):,} parameters",
          flush=True)
    moments = covariance_factors(teacher.to(device), student, states, args.batch_size, device)
    teacher.cpu()
    for name, layer in student.layers.items():
        if name in moments:
            initialize_factors(teacher.layers[name], layer, *moments[name])
        else:
            layer.load_state_dict(teacher.layers[name].state_dict())
    if args.validation:
        states = load_positions(args.validation, args.validation_positions, rng, device)[0]
    print(json.dumps({"beforeFineTuning": agreement(teacher.to(device), student.to(device), states,
                                                    args.batch_size),
                      "agreementData": str(args.validation or args.data),
                      "heldOut": bool(args.validation)}), flush=True)
    if args.export_onnx:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export(student, states[:16], args.output.with_suffix(".onnx"), args.student_tier)
    else:
        save_checkpoint(student, args.output)


if __name__ == "__main__":
    main()
