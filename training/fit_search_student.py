"""Fit a browser HARNet or offline teacher to searched position targets in compact NPZ shards.

Run from the project root with python -m training.fit_search_student --help.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from .harnet import HARNet, build_harnet
from .model import MAX_ONNX_MODEL_BYTES
from .pipeline import _symmetry_maps


EXPORT_LIMITS = {
    "student512k": 512 * 1024,
    "student1_5m": MAX_ONNX_MODEL_BYTES,
    "fast_student1_5m": MAX_ONNX_MODEL_BYTES,
    "line_student": MAX_ONNX_MODEL_BYTES,
    "routing_transformer": MAX_ONNX_MODEL_BYTES,
    "stage2_student": MAX_ONNX_MODEL_BYTES,
    "stage2_moderate": 10 * 1024 * 1024,  # Native float32 model before separate 4-bit packing.
    "board_table": 1_492_760,
    "stage3_teacher": 25 * 1024 * 1024,
    "student5m": 4_429_556,
    "challenger20m": 20 * 1024 * 1024,
    "student25m": 25 * 1024 * 1024,
    "teacher50m": 64 * 1024 * 1024,
    "teacher75m": 96 * 1024 * 1024,
}


def load_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Gen88-style checkpoints add one linear projection per macro board. The
    # two-head evaluation graph has no consumer for these two parameters.
    if isinstance(model, HARNet) and any(key.startswith("ownership_head.") for key in state):
        ownership = {"ownership_head.0.weight": (1, model.macro_channels),
                     "ownership_head.0.bias": (1,)}
        for key, shape in ownership.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError(f"Incompatible ownership projection: expected {key} {shape}")
        state = {key: tensor for key, tensor in state.items() if key not in ownership}
        print("Omitting the checkpoint's auxiliary ownership projection.", flush=True)
    for key, target in model.state_dict().items():
        if key not in state or state[key].shape != target.shape:
            raise ValueError(f"Incompatible checkpoint: expected {key} with shape {tuple(target.shape)}")
    model.load_state_dict(state, strict=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_positions(directory, limit, rng, device, identity=None):
    """Sample positions uniformly over rows, including shards of unequal size."""
    paths = sorted(p for p in directory.glob("*.npz") if not p.name.startswith("."))
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            counts.append(len(shard["v"]))
    total = sum(counts)
    if not total:
        raise ValueError(f"No training positions in {directory}")

    count = min(total, limit)
    selected = (np.arange(total) if count == total
                else np.sort(rng.choice(total, size=count, replace=False)))
    x = np.empty((count, 280), dtype=np.uint8)
    p = np.empty((count, 81), dtype=np.float16)
    v = np.empty(count, dtype=np.float32)
    offset = 0
    start = 0
    for path, rows in zip(paths, counts):
        stop = selected.searchsorted(offset + rows)
        if stop > start:
            indices = selected[start:stop] - offset
            with np.load(path, allow_pickle=False) as shard:
                for key, out in (("x", x), ("p", p), ("v", v)):
                    source = shard[key]
                    if source.shape != (rows, *out.shape[1:]) or source.dtype != out.dtype:
                        raise ValueError(f"{path}: unexpected {key} shape or dtype: "
                                         f"{source.shape}, {source.dtype}")
                    out[start:stop] = source[indices]
        start = stop
        offset += rows

    for first in range(0, count, 8192):
        batch = slice(first, first + 8192)
        policies = p[batch]
        if (x[batch].max() > 1 or not np.isfinite(policies).all()
                or policies.min() < 0 or np.any(policies.sum(axis=1, dtype=np.float32) <= 0)
                or np.any((x[batch, 162:243] == 0) & (policies > 0))
                or not np.isfinite(v[batch]).all() or np.max(np.abs(v[batch])) > 1):
            raise ValueError(f"Invalid state, illegal policy mass, or invalid value in {directory}")
    if identity is not None:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            identity["manifest_sha256"] = file_hash(manifest)
        else:
            inventory = []
            for path in paths:
                info = path.stat()
                inventory.append((path.name, info.st_size, info.st_mtime_ns))
            identity["shards_sha256"] = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
        digest = hashlib.sha256()
        for array in (selected, x, p, v):
            digest.update(memoryview(array).cast("B"))
        identity.update(total_rows=total, sampled_rows=count, sample_sha256=digest.hexdigest())
    print(f"Sampled {count:,}/{total:,} positions from {len(paths)} shards in {directory}",
          flush=True)
    return tuple(torch.from_numpy(a).to(device) for a in (x, p, v))


def target_losses(logits, predicted, x, p, v):
    logits = logits.float().masked_fill(x[:, 162:243] == 0, -1e9)
    p = p.float()
    p = p / p.sum(dim=1, keepdim=True)
    policy_loss = -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    return policy_loss, F.mse_loss(predicted.float().flatten(), v)


@torch.inference_mode()
def validate(model, positions, args, label):
    model.eval()
    states, policies, values = positions
    totals = torch.zeros(2, device=states.device, dtype=torch.float64)
    for start in range(0, len(states), args.batch_size):
        rows = slice(start, start + args.batch_size)
        x = states[rows].float()
        logits, predicted = model(x)
        losses = target_losses(logits, predicted, x, policies[rows], values[rows])
        totals += len(x) * torch.stack(losses).double()
    policy, value = (totals / len(states)).tolist()
    print(f"{label} validation over {len(states):,} positions: policy {policy:.5f}, "
          f"value {value:.5f}, weighted sum {policy + args.value_weight * value:.5f}", flush=True)


@contextmanager
def checkpoint_lock(path):
    if path is None:
        yield
        return
    # Keep the lock inode in place so waiting processes always refer to the same one.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another trainer is using checkpoint {path}") from exc
        yield


def save_checkpoint(path, payload):
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            pending = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def restore_checkpoint(path, identity, model, optimizer, scheduler, device, steps):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != "search-student-resume-v1":
        raise ValueError(f"Unrecognized training checkpoint {path}")
    if state.get("identity") != identity:
        raise ValueError("Resume checkpoint disagrees with the training options, runtime, or sampled data")
    step = state["step"]
    if type(step) is not int or not 1 <= step <= steps or state["scheduler"]["last_epoch"] != step:
        raise ValueError("Resume checkpoint has an inconsistent completed step")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    print(f"Resumed {path} at completed step {step}/{steps}", flush=True)
    return step


def fit(model, primary, replay, args, device, identity=None):
    # Gather uses destination-to-source maps. The shared rules encoder stores the inverse.
    maps = _symmetry_maps()
    x_map, p_map = [torch.as_tensor(np.argsort(np.stack(axis), axis=1), device=device)
                    for axis in zip(*maps)]
    # All rows live on the training device in their compact representation.
    primary_rows = len(primary[0])
    replay_size = round(args.batch_size * args.replay_fraction) if replay else 0
    if replay and not 0 < replay_size < args.batch_size:
        raise ValueError("Replay fraction must give at least one row from each source per batch")
    states, policies, values = (tuple(torch.cat(pair) for pair in zip(primary, replay))
                                if replay else primary)
    if args.value_only:
        # eval() keeps shared BatchNorm buffers fixed. Autograd still records
        # operations in the value head when its parameters require gradients.
        model.eval()
        model.requires_grad_(False)
        model.value_mlp.requires_grad_(True)
        probe = primary[0][:16].float()
        with torch.inference_mode():
            initial_policy = model(probe)[0].contiguous().clone()
    else:
        model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr / 10,
    )
    use_bfloat = device.type == "cuda" and torch.cuda.is_bf16_supported()
    completed = 0
    if args.resume_state is not None and args.resume_state.exists():
        completed = restore_checkpoint(args.resume_state, identity, model, optimizer, scheduler,
                                       device, args.steps)
    for step in range(completed + 1, args.steps + 1):
        indices = torch.randint(primary_rows, (args.batch_size - replay_size,), device=device)
        if replay_size:
            replay_indices = torch.randint(primary_rows, len(states), (replay_size,), device=device)
            indices = torch.cat((indices, replay_indices))
        symmetry = torch.randint(8, (args.batch_size,), device=device)
        x = states[indices].gather(1, x_map[symmetry]).float()
        p = policies[indices].gather(1, p_map[symmetry])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat):
            logits, predicted = model(x)
        policy_loss, value_loss = target_losses(logits, predicted, x, p, values[indices])
        objective = args.value_weight * value_loss
        if not args.value_only:
            objective = policy_loss + objective
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if args.resume_state is not None and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(args.resume_state, {
                "format": "search-student-resume-v1", "identity": identity, "step": step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            })
            print(f"Saved {args.resume_state} at completed step {step}/{args.steps}", flush=True)
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps}: policy {policy_loss.item():.4f}, "
                  f"value {value_loss.item():.4f}", flush=True)
    if args.value_only:
        with torch.inference_mode():
            final_policy = model(probe)[0].contiguous()
            if not torch.equal(initial_policy.view(torch.uint8), final_policy.view(torch.uint8)):
                raise RuntimeError("Policy logits changed during value-only training")
        print(f"Verified byte-identical policy logits on {len(probe)} fixed positions.", flush=True)


def export(model, samples, output, tier):
    model = model.cpu().float().eval()
    training_weights = model.state_dict()
    samples = samples.cpu().float()
    if len(samples) == 1:
        samples = samples.repeat(2, 1)
    if tier == "board_table":
        from .board_table import compile_table, verify_cached_graph
        model, drift = compile_table(model, samples)
        print(f"INT8 lookup vs implicit encoder: {drift}", flush=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        artifact = Path(temporary) / output.name
        torch.onnx.export(
            model, (samples[:1],), str(artifact), input_names=["state"],
            output_names=["policy", "value"], opset_version=14, dynamo=False,
            dynamic_axes={name: {0: "batch"} for name in ("state", "policy", "value")},
        )
        size = artifact.stat().st_size
        limit = EXPORT_LIMITS[tier]
        if size > limit:
            raise ValueError(f"{tier} export is {size:,} bytes, above the {limit:,}-byte limit")
        if tier == "board_table":
            verify_cached_graph(artifact)
        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            for batch in (samples[:1], samples):
                predictions = session.run(["policy", "value"], {"state": batch.numpy()})
                for actual, expected in zip(predictions, model(batch)):
                    if actual.shape != tuple(expected.shape) or not np.isfinite(actual).all():
                        raise ValueError("Invalid policy/value export")
                    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-4, rtol=2e-4)
        weights = artifact.with_suffix(".pt")
        torch.save(training_weights, weights)
        # Hard links publish complete files and preserve another concurrent run's output.
        final_weights = output.with_suffix(".pt")
        os.link(weights, final_weights)
        try:
            os.link(artifact, output)
        except OSError:
            final_weights.unlink()
            raise
    print(f"Verified {output} ({size:,} bytes; SHA-256 "
          f"{hashlib.sha256(output.read_bytes()).hexdigest()})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-fraction", type=float, default=0.2)
    parser.add_argument("--validation", type=Path,
                        help="Independent compact shards for initial/final loss reporting")
    parser.add_argument("--validation-positions", type=int, default=4096)
    parser.add_argument("--max-positions", type=int, default=300_000,
                        help="Maximum sampled rows per corpus resident on the training device")
    parser.add_argument("--initial", type=Path, help="HARNet .pt state dictionary for warm start")
    parser.add_argument("--value-only", action="store_true",
                        help="Train only the warm-started HARNet value head; preserve policy and trunk")
    parser.add_argument("--output", type=Path, required=True, help="Fresh .onnx path; also saves .pt")
    parser.add_argument("--tier", choices=tuple(EXPORT_LIMITS), default="student1_5m")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--resume-state", type=Path,
                        help="Atomically save training state here and resume it when it already exists")
    parser.add_argument("--save-every", type=int,
                        help="Completed steps per checkpoint with --resume-state (default 1000)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.5,
                        help="Coefficient of value MSE in the policy + value objective")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.max_positions, args.validation_positions) < 1 or args.seed < 0:
        parser.error("steps, batch size, and sample limit must be positive, seed nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.replay_fraction < 1:
        parser.error("lr must be positive and finite; replay fraction must be in [0, 1)")
    if not math.isfinite(args.value_weight) or args.value_weight < 0:
        parser.error("value weight must be finite and nonnegative")
    if args.value_only and (not args.initial or args.value_weight == 0):
        parser.error("--value-only requires --initial and a positive --value-weight")
    training_dirs = [args.data] + ([args.replay] if args.replay and args.replay_fraction else [])
    if args.validation and any(args.validation.resolve() == d.resolve() for d in training_dirs):
        parser.error("validation must use a separate directory of withheld shards")
    if args.save_every is not None and (args.save_every < 1 or not args.resume_state):
        parser.error("--save-every requires --resume-state and a positive interval")
    if args.resume_state is not None:
        args.resume_state = args.resume_state.resolve()
        protected = [args.output, args.output.with_suffix(".pt")]
        if args.initial:
            protected.append(args.initial)
        if args.resume_state in {path.resolve() for path in protected}:
            parser.error("--resume-state needs a separate path from --output and --initial")
        data_dirs = training_dirs + ([args.validation] if args.validation else [])
        if (args.resume_state.parent in {d.resolve() for d in data_dirs}
                and (args.resume_state.suffix == ".npz" or args.resume_state.name == "manifest.json")):
            parser.error("--resume-state must be separate from training shards and their manifests")
        args.save_every = args.save_every or 1000
    with checkpoint_lock(args.resume_state):
        run_training(args, parser)


def run_training(args, parser):
    if (args.output.suffix != ".onnx" or args.output.exists()
            or args.output.with_suffix(".pt").exists()):
        parser.error("--output must name fresh .onnx and sibling .pt files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_state:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    identity = None
    if args.resume_state:
        options = ("tier", "seed", "steps", "batch_size", "lr", "value_only", "value_weight",
                   "replay_fraction", "max_positions", "validation_positions")
        identity = {key: getattr(args, key) for key in options}
        identity.update(initial_sha256=file_hash(args.initial) if args.initial else None,
                        device=device.type, torch_version=str(torch.__version__),
                        torch_threads=torch.get_num_threads())
        if device.type == "cuda":
            identity.update(cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                            gpu=torch.cuda.get_device_name(device), bf16=torch.cuda.is_bf16_supported(),
                            cublas_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.tier == "line_student":
        from .line_student import LineStudent
        model = LineStudent()
    elif args.tier == "routing_transformer":
        from .routing_transformer import RoutingTransformer
        model = RoutingTransformer()
    elif args.tier == "board_table":
        from .board_table import BoardTable
        model = BoardTable()
    elif args.tier in ("stage2_student", "stage2_moderate", "stage3_teacher"):
        from .stage2_student import Stage2Net
        model = Stage2Net(compact=args.tier != "stage3_teacher", moderate=args.tier == "stage2_moderate")
    else:
        model = build_harnet(args.tier)
    if args.value_only and not isinstance(model, HARNet):
        parser.error("--value-only requires a HARNet with an independent value_mlp")
    # The Gen69 ONNX graph adds ~16 KiB to the parameter payload. This is a
    # preflight estimate; export enforces the exact standalone file size.
    estimated = sum(p.numel() * p.element_size() for p in model.parameters()) + 20 * 1024
    if args.tier == "board_table":
        estimated = model.estimated_export_bytes()
    if estimated > EXPORT_LIMITS[args.tier]:
        parser.error(f"estimated {args.tier} export of {estimated:,} bytes exceeds its limit")
    print(f"Estimated export {estimated:,} bytes; {args.tier} limit {EXPORT_LIMITS[args.tier]:,}",
          flush=True)
    if args.initial:
        load_weights(model, args.initial)
    model.to(device)
    print(f"Training {args.tier} on {device} from {args.initial or 'random initialization'} "
          f"with seed {args.seed}, value weight {args.value_weight}", flush=True)
    def corpus(name, directory, limit):
        source_identity = None if identity is None else identity.setdefault(name, {})
        return load_positions(directory, limit, rng, device, source_identity)

    primary = corpus("primary", args.data, args.max_positions)
    replay = None
    if args.replay and args.replay_fraction:
        replay = corpus("replay", args.replay, args.max_positions)
    if args.validation:
        validation = corpus("validation", args.validation, args.validation_positions)
        validate(model, validation, args, "Initial")
    fit(model, primary, replay, args, device, identity)
    if args.validation:
        validate(model, validation, args, "Final")
    samples = primary[0][:16]
    if args.tier == "board_table":
        # Exercise the table on positions spread over the withheld shards when supplied.
        states = validation[0] if args.validation else primary[0]
        count = min(len(states), 1024)
        rows = torch.arange(count, device=states.device) * len(states) // count
        samples = states[rows]
    export(model, samples, args.output, args.tier)


if __name__ == "__main__":
    main()
