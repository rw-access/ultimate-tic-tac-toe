"""Relabel compact replay shards with a policy/value teacher and optional one-ply search.

For a legal action a, target policy is proportional to teacher_prior(a) *
exp(-child_value(a) / temperature). Its expected action value is blended with
the teacher's root value and the input labels. A provable win overrides both.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


LINES = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
                  [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])


def won(marks):
    return marks[..., LINES].all(axis=-1).any(axis=-1)


def legal_mask(states, closed, terminal):
    allowed = ~closed & (states[:, 270:279] | states[:, 279:280]).astype(bool)
    empty = ~(states[:, :81] | states[:, 81:162]).astype(bool)
    return empty & np.repeat(allowed, 9, axis=1) & ~terminal[:, None]


def validate(states, policy, value):
    if states.ndim != 2 or states.shape[1] != 280 or states.dtype != np.uint8 or (states > 1).any():
        raise ValueError("Expected 280 binary uint8 features per root")
    if policy.shape != (len(states), 81) or value.shape != (len(states),):
        raise ValueError("Expected 81 policy targets and one value per root")
    own = states[:, :81].reshape(-1, 9, 9)
    other = states[:, 81:162].reshape(-1, 9, 9)
    if (own & other).any() or not np.isin(other.sum((1, 2), dtype=int) - own.sum((1, 2), dtype=int), [0, 1]).all():
        raise ValueError("Invalid mark counts")
    own_won, other_won = won(own), won(other)
    drawn = (own | other).all(-1) & ~own_won & ~other_won
    closed = own_won | other_won | drawn
    if ((own_won & other_won).any() or not np.array_equal(
            np.concatenate([own_won, other_won, drawn], axis=1), states[:, 243:270])):
        raise ValueError("Mini-board status disagrees with marks")
    if (states[:, 270:].sum(1) != 1).any() or (closed & states[:, 270:279]).any():
        raise ValueError("Invalid forced-board marker")
    terminal = won(own_won) | won(other_won) | closed.all(1)
    legal = legal_mask(states, closed, terminal)
    if terminal.any() or not np.array_equal(legal, states[:, 162:243]):
        raise ValueError("Expected live roots with a complete legal-move mask")
    if (not np.isfinite(policy).all() or (policy < 0).any() or not (policy.sum(1) > 0).all()
            or (policy[~legal] != 0).any() or not (np.abs(value) <= 1).all()):
        raise ValueError("Invalid input policy or value targets")


def children(states, rows, moves):
    """Vectorized equivalent of encodeState(play(state, board, cell))."""
    root = states[rows]
    i = np.arange(len(rows))
    board, cell = moves // 9, moves % 9
    if not root[i, 162 + moves].all():
        raise ValueError("Attempted an illegal child move")
    child = np.zeros_like(root)
    child[:, :81] = root[:, 81:162]
    child[:, 81:162] = root[:, :81]
    child[i, 81 + moves] = 1
    local = child[:, 81:162].reshape(-1, 9, 9)[i, board]
    has_line = won(local)
    occupied = local | child[:, :81].reshape(-1, 9, 9)[i, board]
    child[:, 243:252] = root[:, 252:261]
    child[:, 252:261] = root[:, 243:252]
    child[:, 261:270] = root[:, 261:270]
    child[i, 252 + board] = has_line
    child[i, 261 + board] = occupied.all(1) & ~has_line
    closed = (child[:, 243:252] | child[:, 252:261] | child[:, 261:270]).astype(bool)
    win = won(child[:, 252:261])
    terminal = win | closed.all(1)
    child[i, 270 + np.where(closed[i, cell], 9, cell)] = 1
    child[:, 162:243] = legal_mask(child, closed, terminal)
    return child, terminal, win


def distribution(logits, allowed):
    masked = np.where(allowed, logits, -np.inf)
    exp = np.exp(masked - masked.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def symmetry_evaluator(infer, batch_size):
    """Average the eight D4 views while keeping each backend call within its row cap."""
    from .pipeline import _symmetry_maps

    maps = _symmetry_maps()
    # Maps are source -> destination: gather with the inverse to transform a
    # state, and gather with the forward policy map to return to source order.
    input_indices = np.argsort(np.stack([features for features, _ in maps]), axis=1)
    policy_indices = np.stack([moves for _, moves in maps])

    def averaged(states, value_only=False):
        values = np.zeros(len(states), dtype=np.float64)
        probabilities = None if value_only else np.zeros((len(states), 81), dtype=np.float64)
        for start in range(0, len(states) * 8, batch_size):
            indices = np.arange(start, min(start + batch_size, len(states) * 8))
            rows, views = np.divmod(indices, 8)
            transformed = states[rows[:, None], input_indices[views]]
            if value_only:
                view_values = infer(transformed, value_only=True)
            else:
                logits, view_values = infer(transformed)
                legal = transformed[:, 162:243].astype(bool)
                view_p = distribution(logits.astype(np.float64), legal)
                aligned = np.take_along_axis(view_p, policy_indices[views], axis=1)
                np.add.at(probabilities, rows, aligned)
            np.add.at(values, rows, view_values)
        values = (values / 8).astype(np.float32)
        if value_only:
            return values
        # targets() consumes logits, then applies any one-ply tilt and legal
        # mask. Log of the arithmetic probability mixture gives the same prior.
        # Finite sentinels also cover underflowed or illegal probabilities.
        logits = np.log(np.maximum(probabilities / 8, np.finfo(np.float64).tiny))
        return logits.astype(np.float32), values

    return averaged


def targets(x, p, v, infer, batch_size, lookahead, temperature, backup_weight, original_weight):
    validate(x, p, v)
    logits, root_v = infer(x)
    legal = x[:, 162:243].astype(bool)
    allowed = legal
    solved = np.zeros(len(x), dtype=bool)
    if lookahead:
        rows, moves = np.nonzero(legal)
        qs = np.zeros_like(logits)
        exact = np.zeros_like(legal)
        for start in range(0, len(rows), batch_size):
            r, m = rows[start:start + batch_size], moves[start:start + batch_size]
            child, terminal, win = children(x, r, m)
            q = win.astype(np.float32)
            if not terminal.all():
                q[~terminal] = -infer(child[~terminal], value_only=True)
            qs[r, m] = q
            exact[r, m] = terminal
        immediate_win = exact & (qs == 1)
        solved = immediate_win.any(1) | (exact == legal).all(1)
        # If every legal child terminates, the best outcome is exact.
        best = np.where(legal, qs, -np.inf).max(axis=1)
        allowed = np.where(solved[:, None], exact & (qs == best[:, None]), legal)
        logits = logits + qs / temperature
    teacher_p = distribution(logits, allowed)
    teacher_v = root_v
    if lookahead:
        backed_up = (teacher_p * qs).sum(1)
        teacher_v = (1 - backup_weight) * root_v + backup_weight * backed_up
    original_p = p.astype(np.float32)
    original_p /= original_p.sum(1, keepdims=True)
    policy = (1 - original_weight) * teacher_p + original_weight * original_p
    value = (1 - original_weight) * teacher_v + original_weight * v
    if solved.any():
        policy[solved], value[solved] = teacher_p[solved], best[solved]
    return policy.astype(np.float16), value.astype(np.float32), int(solved.sum())


def ort_evaluator(path, device, threads):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers.insert(0, "CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise ValueError(f"Requested {providers[0]}, got {session.get_providers()}")
    if [i.name for i in session.get_inputs()] != ["state"] or not {"policy", "value"} <= {
        i.name for i in session.get_outputs()
    }:
        raise ValueError("Expected a native project policy/value ONNX model")

    def infer(x, value_only=False):
        outputs = session.run(["value"] if value_only else ["policy", "value"],
                              {"state": np.ascontiguousarray(x, dtype=np.float32)})
        return checked_outputs(outputs, len(x), value_only)
    return infer


def checked_outputs(outputs, rows, value_only):
    value = outputs[-1]
    if value.shape != (rows, 1) or not (np.abs(value) <= 1.00001).all():
        raise ValueError("Invalid teacher value output")
    value = np.clip(value[:, 0], -1, 1)
    if value_only:
        return value
    policy = outputs[0]
    if policy.shape != (rows, 81) or not np.isfinite(policy).all():
        raise ValueError("Invalid teacher policy output")
    return policy, value


@contextmanager
def evaluator(args):
    if args.device != "node-cuda":
        yield ort_evaluator(args.teacher, args.device, args.threads)
        return
    command = [str(args.node), str(Path(__file__).with_name("oracle_ort.cjs")),
               str(args.teacher.resolve()), str(args.node_modules.resolve())]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
        try:
            if worker.stdout.read(4) != b"UO01":
                raise RuntimeError("Node CUDA worker failed to initialize; see stderr")

            def infer(x, value_only=False):
                worker.stdin.write(struct.pack("<II", len(x), value_only))
                worker.stdin.write(np.ascontiguousarray(x, dtype=np.uint8).tobytes())
                worker.stdin.flush()
                outputs = []
                for width in ((1,) if value_only else (81, 1)):
                    size = 4 * len(x) * width
                    data = worker.stdout.read(size)
                    if len(data) != size:
                        raise RuntimeError("Node CUDA worker returned an incomplete batch; see stderr")
                    outputs.append(np.frombuffer(data, dtype="<f4").reshape(len(x), width))
                return checked_outputs(outputs, len(x), value_only)

            yield infer
            worker.stdin.close()
            if worker.wait() != 0:
                raise RuntimeError("Node CUDA worker exited unsuccessfully")
        finally:
            if worker.poll() is None:
                worker.kill()


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def relabel(args, infer):
    if getattr(args, "symmetry_views", 1) == 8:
        infer = symmetry_evaluator(infer, args.batch_size)
    paths = sorted(args.data.glob("*.npz")) if args.data.is_dir() else [args.data]
    sizes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            sizes.append(len(shard["v"]))
    total = sum(sizes)
    if not total:
        raise ValueError("No input training positions")
    if args.output.exists():
        raise ValueError("Output directory must be new")
    rng = np.random.default_rng(args.seed)
    n = min(total, args.max_positions)
    selected = np.sort(rng.choice(total, size=n, replace=False)) if n < total else np.arange(total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(teacher_sha256=checksum(args.teacher), source_rows=total, positions=n,
                    solved=0, sources=[], shards=[])
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staging = Path(temp) / "output"
        staging.mkdir()
        offset = done = 0
        for index, (path, size) in enumerate(zip(paths, sizes)):
            end = int(np.searchsorted(selected, offset + size))
            picks = selected[done:end] - offset
            offset += size
            if not len(picks):
                continue
            with np.load(path, allow_pickle=False) as shard:
                x, p, v = [shard[key][picks] for key in ("x", "p", "v")]
            output_p = np.empty((len(x), 81), dtype=np.float16)
            output_v = np.empty(len(x), dtype=np.float32)
            for start in range(0, len(x), args.batch_size):
                batch = slice(start, start + args.batch_size)
                output_p[batch], output_v[batch], exact = targets(
                    x[batch], p[batch], v[batch], infer, args.batch_size, args.lookahead,
                    args.temperature, args.backup_weight, args.original_weight)
                manifest["solved"] += exact
            name = f"shard-{index:05d}.npz"
            np.savez_compressed(staging / name, x=x, p=output_p, v=output_v)
            manifest["sources"].append({"file": path.name, "sha256": checksum(path), "rows": size})
            manifest["shards"].append({"file": name, "positions": len(picks)})
            done = end
            print(f"Relabeled {done:,}/{n:,} positions", flush=True)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(args.output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Compact NPZ shard or shard directory")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, choices=(0, 1), default=1)
    parser.add_argument("--symmetry-views", type=int, choices=(1, 8), default=1,
                        help="Average all eight board orientations for teacher priors and values, or use one view")
    parser.add_argument("--temperature", type=float, default=0.18, help="Scale of prior-guided one-ply value tilt")
    parser.add_argument("--backup-weight", type=float, default=0.65, help="Fraction of expected child value in teacher value")
    parser.add_argument("--original-weight", type=float, default=0.25, help="Fraction of input policy and value retained")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--device", choices=("cpu", "cuda", "node-cuda"), default="cpu")
    parser.add_argument("--node", type=Path, default=Path("node"), help="Node executable for node-cuda")
    parser.add_argument("--node-modules", type=Path, default=Path("node_modules"), help="Directory containing onnxruntime-node")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.max_positions, args.threads) < 1 or args.seed < 0:
        parser.error("batch size, sample count and threads must be positive; seed must be nonnegative")
    if not 0 <= args.backup_weight <= 1 or not 0 <= args.original_weight <= 1:
        parser.error("mixing weights must be within [0, 1]")
    if not math.isfinite(args.temperature) or not 1e-5 <= args.temperature:
        parser.error("temperature must be finite and at least 1e-5")
    if args.device == "node-cuda" and args.batch_size > 65536:
        parser.error("Node worker supports up to 65,536 positions per inference batch")
    with evaluator(args) as infer:
        manifest = relabel(args, infer)
    print(f"Wrote {manifest['positions']:,} positions ({manifest['solved']:,} solved) to {args.output}")


if __name__ == "__main__":
    main()
