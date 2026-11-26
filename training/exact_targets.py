"""Prove W/D/L for every legal move in a bounded sample of endgame positions.

Example: python -m training.exact_targets --data replay/ --output exact/ --threads 2
Only completed roots are exported. --max-positions limits attempted eligible roots.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .distill_oracle import validate


def eligible(states, maximum):
    if states.ndim != 2 or states.shape[1:] != (280,) or states.dtype != np.uint8:
        raise ValueError("Expected a compact 280-feature uint8 shard")
    open_boards = ~states[:, 243:270].reshape(-1, 3, 9).any(1)
    empty = (states[:, :81] | states[:, 81:162]) == 0
    remaining = (empty.reshape(-1, 9, 9).sum(2) * open_boards).sum(1)
    return np.flatnonzero((remaining <= maximum) & states[:, 162:243].any(1))


def root_rows(states):
    bit = 1 << np.arange(9, dtype=np.int32)
    own = states[:, :81].reshape(-1, 9, 9) @ bit
    other = states[:, 81:162].reshape(-1, 9, 9) @ bit
    masks = np.stack([own, other], axis=2).reshape(-1, 18)
    return np.column_stack([states[:, 270:].argmax(1), masks])


def send_roots(worker, states):
    rows = root_rows(states)
    lines = [str(len(rows)), *(" ".join(map(str, row)) for row in rows)]
    worker.stdin.write("\n".join(lines) + "\n")
    worker.stdin.flush()


def position_values(worker, states):
    """Read only proved side-to-move values from a solver launched with --value."""
    send_roots(worker, states)
    values, selected = [], []
    nodes = 0
    for index in range(len(states)):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"Endgame worker stopped with exit code {worker.poll()}")
        count, token = line.split()
        nodes += int(count)
        if token == "?":
            continue
        if token not in ("L", "D", "W"):
            raise ValueError("Expected a single proved position value")
        selected.append(index)
        values.append({"L": -1, "D": 0, "W": 1}[token])
    return np.asarray(selected, dtype=int), np.asarray(values, dtype=np.int8), nodes


def classify(worker, states):
    send_roots(worker, states)
    total_nodes = 0
    solved, classes = [], []
    for index in range(len(states)):
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"Endgame worker stopped with exit code {worker.poll()}")
        node_count, result = line.split()
        total_nodes += int(node_count)
        if result == "?":
            continue
        data = np.frombuffer(result.encode("ascii"), dtype=np.uint8)
        if len(data) != 81 or not np.isin(data, list(b".LDW")).all():
            raise RuntimeError("Invalid solver reply")
        if not np.array_equal(data != ord("."), states[index, 162:243]):
            raise RuntimeError("Solver and input disagree on the legal moves")
        solved.append(index)
        classes.append(data)
    return np.array(solved, dtype=int), np.array(classes, np.uint8).reshape(-1, 81), total_nodes


def proved_targets(classes, prior, uniform_weight):
    # '.' is below loss. Solved-losing states keep all legal moves in support.
    values = np.select([classes == ord("W"), classes == ord("D"), classes == ord("L")],
                       [1, 0, -1], default=-2)
    value = values.max(1)
    optimal = values == value[:, None]
    uniform = optimal / optimal.sum(axis=1, keepdims=True)
    masked = prior.astype(np.float64) * optimal
    total = masked.sum(1, keepdims=True)
    restricted = np.divide(masked, total, out=uniform.copy(), where=total > 0)
    policy = uniform_weight * uniform + (1 - uniform_weight) * restricted
    return policy.astype(np.float16), value.astype(np.float32)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generate(args):
    paths = sorted(args.data.glob("*.npz")) if args.data.is_dir() else [args.data]
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            counts.append(len(eligible(a["x"], args.max_remaining)))
    total = sum(counts)
    if not total:
        raise ValueError("No eligible positions")
    limit = min(args.max_positions, total)
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(total, size=limit, replace=False)) if limit < total else np.arange(total)
    report = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report.update(format="uttt-exact-wdl-v1", eligible=total, attempted=limit, positions=0,
                  capped=0, nodes=0, wins=0, draws=0, losses=0, sources=[], shards=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        temp = Path(directory)
        solver = args.solver
        if solver is None:
            solver = temp / "exact-targets"
            source = Path(__file__).with_suffix(".cpp")
            subprocess.run(["c++", "-std=c++17", "-O3", "-pthread", str(source), "-o", str(solver)], check=True)
            report["solver_source_sha256"] = sha(source)
        report["solver_sha256"] = sha(solver)
        output = temp / "output"
        output.mkdir()
        with subprocess.Popen([str(solver.resolve()), str(args.nodes), str(args.threads)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
            try:
                done = offset = 0
                for shard, (path, count) in enumerate(zip(paths, counts)):
                    end = int(picks.searchsorted(offset + count))
                    local = picks[done:end] - offset
                    offset += count
                    if not len(local):
                        continue
                    with np.load(path, allow_pickle=False) as a:
                        states = a["x"]
                        candidates = eligible(states, args.max_remaining)
                        if len(candidates) != count:
                            raise ValueError(f"Shard changed while sampling: {path}")
                        selected = candidates[local]
                        x, p, v = states[selected], a["p"][selected], a["v"][selected]
                    selected_x, labels_p, labels_v = [], [], []
                    for first in range(0, len(x), 256):
                        batch = slice(first, first + 256)
                        validate(x[batch], p[batch], v[batch])
                        kept, classes, nodes = classify(worker, x[batch])
                        report["nodes"] += nodes
                        report["capped"] += len(x[batch]) - len(kept)
                        if len(kept):
                            new_p, new_v = proved_targets(classes, p[first + kept], args.uniform_weight)
                            selected_x.append(x[first + kept])
                            labels_p.append(new_p)
                            labels_v.append(new_v)
                    report["sources"].append({"file": path.name, "sha256": sha(path), "eligible": count,
                                              "attempted": len(x)})
                    if selected_x:
                        name = f"shard-{shard:05d}.npz"
                        values = np.concatenate(labels_v)
                        np.savez_compressed(output / name, x=np.concatenate(selected_x),
                                            p=np.concatenate(labels_p), v=values)
                        report["positions"] += len(values)
                        for field, score in (("wins", 1), ("draws", 0), ("losses", -1)):
                            report[field] += int((values == score).sum())
                        report["shards"].append({"file": name, "positions": len(values)})
                    done = end
                    print(f"Attempted {done:,}/{limit:,}, proved {report['positions']:,}, "
                          f"capped {report['capped']:,}", flush=True)
                worker.stdin.close()
                if worker.wait() != 0:
                    raise RuntimeError("Endgame worker failed")
            finally:
                if worker.poll() is None:
                    worker.kill()
        if not report["positions"]:
            raise ValueError(f"No fully proved roots out of {limit:,} attempts with cap {args.nodes:,}")
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        output.rename(args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", type=Path, help="Precompiled exact_targets.cpp executable")
    parser.add_argument("--max-positions", type=int, default=50_000)
    parser.add_argument("--max-remaining", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=50_000, help="Shared node budget across all actions of one root")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--uniform-weight", type=float, default=.05,
                        help="Uniform mass over proved-optimal moves, mixed with their normalized original prior")
    args = parser.parse_args()
    if args.max_positions <= 0 or not 1 <= args.max_remaining <= 28 or not 1 <= args.nodes <= 2**63-1:
        parser.error("Use positive limits with at most 28 remaining squares and a signed 64-bit node budget")
    if not 1 <= args.threads <= 64 or args.seed < 0 or not 0 <= args.uniform_weight <= 1:
        parser.error("Invalid thread count, seed or uniform mixture")
    generate(args)


if __name__ == "__main__":
    main()
