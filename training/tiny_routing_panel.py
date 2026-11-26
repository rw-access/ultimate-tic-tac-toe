"""Mine actual Tiny search choices and evaluate a frozen independent official-search panel."""

import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.reanalyse_positions import response, targets
from training.tiny_midgame import INCUMBENT, action_source, all_views, best_moves, plausible
from training.tiny_proof_consensus import digest, packed
from training.pipeline import _symmetry_maps


def freeze(args):
    source, h = action_source(args.reference, 100000, args.visits, np.random.default_rng(1))
    with np.load(args.fresh, allow_pickle=False) as f:
        raw = f["x"]
    if not np.isin(packed(source["x"]), packed(raw)).all():
        raise ValueError("Official panel contains positions outside the fresh source")
    blocks = np.load(args.data / "excluded.npy", allow_pickle=False)
    candidates = np.flatnonzero(plausible(source["x"], source["n"]))
    rng = np.random.default_rng(args.seed)
    indices = []
    seen = set()
    views = [packed(source["x"][candidates][:, m]) for m, _ in _symmetry_maps()]
    for j in rng.permutation(len(candidates)):
        i = candidates[j]
        key = min(arr[j].tobytes() for arr in views)
        if key not in seen:
            seen.add(key)
            indices.append(i)
        if len(indices) == args.count:
            break
    if not indices:
        raise ValueError("No unique, eligible held-out routing positions")
    picked = {k: a[indices] for k, a in source.items()}
    if not np.isin(packed(picked["x"]), blocks).all():
        raise ValueError("Panel was not reserved before training")
    with np.load(args.data / "roots.npz", allow_pickle=False) as training:
        if np.isin(packed(training["x"]), all_views(picked["x"])).any():
            raise ValueError("Routing panel leaked into training")
    with args.output.open("xb") as f:
        np.savez_compressed(
            f,
            **picked,
            reference_manifest=np.asarray(h),
            reference_visits=np.int32(args.visits),
            fresh_sha=np.asarray(digest(args.fresh)),
        )
    print(
        "Frozen routing panel",
        len(indices),
        "from eligible",
        len(candidates),
        digest(args.output),
        flush=True,
    )


def search(args):
    pin = digest(args.model)
    if pin != args.sha256:
        raise ValueError("Model digest mismatch")
    with np.load(args.input, allow_pickle=False) as f:
        x = f["x"]
    cmd = [
        str(args.node),
        "--experimental-strip-types",
        str(args.worker),
        "--model",
        str(args.model),
        "--simulations",
        "256",
        "--parallel",
        "48",
        "--seed",
        str(args.seed),
    ]
    choices = []
    visits = []
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as process:
        hello = response(process, "ready")
        if (hello["modelSha256"], hello["simulations"], hello["parallel"], hello["seed"]) != (
            pin,
            256,
            48,
            args.seed,
        ):
            raise ValueError("Search model, budget, batch width, or seed differs from the request")
        for i in range(0, len(x), 48):
            chunk = x[i : i + 48]
            request = {
                "ids": list(range(i, i + len(chunk))),
                "x": base64.b64encode(chunk.tobytes()).decode(),
            }
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            reply = response(process, "targets")
            rows = reply["targets"]
            _, _, completed, _ = targets(rows, chunk, 256)
            choice = np.asarray([item["move"] for item in rows])
            if (
                choice.dtype.kind not in "iu"
                or choice.shape != (len(chunk),)
                or np.any((choice < 0) | (choice >= 81))
                or not chunk[np.arange(len(chunk)), 162 + choice].all()
            ):
                raise ValueError("Illegal selected move")
            choices.extend(choice.tolist())
            visits.extend(completed.tolist())
            if (i // 48) % 6 == 0:
                print("Fixed-visit choices", len(choices), "/", len(x), flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Fixed-work search process failed")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])
    bare = []
    for i in range(0, len(x), 48):
        y = x[i : i + 48]
        logits = session.run(["policy"], {"state": y.astype(np.float32)})[0]
        if logits.shape != (len(y), 81) or not np.isfinite(logits).all():
            raise ValueError("Bare inference produced invalid logits")
        bare.extend(np.where(y[:, 162:243], logits, -np.inf).argmax(1).tolist())
    with args.output.open("xb") as f:
        np.savez_compressed(
            f,
            move=np.asarray(choices, np.uint8),
            bare=np.asarray(bare, np.uint8),
            visits=np.asarray(visits, np.uint16),
            model_sha=np.asarray(pin),
            input_sha=np.asarray(digest(args.input)),
            seed=np.int64(args.seed),
        )
    print("Scored", len(choices), digest(args.output), flush=True)


def score(args):
    with np.load(args.panel, allow_pickle=False) as f:
        x, n, q, v = [f[k] for k in ("x", "n", "q", "v")]
        reference_visits = int(f["reference_visits"])
    best, backed, support = best_moves(n, q, v)
    i = np.arange(len(x))
    best_q = backed[i, best]
    counts = n.max(1)
    result = {
        "panels": len(i),
        "panelSha256": digest(args.panel),
        "referenceVisitsPerRoot": reference_visits,
        "minOfficialActionVisits": 32,
        "moveQShrink": 64,
        "models": {},
        "paired": {},
    }
    base = None
    for file in args.choices:
        with np.load(file, allow_pickle=False) as f:
            if str(f["input_sha"].item()) != result["panelSha256"]:
                raise ValueError("Choices belong to a different panel")
            pin = str(f["model_sha"].item())
            actions = {k: f[k].astype(int) for k in ("bare", "move")}
            observed = f["visits"].astype(int)
            seed = int(f["seed"])
        if pin in result["models"]:
            raise ValueError("Each contestant's model hash must be distinct")
        if base is None and pin != INCUMBENT:
            raise ValueError("First contestant must be the incumbent")
        if observed.shape != (len(x),) or not np.isin(observed, [0, 256]).all():
            raise ValueError("Invalid completed search budget")
        for action in actions.values():
            if (
                action.shape != (len(x),)
                or np.any((action < 0) | (action >= 81))
                or not x[i, 162 + action].all()
            ):
                raise ValueError("Recorded action is illegal")
        if "seed" in result and seed != result["seed"]:
            raise ValueError("Contestants have different search seeds")
        result["seed"] = seed
        metrics = {}
        pairs = {}
        for label, a in actions.items():
            ok = support[i, a]
            agree = n[i, a] == counts
            regret = best_q - backed[i, a]
            metrics[label] = {
                "topVisitAgreement": int(agree.sum()),
                "withSupportedAction": int(ok.sum()),
                "meanBackedRegretOnSupported": float(regret[ok].mean()),
                "meanVisitShare": float((n[i, a] / n.sum(1)).mean()),
            }
            if base is not None:
                old = base[label]
                common = ok & support[i, old]
                change = (backed[i, a] - backed[i, old])[common]
                old_agree = n[i, old] == counts
                rng = np.random.default_rng(26316731)
                bootstrap = change[rng.integers(len(change), size=(4096, len(change)))].mean(1)
                pairs[label] = {
                    "commonSupported": int(common.sum()),
                    "meanParentQGain": float(change.mean()),
                    "meanCandidateRegret": float(regret[common].mean()),
                    "meanIncumbentRegret": float((best_q - backed[i, old])[common].mean()),
                    "pairedGain95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "visitCorrections": int((agree & ~old_agree).sum()),
                    "visitRegressions": int((~agree & old_agree).sum()),
                    "changedMoves": int((a != old).sum()),
                }
        result["models"][pin] = dict(
            metrics, work=int(observed.sum()), immediateWins=int((observed == 0).sum())
        )
        if pairs:
            result["paired"][pin] = pairs
        if base is None:
            base = actions
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


def episode_clusters(path, states):
    """Bind every panel state to its source game or paired opening by full-state equality."""
    keys = [row.tobytes() for row in packed(states)]
    required = set(keys)
    assigned = {}

    def assign(features, group):
        for row in packed(np.asarray(features, dtype=np.uint8).reshape(-1, 280)):
            key = row.tobytes()
            if key in required:
                if key in assigned and assigned[key] != group:
                    raise ValueError("A panel state belongs to more than one episode")
                assigned[key] = group

    if path.suffix == ".jsonl":
        unit = "self-play game"
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                assign([sample[0] for sample in record["samples"]], record["game"])
    else:
        from training.opponent_replay import trajectory

        unit = "opponent-game opening pair"
        report = json.loads(path.read_bytes())
        if report.get("format") != "official-uttt-arena-v1" or report.get("status") != "complete":
            raise ValueError("Expected a complete original-engine game record")
        for pair in report["results"]:
            for game in pair["games"]:
                samples, _ = trajectory(game, pair["opening"], 12)
                assign([state for state, _ in samples], pair["pair"])
    if assigned.keys() != required:
        raise ValueError("Panel position absent from the original game records")
    return unit, np.asarray([assigned[key] for key in keys])


def clustered(args):
    with np.load(args.panel, allow_pickle=False) as panel:
        x, n, q, v = (panel[key] for key in ("x", "n", "q", "v"))
    unit, groups = episode_clusters(args.episodes, x)
    _, backed, supported = best_moves(n, q, v)
    pin = digest(args.panel)
    actions = []
    shas = []
    seeds = []
    for file in (args.incumbent, args.candidate):
        with np.load(file, allow_pickle=False) as src:
            if str(src["input_sha"].item()) != pin:
                raise ValueError("Search output belongs to another panel")
            actions.append({key: src[key].astype(int) for key in ("bare", "move")})
            shas.append(str(src["model_sha"].item()))
            seeds.append(int(src["seed"]))
    if shas[0] != INCUMBENT or shas[0] == shas[1] or seeds[0] != seeds[1]:
        raise ValueError("Expected incumbent and challenger searched at one common seed")
    rows = np.arange(len(x))
    result = dict(
        panelSha256=pin,
        incumbentSha256=shas[0],
        candidateSha256=shas[1],
        seed=seeds[0],
        groupingUnit=unit,
        groupCount=len(np.unique(groups)),
        rawGameRecordSha256=digest(args.episodes),
        metrics={},
    )
    for key in ("bare", "move"):
        old, new = (table[key] for table in actions)
        use = supported[rows, old] & supported[rows, new]
        eligible = rows[use]
        if not len(eligible):
            raise ValueError("No commonly supported comparisons")
        distinct, indexes = np.unique(groups[use], return_inverse=True)
        if len(distinct) < 2:
            raise ValueError("At least two independent games or openings are required")
        gains = backed[eligible, new[use]] - backed[eligible, old[use]]
        sums = np.bincount(indexes, weights=gains)
        sizes = np.bincount(indexes)
        rng = np.random.default_rng(26316731)
        sampled = rng.integers(len(distinct), size=(8192, len(distinct)))
        means = sums[sampled].sum(1) / sizes[sampled].sum(1)
        result["metrics"][key] = dict(
            rows=int(use.sum()),
            clusters=len(distinct),
            meanParentQGain=float(gains.mean()),
            pairedCluster95=np.quantile(means, [0.025, 0.975]).tolist(),
        )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    freeze_p = s.add_parser("freeze")
    search_p = s.add_parser("search")
    score_p = s.add_parser("score")
    for k in ("reference", "fresh", "data", "output"):
        freeze_p.add_argument("--" + k, type=Path, required=True)
    freeze_p.add_argument("--count", type=int, default=512)
    freeze_p.add_argument("--seed", type=int, default=26316711)
    freeze_p.add_argument("--visits", type=int, choices=(3072, 8192), default=3072)
    for k in ("node", "worker", "model", "input", "output"):
        search_p.add_argument("--" + k, type=Path, required=True)
    search_p.add_argument("--sha256", required=True)
    search_p.add_argument("--seed", type=int, default=26316712)
    score_p.add_argument("--panel", type=Path, required=True)
    score_p.add_argument("--choices", type=Path, nargs="+", required=True)
    score_p.add_argument("--output", type=Path, required=True)
    cluster_p = s.add_parser("cluster")
    for name in ("panel", "episodes", "incumbent", "candidate", "output"):
        cluster_p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    {"freeze": freeze, "search": search, "score": score, "cluster": clustered}[a.mode](a)


if __name__ == "__main__":
    main()
