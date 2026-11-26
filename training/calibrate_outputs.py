"""Fit and validate output calibration for line students and full Stage 2 teachers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_files(paths, seed):
    if len(paths) < 2:
        raise ValueError("Need at least two disjoint nonempty shard files")
    shuffled = np.random.default_rng(seed).permutation(len(paths))
    count = min(len(paths) - 1, max(2 if len(paths) >= 4 else 1, round(len(paths) / 3)))
    heldout = set(shuffled[:count])
    return [x for i, x in enumerate(paths) if i not in heldout], [x for i, x in enumerate(paths) if i in heldout]


def sample(paths, sizes, limit, seed):
    from training.distill_oracle import validate
    counts = [sizes[p] for p in paths]
    total = sum(counts)
    ids = np.sort(np.random.default_rng(seed).choice(total, min(total, limit), replace=False))
    output = {key: [] for key in ("x", "p", "v", "file", "visits")}
    offset = 0
    for i, (path, count) in enumerate(zip(paths, counts)):
        dest = ids[ids.searchsorted(offset):ids.searchsorted(offset + count)] - offset
        offset += count
        if not len(dest):
            continue
        with np.load(path, allow_pickle=False) as shard:
            x, p, v = (shard[key][dest] for key in ("x", "p", "v"))
            if p.dtype != np.float16 or v.dtype != np.float32:
                raise ValueError(f"{path}: expected compact p=float16 and v=float32 targets")
            validate(x, p, v)
            visits = shard["visits"][dest] if "visits" in shard else np.full(len(x), -1)
            if visits.shape != (len(x),):
                raise ValueError("Invalid visit-count vector")
        for key, value in zip(output, (x, p, v, np.full(len(x), i), visits)):
            output[key].append(value)
    return {key: np.concatenate(arr) for key, arr in output.items()}


def unique_rows(corpus, maps, seen):
    """Remove identical or symmetry-equivalent states, including train/validation overlap."""
    keep = []
    for i, x in enumerate(corpus["x"]):
        # Pack 280 binary features in each D4 orientation. Sorting bytes selects
        # a deterministic orbit representative regardless of the board's orientation.
        variants = np.packbits(x[maps], axis=1)
        key = min(row.tobytes() for row in variants)
        if key not in seen:
            keep.append(i)
            seen.add(key)
    if not keep:
        raise ValueError("No independent rows remain in a split after D4 duplicate removal")
    return {key: array[keep] for key, array in corpus.items()}


def head_layers(model, tier):
    from torch import nn
    if tier == "line_student":
        p, v = model.policy_head[-1], model.value_head[-2]
        if not isinstance(model.value_head[-1], nn.Tanh) or not all(isinstance(m, nn.Linear) for m in (p, v)):
            raise ValueError("Unexpected LineStudent heads")
    else:
        p, v = model.layers["logit_out"], model.layers["value_out"]
        if not all(isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) for m in (p, v)):
            raise ValueError("Unexpected dense Stage2Net heads")
    if p.bias is None or v.bias is None:
        raise ValueError("Expected biased output layers")
    return p, v


def policy_logits(raw, scale, clipped):
    logits = raw * scale
    return np.clip(logits, -32, 32) if clipped else logits


def collect(model, tier, data, device, batch):
    import torch
    result = [[], []]
    heads = head_layers(model, tier)
    captured = {}
    def save(name):
        def hook(_module, _arguments, output):
            captured[name] = output.detach()
        return hook
    handles = [m.register_forward_hook(save(i)) for i, m in enumerate(heads)]
    try:
        with torch.inference_mode():
            for first in range(0, len(data), batch):
                x = torch.from_numpy(data[first:first+batch]).to(device, dtype=torch.float32)
                logits, value = model(x)
                raw = captured[0].reshape(-1, 81)
                if tier == "stage3_teacher":
                    raw = raw.index_select(1, model.order)
                raw_p = raw.cpu().double().numpy()
                z = captured[1].reshape(-1).cpu().double().numpy()
                if not np.isfinite(raw_p).all() or not np.isfinite(z).all():
                    raise ValueError("Nonfinite checkpoint preactivations")
                np.testing.assert_allclose(policy_logits(raw_p, 1, tier == "stage3_teacher"),
                                           logits.cpu().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(np.tanh(z), value.cpu().numpy().reshape(-1), atol=2e-6)
                result[0].append(raw_p)
                result[1].append(z)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(np.concatenate(array) for array in result)


class Scores:
    def __init__(self, raw, z, data, clipped):
        self.raw = raw
        self.z = z
        self.clipped = clipped
        self.legal = data["x"][:, 162:243].astype(bool)
        self.target = data["p"].astype(np.float64)
        self.target /= self.target.sum(1, keepdims=True)
        self.values = data["v"].astype(np.float64)

    def nll(self, scale):
        lg = policy_logits(self.raw, scale, self.clipped)
        shift = np.where(self.legal, lg, -np.inf).max(1, keepdims=True)
        lg -= shift
        denominator = np.where(self.legal, np.exp(np.minimum(lg, 0)), 0).sum(1)
        return np.log(denominator) - (self.target * lg).sum(1)

    def mse(self, scale, bias):
        return (np.tanh(self.z * scale + bias) - self.values) ** 2


def fit_scales(scores):
    # Include identity explicitly at each search start; grid fits use train rows only.
    best_p, best_v = 1., (1., 0.)
    low_p, low_v = float(scores.nll(1.).mean()), float(scores.mse(1., 0.).mean())
    span_s, span_b = 1.0, .5
    for _ in range(5):
        def candidates(center, width, low, high):
            return np.unique(np.r_[center, np.clip(np.linspace(center-width, center+width, 13),low,high)])
        ps = candidates(best_p, span_s, .5, 2.)
        vs = candidates(best_v[0], span_s, .5, 2.)
        bs = candidates(best_v[1], span_b, -.5, .5)
        for scale in ps:
            loss = float(scores.nll(scale).mean())
            if loss < low_p - 1e-12:
                best_p, low_p = float(scale), loss
        for s in vs:
            for b in bs:
                loss = float(scores.mse(s, b).mean())
                if loss < low_v - 1e-12:
                    best_v, low_v = (float(s), float(b)), loss
        span_s /= 4
        span_b /= 4
    return best_p, *best_v


def summary(base, new, files, seed, paths, root):
    delta = base - new
    ids, counts = np.unique(files, return_counts=True)
    sums = np.array([delta[files == i].sum() for i in ids])
    rng = np.random.default_rng(seed)
    groups = len(ids)
    if groups > 1:
        # Resample whole shards with replacement, keeping the original per-position weighting.
        means = []
        for _ in range(32):
            draws = rng.integers(groups, size=(128, groups))
            means.extend(sums[draws].sum(1) / counts[draws].sum(1))
        method = "paired-shard"
    else:
        means = []
        for _ in range(32):
            draws = rng.integers(len(delta), size=(128, len(delta)))
            means.extend(delta[draws].mean(1))
        method = "paired-row-only"
    return {
        "rows": len(base), "shards": groups, "baseline": float(base.mean()), "candidate": float(new.mean()),
        "reduction": float(delta.mean()), "bootstrap95": np.quantile(means,[.025,.975]).tolist(),
        "bootstrap_method": method,
        "by_file": [{"file": str(paths[i].relative_to(root)), "rows": int(n),
                     "reduction": float(s / n)} for i, n, s in zip(ids, counts, sums)],
    }


def evaluate(scores, data, path_list, data_root, fitted, seed):
    p_scale, v_scale, bias = fitted
    losses = {"policy": (scores.nll(1), scores.nll(p_scale)),
              "value": (scores.mse(1, 0), scores.mse(v_scale, bias))}
    results = {}
    for head, (base, new) in losses.items():
        entry = summary(base, new, data["file"], seed, path_list, data_root)
        for label, mask in (("searched", data["visits"] > 0), ("direct", data["visits"] == 0)):
            if mask.any():
                entry[label] = {"rows": int(mask.sum()), "reduction": float((base[mask]-new[mask]).mean())}
        results[head] = entry
    return results


def accepted(entry):
    return (entry["shards"] >= 2 and entry["bootstrap95"][0] > 0
            and entry["reduction"] > 1e-6 and entry.get("searched", {"reduction": 1})["reduction"] >= 0)


def export_if_validated(args, model, data, scores, fitted, approvals):
    import torch
    from training.fit_search_student import export
    pscale = fitted[0] if approvals["policy"] else 1.
    vscale, vbias = fitted[1:] if approvals["value"] else (1., 0.)
    result = {"requested": args.export is not None, "applied_if_exported": [pscale, vscale, vbias], "written": False}
    if args.export is None or not any(approvals.values()):
        return result
    expected_p = policy_logits(scores.raw, pscale, scores.clipped)
    expected_v = np.tanh(scores.z*vscale + vbias)
    with torch.inference_mode():
        p, v = head_layers(model, args.tier)
        p.weight.mul_(pscale)
        p.bias.mul_(pscale)
        v.weight.mul_(vscale)
        v.bias.mul_(vscale).add_(vbias)
        max_p = max_v = 0.
        for first in range(0, len(data["x"]), args.batch_size):
            sl = slice(first, first + args.batch_size)
            x = torch.from_numpy(data["x"][sl]).to(args.device, dtype=torch.float32)
            p, v = (t.cpu().numpy() for t in model(x))
            v = v.reshape(-1)
            np.testing.assert_allclose(p, expected_p[sl], atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(v, expected_v[sl], atol=2e-5, rtol=2e-5)
            max_p = max(max_p,float(abs(p-expected_p[sl]).max()))
            max_v = max(max_v,float(abs(v-expected_v[sl]).max()))
    args.export.parent.mkdir(parents=True, exist_ok=True)
    export(model, torch.from_numpy(data["x"][:16]), args.export, args.tier)
    result.update(written=True, onnx_bytes=args.export.stat().st_size, onnx_sha256=sha256(args.export),
                  fold_probe_rows=len(data["x"]), max_policy_error=max_p, max_value_error=max_v)
    return result


def receipt(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
        os.link(file.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", required=True, choices=("line_student","stage3_teacher"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON receipt")
    parser.add_argument("--export", type=Path, help="Fresh optional .onnx (also publishes sibling .pt)")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261207)
    args = parser.parse_args()
    if not 2 <= args.max_rows <= 12000 or args.batch_size < 1 or args.seed < 0:
        parser.error("Choose 2..12000 sampled rows, a positive batch size, and a nonnegative seed")
    outputs = [args.output] + ([args.export, args.export.with_suffix(".pt")] if args.export else [])
    if (args.export and args.export.suffix != ".onnx") or len({p.resolve() for p in outputs}) != len(outputs) or any(p.exists() for p in outputs):
        parser.error("Receipt and optional .onnx/.pt outputs must have distinct fresh paths")
    import torch
    from training.fit_search_student import load_weights
    from training.pipeline import _symmetry_maps
    from training.line_student import LineStudent
    from training.stage2_student import Stage2Net
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
    model = LineStudent() if args.tier == "line_student" else Stage2Net(compact=False)
    load_weights(model,args.checkpoint)
    model.eval().to(args.device)
    paths = sorted(p for p in args.data.rglob("*.npz") if not p.name.startswith("."))
    sizes = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            n = len(shard["v"])
            if n:
                sizes[path] = n
    train, valid = split_files(list(sizes),args.seed)
    n = min(sum(sizes.values()),args.max_rows)
    size_v = min(sum(sizes[p] for p in valid), max(1, round(n/3)))
    size_t = min(sum(sizes[p] for p in train), n-size_v)
    size_v = min(sum(sizes[p] for p in valid), n-size_t)
    seen = set()
    maps = np.stack([features for features,_ in _symmetry_maps()])
    datasets, scores, sampled = [], [], []
    for files, limit, seed in ((train,size_t,args.seed+1),(valid,size_v,args.seed+2)):
        data = sample(files,sizes,limit,seed)
        sampled.append(len(data["x"]))
        data = unique_rows(data,maps,seen)
        datasets.append(data)
        raw, z = collect(model,args.tier,data["x"],args.device,args.batch_size)
        scores.append(Scores(raw,z,data,args.tier=="stage3_teacher"))
    fitted = fit_scales(scores[0])
    split_metrics = [evaluate(sc,data,files,args.data,fitted,args.seed+3)
                     for sc,data,files in zip(scores,datasets,(train,valid))]
    approvals = {h:accepted(entry) for h,entry in split_metrics[1].items()}
    info = {
        "tier": args.tier, "checkpoint_sha256": sha256(args.checkpoint), "seed": args.seed,
        "sampled_rows_before_orbit_dedup": sampled,
        "unique_rows": [len(d["x"]) for d in datasets],
        "policy_temperature": 1/fitted[0], "policy_logit_scale": fitted[0],
        "policy_stage3_clamp": "after scaling, [-32,32]" if args.tier=="stage3_teacher" else None,
        "value_pre_tanh_scale": fitted[1], "value_pre_tanh_bias": fitted[2],
        "bounds": {"policy_logit_scale":[.5,2.], "value_pre_tanh_scale":[.5,2.], "value_pre_tanh_bias":[-.5,.5]},
        "metric_convention": "Equal row weight, legal normalized soft policy NLL and signed value MSE; positive reduction improves.",
        "acceptance": "Each head requires positive lower held-out 95% paired shard-bootstrap reduction, reduction >1e-6, >=2 validation shards, and no searched-subset loss increase when tracked.",
        "train": split_metrics[0], "heldout": split_metrics[1], "accepted_heads": approvals,
        "export": export_if_validated(args,model,datasets[1],scores[1],fitted,approvals),
    }
    receipt(args.output,info)
    print(json.dumps({"fit": fitted, "accepted_heads": approvals,
                      "holdout_reductions":{h:r["reduction"] for h,r in split_metrics[1].items()},
                      "written":info["export"]["written"]}))


if __name__ == "__main__":
    main()
