"""Train a compact, ply-gated Tiny routing expert from search action values."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.distill_oracle import children, validate
from training.fit_search_student import export, load_weights, save_checkpoint
from training.line_student import LineStudent
from training.pipeline import _symmetry_maps
from training.suite_exclusion import Exclusion
from training.tiny_proof_consensus import digest, packed


OFFICIAL = "b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7"
INCUMBENT = "111ecb6e1dbc4e0d39cb104a6727ca8e233f623c93a89c3b06fca56ea7d059ec"


class ReplyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reply_weight = nn.Linear(80, 1, bias=False)
        self.reply_code = nn.Linear(80, 8, bias=False)
        self.move = nn.Sequential(nn.Linear(99, 24), nn.ReLU(), nn.Linear(24, 1))
        i = torch.arange(9)
        dr = (i[:, None] // 3 - i // 3).abs()
        dc = (i[:, None] % 3 - i % 3).abs()
        self.register_buffer(
            "distances", torch.stack(((dr + dc) / 4, torch.maximum(dr, dc) / 2), -1), persistent=False
        )
        nn.init.zeros_(self.move[-1].weight)
        nn.init.zeros_(self.move[-1].bias)

    def forward(self, action, routing, available):
        # For destination c, pool the plausible replies (c,d). Closed boards have
        # no legal local reply and contribute zero. The trunk separately encodes free routing.
        scores = self.reply_weight(action).squeeze(-1).masked_fill(available < 0.5, -1e4)
        weights = scores.softmax(2) * available
        reply = (weights.unsqueeze(-1) * self.reply_code(action).tanh()).sum(2)
        destinations = reply.unsqueeze(1).expand(-1, 9, -1, -1)
        relation = self.distances.unsqueeze(0).expand(action.shape[0], -1, -1, -1)
        send_back = available.transpose(1, 2).unsqueeze(-1)
        features = torch.cat((action, routing, destinations, relation, send_back), -1)
        return self.move(features).reshape(-1, 81), reply


class MidgameLine(LineStudent):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
        self.late_policy = ReplyPolicy()
        self.late_value = nn.Sequential(nn.Linear(336, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.late_value[-1].weight)
        nn.init.zeros_(self.late_value[-1].bias)

    def warm_start(self, path):
        base = LineStudent()
        load_weights(base, path)
        missing, unexpected = self.load_state_dict(base.state_dict(), strict=False)
        if unexpected or any(not name.startswith(("late_policy.", "late_value.")) for name in missing):
            raise ValueError("Incompatible Tiny warm start")

    def predictions(self, state):
        action, summary, routing, available = self.representations(state)
        p = self.policy_head(action).reshape(-1, 81)
        v = self.value_head[:-1](summary)
        ply = state[:, :162].sum(1, keepdim=True)
        left = available.sum((1, 2)).unsqueeze(1)
        gate = ((ply - 7) / 5).clamp(0, 1) * ((44 - ply) / 9).clamp(0, 1) * ((left - 18) / 9).clamp(0, 1)
        prior, reply = self.late_policy(action, routing, available)
        value = self.late_value(torch.cat((summary, reply.mean(1), reply.amax(1)), 1))
        return (
            p + gate * 2.8 * (prior / 2.8).tanh(),
            (v + gate * 1.4 * (value / 1.4).tanh()).tanh(),
            p,
            v.tanh(),
        )

    def forward(self, state):
        return self.predictions(state)[:2]


def all_views(x):
    return np.unique(np.concatenate([packed(x[:, m]) for m, _ in _symmetry_maps()]))


def read_holdouts(paths):
    x, hashes = [], []
    for path in paths:
        for f in [path] if path.is_file() else sorted(path.glob("*.npz")):
            with np.load(f, allow_pickle=False) as src:
                x.append(src["x"])
            hashes.append(digest(f))
    if not x:
        raise ValueError("Expected fresh, non-training positions")
    return all_views(np.concatenate(x)), hashes


def action_source(directory, limit, expected_visits, rng):
    manifest = directory / "manifest.json"
    info = json.loads(manifest.read_bytes())
    if expected_visits and (
        info.get("visitsPerRoot") != expected_visits
        or info.get("modelSha256") != OFFICIAL
        or info.get("rootVisits") != expected_visits * info["roots"]
    ):
        raise ValueError("Unexpected independent official engine or search budget")
    entries = info["shards"]
    if len({s["file"] for s in entries}) != len(entries):
        raise ValueError("Duplicate source shard")
    lengths = np.cumsum([0, *(s["rows"] for s in entries)])
    ids = np.sort(rng.choice(lengths[-1], min(limit, lengths[-1]), replace=False))
    data = {k: [] for k in ("x", "p", "v", "n", "q")}
    for i, shard in enumerate(entries):
        path = directory / shard["file"]
        if digest(path) != shard["sha256"]:
            raise ValueError("Mismatched source chunk")
        rows = ids[ids.searchsorted(lengths[i]) : ids.searchsorted(lengths[i + 1])] - lengths[i]
        with np.load(path, allow_pickle=False) as src:
            if len(src["v"]) != shard["rows"]:
                raise ValueError("Incorrect shard row count")
            for k in data:
                data[k].append(src[k][rows])
    data = {k: np.concatenate(v) for k, v in data.items()}
    x, p, v, n, q = (data[k] for k in ("x", "p", "v", "n", "q"))
    validate(x, p, v)
    if (
        n.shape != p.shape
        or q.shape != p.shape
        or n.dtype.kind not in "iu"
        or (n < 0).any()
        or not np.isfinite(q).all()
        or (np.abs(q) > 1.00001).any()
        or (q[n == 0] != 0).any()
        or (n[x[:, 162:243] == 0] != 0).any()
        or (expected_visits and not (n.sum(1) == expected_visits - 1).all())
    ):
        raise ValueError("Invalid action visit or parent utility targets")
    return data, digest(manifest)


def plausible(x, n):
    ply = x[:, :162].sum(1)
    enough = n >= np.maximum(16, 0.012 * n.sum(1, keepdims=True))
    return (ply >= 12) & (ply <= 35) & (enough.sum(1) >= 2)


def prepare(args):
    rng = np.random.default_rng(args.seed)
    excluded, hold_hashes = read_holdouts(args.holdout)
    ex = Exclusion(args.suite)
    sources = [
        ("official", args.official, 100000, 3072),
        ("legacy", args.legacy, 105000, 0),
        ("hard", args.shared, 100000, 3072),
    ]
    if args.opponent:
        sources.append(("opponent", args.opponent, 100000, 8192))
    groups = []
    info = []
    for tag, path, limit, visits in sources:
        data, h = action_source(path, limit, visits, rng)
        x = data["x"]
        candidate = plausible(x, data["n"])
        held = np.isin(packed(x), excluded)
        keep = candidate & ~held & ex.keep(x)
        if not keep.any():
            raise ValueError(f"No routing roots from {tag}")
        info.append(
            dict(
                tag=tag,
                manifestSha256=h,
                available=len(x),
                eligible=int(candidate.sum()),
                excludedHoldout=int((candidate & held).sum()),
                kept=int(keep.sum()),
            )
        )
        groups.append({k: a[keep] for k, a in data.items()})
    combined = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
    combined["source"] = np.repeat(np.arange(len(groups), dtype=np.int8), [len(g["x"]) for g in groups])
    # Keep the strongest source for a duplicate root, independently of file order.
    priority = np.take([2, 0, 1, 3], combined["source"])
    order = np.argsort(-priority, kind="stable")
    _, first = np.unique(packed(combined["x"][order]), return_index=True)
    chosen = np.sort(order[first])
    combined = {k: a[chosen] for k, a in combined.items()}
    # The 256-visit student's own mistakes will form an explicit training stratum.
    mine = np.flatnonzero(combined["source"] != 1)
    mine = np.sort(rng.choice(mine, min(len(mine), 3072), replace=False))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    if args.output.exists():
        raise ValueError("Expected new dataset directory")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as td:
        out = Path(td) / "routing"
        out.mkdir()
        np.savez_compressed(out / "roots.npz", **combined)
        np.savez_compressed(out / "mine.npz", x=combined["x"][mine], row=mine)
        np.save(out / "excluded.npy", excluded, allow_pickle=False)
        rec = dict(
            seed=args.seed,
            inputs=info,
            holdoutSha256=hold_hashes,
            rows=len(chosen),
            minedRows=len(mine),
            rootSha256=digest(out / "roots.npz"),
            mineSha256=digest(out / "mine.npz"),
            excludedSha256=digest(out / "excluded.npy"),
        )
        (out / "manifest.json").write_text(json.dumps(rec, indent=2))
        out.rename(args.output)
    print(json.dumps(rec), flush=True)


def best_moves(visits, q, v):
    conf = visits.astype(np.float32) / (visits.astype(np.float32) + 64)
    backed = v[:, None] + conf * (q - v[:, None])
    supported = visits >= 32
    best = np.where(supported, backed, -2).argmax(1)
    return best, backed, supported


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.15)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    identity = json.loads((args.data / "manifest.json").read_text())
    for name, key in [
        ("roots.npz", "rootSha256"),
        ("mine.npz", "mineSha256"),
        ("excluded.npy", "excludedSha256"),
    ]:
        if digest(args.data / name) != identity[key]:
            raise ValueError("Prepared routing data changed")
    with np.load(args.data / "roots.npz", allow_pickle=False) as src:
        x, n, q, p, v, source = [src[k] for k in ("x", "n", "q", "p", "v", "source")]
    with np.load(args.data / "mine.npz", allow_pickle=False) as requests:
        mined = requests["row"]
    with np.load(args.mined, allow_pickle=False) as result:
        if (
            str(result["model_sha"].item()) != INCUMBENT
            or str(result["input_sha"].item()) != identity["mineSha256"]
        ):
            raise ValueError("Search mining corpus or incumbent mismatch")
        actual = result["move"].astype(int)
    if len(actual) != len(mined) or not x[mined, 162 + actual].all():
        raise ValueError("Invalid mined moves")
    best, backed, support = best_moves(n, q, v)
    ii = np.arange(len(x))
    top = n.argmax(1)
    left = top.copy()
    alt = np.sqrt(n.astype(np.float32))
    alt[n < 16] = 0
    alt[ii, left] = 0
    alternatives = alt.sum(1) > 0
    gumbel = -np.log(-np.log(np.clip(rng.random(n.shape, dtype=np.float32), 1e-7, 1 - 1e-7)))
    right = np.where(alt > 0, np.log(np.maximum(alt, 1e-10)) + gumbel, -1e6).argmax(1)
    gap = backed[mined, best[mined]] - backed[mined, actual]
    hard = support[mined, actual] & (gap > 0.045) & (best[mined] != actual)
    mistakes = mined[hard]
    left[mistakes] = best[mistakes]
    right[mistakes] = actual[hard]
    # Child side-to-move is the opposite of the stored parent action Q.
    children_x = []
    children_q = []
    weights = []
    excl = np.load(args.data / "excluded.npy", allow_pickle=False)
    safe = alternatives.copy()
    for moves in (left, right):
        child, term, won = children(x, ii, moves)
        safe &= ~np.isin(packed(child), excl)
        safe &= Exclusion(args.suite).keep(child)
        confidence = n[ii, moves].astype(np.float32) / (n[ii, moves] + 64.0)
        children_x.append(child)
        children_q.append(np.where(term, won, q[ii, moves]).astype(np.float32))
        weights.append(np.where(term, 0.0, confidence).astype(np.float32))
    safe_idx = np.flatnonzero(safe)
    if len(safe_idx) < 128:
        raise ValueError("Not enough eligible searched child pairs")

    def tensor(a):
        return torch.as_tensor(a, device=device)

    p = p.astype(np.float32)
    p /= p.sum(1, keepdims=True)
    p *= np.exp(np.clip((backed - (p * backed).sum(1, keepdims=True)) / 0.48, -1.0, 1.0))
    p /= p.sum(1, keepdims=True)
    x, p, v, source = tensor(x), tensor(p.astype(np.float16)), tensor(v), tensor(source)
    a, b = map(tensor, children_x)
    qa, qb = map(tensor, children_q)
    wa, wb = map(tensor, weights)
    left, right = tensor(left), tensor(right)
    ids = [torch.nonzero(source == i).flatten() for i in range(int(source.max().item()) + 1)]
    if not len(mistakes):
        print("No strongly supported mined disagreements; using all sources uniformly", flush=True)
    hard_ids = tensor(mistakes)
    pair_ids = tensor(safe_idx)
    hard_pairs = tensor(np.intersect1d(mistakes, safe_idx))
    model = MidgameLine().to(device).eval()
    if args.resume_model:
        load_weights(model, args.resume_model)
    else:
        model.warm_start(args.initial)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    live = [p for p in model.parameters() if p.requires_grad]
    shadow = [p for k, p in ema.named_parameters() if k.startswith(("late_policy.", "late_value."))]
    optimizer = torch.optim.AdamW(live, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.steps[-1], eta_min=args.lr * 0.12
    )
    counts = [48, 40, 24, 24] if len(ids) == 4 else [64, 48, 24]
    fingerprint = {
        "version": "midgame-routing-v1",
        "seed": args.seed,
        "initial": digest(args.resume_model or args.initial),
        "dataset": digest(args.data / "manifest.json"),
        "mined": digest(args.mined),
        "steps": args.steps,
        "lr": args.lr,
    }
    receipt = dict(
        identity=fingerprint,
        rawRows=len(x),
        hardSearchErrors=len(mistakes),
        safeChildPairs=len(safe_idx),
        excludedChildPairs=int((~safe).sum()),
        rootCounts=[len(i) for i in ids],
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.state.parent.mkdir(exist_ok=True, parents=True)
    done = 0
    if args.state.exists():
        saved = torch.load(args.state, map_location="cpu", weights_only=True)
        if saved["identity"] != fingerprint:
            raise ValueError("Incompatible routing trainer resume")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["opt"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda"], device)
        done = saved["step"]

    def sample(pool, count):
        return pool[torch.randint(len(pool), (count,), device=device)]

    def publish(step):
        dest = args.output.with_name(f"{args.output.name}-{step}.onnx")
        if not dest.exists():
            export(copy.deepcopy(ema).cpu(), x[:24].cpu(), dest, "line_student")
        report = dest.with_suffix(".metrics.json")
        if not report.exists():
            with report.open("x") as f:
                json.dump(dict(receipt, step=step, sha256=digest(dest)), f, indent=2)

    if done in args.steps:
        publish(done)
    print("Ready", receipt, flush=True)
    for step in range(done + 1, args.steps[-1] + 1):
        roots = torch.cat([sample(pool, k) for pool, k in zip(ids, counts)])
        if len(hard_ids):
            roots = torch.cat((roots, sample(hard_ids, 24)))
        pairs = sample(pair_ids, 64)
        if len(hard_pairs):
            pairs[:24] = sample(hard_pairs, 24)
        idx = torch.cat((roots, pairs))
        states = torch.cat((x[idx], a[pairs], b[pairs])).float()
        logits, y, old, old_y = model.predictions(states)
        y = y.float().flatten()
        nroot = len(idx)
        legal = x[idx, 162:243] > 0
        logp = logits[:nroot].masked_fill(~legal, -1e9).log_softmax(1)
        prior = old[:nroot].masked_fill(~legal, -1e9).softmax(1)
        trust = torch.where(source[idx] == 1, 0.42, 0.86)
        target = trust[:, None] * p[idx].float() + (1 - trust[:, None]) * prior
        loss_policy = -(target * logp).sum(1).mean()
        outcome = trust * v[idx] + (1 - trust) * old_y[:nroot].detach().flatten()
        loss_value = (y[:nroot] - outcome).square().mean()
        pred_a = -y[nroot : nroot + len(pairs)]
        pred_b = -y[nroot + len(pairs) :]
        loss_child = (
            wa[pairs] * (pred_a - qa[pairs]).square() + wb[pairs] * (pred_b - qb[pairs]).square()
        ).mean()
        margin = qa[pairs] - qb[pairs]
        pairconf = torch.minimum(wa[pairs], wb[pairs])
        loss_margin = (
            pairconf * F.smooth_l1_loss(pred_a - pred_b, margin, beta=0.2, reduction="none")
        ).mean()
        rid = torch.arange(len(pairs), device=device) + len(roots)
        delta = logits[rid, left[pairs]] - logits[rid, right[pairs]]
        ranking = (
            pairconf
            * (margin.abs() > 0.06)
            * ((margin.abs() * 3).clamp_max(1.5) - margin.sign() * delta).clamp(0, 3).square()
        ).mean()
        loss = loss_policy + 0.9 * loss_value + 0.72 * loss_child + 0.45 * loss_margin + 0.06 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(live, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            torch._foreach_lerp_(shadow, live, 0.008)
        if step == 1 or step % 250 == 0:
            print("Step", step, loss.item(), flush=True)
        if step % 500 == 0 or step in args.steps:
            save_checkpoint(
                args.state,
                dict(
                    identity=fingerprint,
                    step=step,
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    opt=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                ),
            )
        if step in args.steps:
            publish(step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    data = sub.add_parser("prepare")
    model = sub.add_parser("fit")
    for key in ("official", "legacy", "shared", "suite", "output"):
        data.add_argument("--" + key, type=Path, required=True)
    data.add_argument("--opponent", type=Path)
    data.add_argument("--holdout", type=Path, nargs="+", required=True)
    data.add_argument("--seed", type=int, default=26316701)
    for key in ("data", "suite", "mined", "initial", "state", "output"):
        model.add_argument("--" + key, type=Path, required=True)
    model.add_argument("--resume-model", type=Path)
    model.add_argument("--seed", type=int, default=26316702)
    model.add_argument("--lr", type=float, default=0.0011)
    model.add_argument("--steps", type=int, nargs="+", default=[1400, 4200])
    model.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
