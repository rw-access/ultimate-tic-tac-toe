r"""Mine positions that distinguish a packed student from its teacher, then refine the student.

The `mine` command compares each model's preferred legal action and the search
target's preferred action. The child-value gap is measured from the root seat.
Only search-supported pairs become ranking targets. Exact positions use their
proved action set for both mining and training.

Run `python -m training.regret_replay <command> --help`. Supply the packed
convolutional source graph and its matching moderate Stage 2 checkpoint to
`mine` and `fit`. The dense Stage 2 teacher is used only for mining. Run
`filter` with the public tournament state file before fitting to exclude those
roots, their symmetry equivalents and all roots that can reach one in a move.
The optimizer can train centroids, directly stored biases, and optionally
weight assignments. Each production ONNX export uses the existing optimized
project graph and is compared numerically with the trained model.

The 1,200-step Gen101 experiment used the following recipe. Paths stand for the
matching compact x/p/v shard directories and the source model files. Sources
are read in the order shown and sampled without replacement. Each sampled
source gets an identity digest in the mining manifest.

    python -m training.regret_replay mine \
        --input gen101-convolution.onnx --base moderate.pt --teacher wave.pt \
        --seed 2026112902 --batch-size 256 --output mined \
        --source deep replay/1024-visits 65536 --source exact replay/proved 42000 \
        --source paper replay/stage2 100000 --source replay replay/archived 32000 \
        --source oracle replay/symmetry-oracle 32000
    python -m training.regret_replay filter --data mined --output protected \
        --published-suite utttai/utttpy/selfplay/evaluation_uttt_states.py
    python -m training.regret_replay fit \
        --input gen101-convolution.onnx --base moderate.pt --data protected \
        --mix deep .34 --mix exact .15 --mix paper .28 --mix oracle .16 --mix replay .07 \
        --hard-fraction .42 --margin .24 --teacher-scale 1 --value-weight .85 \
        --seed 2026112903 --batch-size 192 --lr .000075 --schedule-steps 9200 \
        --at 1200 --save-raw --name regret-student --output checkpoints

The source convolution graph has SHA-256
f9c549d9d4f41419f8516f3af58d1e2518d4c98173e1ed958c6894aa396a3027.
The matching moderate checkpoint has SHA-256
2ad7148422bec12b791e501807f66c5466eb50c25dc877694a1027043fc7998d.
The full teacher checkpoint has SHA-256
a9aed5cf002c32ccc0d279d95110b8903b1448a665c41e1087244728395fe000.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch.nn import functional as F

from .distill_oracle import children
from .fast_stage2_graph import TokenGraph, verify
from .fit_search_student import file_hash, load_positions, load_weights
from .pipeline import _symmetry_maps
from .refine_packed import bind_codebooks, check_parity, layout_hash, update_centroids
from .stage2_student import Stage2Net


def probability(logits, legal):
    return logits.float().masked_fill(~legal, -1e9).softmax(1)


def normalize(p):
    return p / p.sum(1, keepdim=True).clamp_min(1e-20)


def stratified_hardness(x, scores):
    """Convert difficulty to within-stratum ranks, so late free moves cannot dominate."""
    ply = x[:, :162].sum(1)
    branches = x[:, 162:243].sum(1)
    stratum = np.digitize(ply, (15, 32, 48)) * 4 + np.digitize(branches, (2, 5, 10))
    rank = np.zeros(len(scores), np.float32)
    for group in np.unique(stratum):
        rows = np.flatnonzero(stratum == group)
        _, inverse, counts = np.unique(scores[rows], return_inverse=True, return_counts=True)
        # Tied observations have the same midpoint; constant strata contribute no mining preference.
        mid = (np.cumsum(counts) - counts / 2) / len(rows)
        rank[rows] = mid[inverse]
    return rank


def immutable(path, write):
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        write(pending)
        os.link(pending, path)


def emit(payload):
    print(json.dumps(payload, allow_nan=False), flush=True)


def student_from(input_path, base, device):
    packed = onnx.load(input_path)
    student = Stage2Net(compact=True, moderate=True)
    load_weights(student, base)
    # A refined source can differ from its original .pt in these direct
    # float32 tables as well as quantized kernels.
    with torch.no_grad():
        for _, parameter, stored in stored_affines(packed, student):
            parameter.copy_(torch.from_numpy(stored.copy()))
    books = bind_codebooks(packed, student)
    return packed, student.eval().to(device), books


@torch.inference_mode()
def child_scores(x, choices, teacher, device):
    """Evaluate unique (root, move) pairs once, preserving exact terminal root values."""
    n, k = choices.shape
    # Include root index to prevent merging equal move indices from different states.
    pairs, inverse = np.unique(np.arange(n)[:, None] * 81 + choices, return_inverse=True)
    states, terminal, won = children(x, pairs // 81, pairs % 81)
    q = won.astype(np.float32)
    live = np.flatnonzero(~terminal)
    for start in range(0, len(live), 256):
        rows = live[start:start + 256]
        child = torch.from_numpy(states[rows]).to(device).float()
        q[rows] = -teacher(child)[1].flatten().cpu().numpy()
    return q[inverse].reshape(n, k), terminal[inverse].reshape(n, k)


def ranking_evidence(target, choices, q, terminal, exact):
    """Choose a search-supported better alternative to the student's own top move."""
    rows = np.arange(len(target))
    incumbent, teacher, searched = choices.T
    priority = target[rows[:, None], choices]
    gap = q - q[:, :1]
    # A proven terminal win is sufficient. For estimates require either
    # positive visit-mass support or both independent action selectors to agree.
    supported = ((priority >= priority[:, :1] + .04) & (priority >= .5 * priority.max(1, keepdims=True))) | (
        (teacher == searched)[:, None] & (choices == teacher[:, None]))
    eligible = (supported & (gap >= .12)) | (terminal & (q == 1) & (gap > 0))
    eligible[:, 0] = False
    eligible &= choices != incumbent[:, None]
    alternatives = np.where(eligible, gap, -1)
    best = np.argmax(alternatives, axis=1)
    gain = np.maximum(0, alternatives[rows, best])
    better = choices[rows, best].copy()
    if exact:
        # Every proved optimal action has positive target mass. A larger prior
        # on another optimal action does not constitute a tactical mistake.
        misses = target[rows, incumbent] == 0
        better = np.where(misses, searched, incumbent)
        gain = misses.astype(np.float32)
    return better, np.clip(gain, 0, 1).astype(np.float32)


def validate_alternative_manifest(data, expected_digest):
    """Authenticate the unversioned output of prove_alternatives.py.

    That producer solves every legal action and mixes uniform mass across all
    optimal moves. Pinning its receipt and checking each shard preserves this
    stronger support guarantee when training from the exported policy alone.
    """
    manifest = data/"manifest.json"
    if not expected_digest or file_hash(manifest) != expected_digest:
        raise ValueError("Alternative proof manifest requires its independently pinned digest")
    proof = json.loads(manifest.read_bytes())
    from .suite_exclusion import PUBLISHED_SUITE_SHA256
    digests = ("sourceManifestSha256", "sourceTeacherSha256", "solverSha256")
    def sha_ok(value):
        return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
    counters = ("eligible","attemptedRoots","solvedRoots","attemptedChildren","solvedChildren",
                "terminalChildren","nodes","nodesPerRoot","minRemaining","maxRemaining")
    if (any(type(proof.get(k)) is not int for k in counters) or
            any(not sha_ok(proof.get(k)) for k in digests) or proof.get("excludedSuiteSha256") != PUBLISHED_SUITE_SHA256 or
            not 15 <= proof.get("minRemaining",0) <= proof.get("maxRemaining",0) <= 28 or
            not 0 < proof.get("solvedRoots",0) <= proof.get("attemptedRoots",0) <= proof.get("eligible",0) or
            not 0 < proof.get("solvedChildren",0) <= proof.get("attemptedChildren",0) or
            min(proof["nodes"], proof["nodesPerRoot"]) <= 0 or proof["terminalChildren"] < 0 or
            proof["attemptedChildren"] + proof["terminalChildren"] > 2*proof["solvedRoots"] or
            proof["nodes"] > (proof["nodesPerRoot"]+1)*(proof["attemptedRoots"]+proof["attemptedChildren"])):
        raise ValueError("Incomplete parent/child all-action proof provenance")
    files = {p.name:p for p in data.glob("*.npz")}
    listed = proof.get("shards",[])
    if not listed or len(listed) != len(files) or {s["file"] for s in listed} != set(files):
        raise ValueError("Alternative proof inventory differs from its receipt")
    total = 0
    for shard in listed:
        path = files[shard["file"]]
        if not sha_ok(shard["sha256"]) or file_hash(path) != shard["sha256"]:
            raise ValueError("Changed alternative proof shard")
        with np.load(path,allow_pickle=False) as arr:
            v = arr["v"]
            if len(v) != shard["rows"] or not np.isin(v,(-1,0,1)).all():
                raise ValueError("Alternative proof rows disagree with their receipt or W/D/L")
            total += len(v)
    if total != proof["solvedRoots"] + proof["solvedChildren"]:
        raise ValueError("Alternative proof totals disagree with solved parents and children")


@torch.inference_mode()
def mine_source(data, limit, seed, teacher, student, kind, batch_size, device, proof_digest=None):
    exact = kind in ("proof","exact")
    if kind == "proof":
        validate_alternative_manifest(data, proof_digest)
    identity = {}
    states, policies, values = load_positions(data, limit, np.random.default_rng(seed), torch.device("cpu"), identity)
    x, p, v = (a.numpy() for a in (states, policies, values))
    if kind == "exact":
        proof = json.loads((data / "manifest.json").read_bytes())
        minimum_mass = 81 * float(np.nextafter(np.float16(0), np.float16(1)))
        if (proof.get("format") != "uttt-exact-wdl-v1" or not np.isin(v, (-1, 0, 1)).all() or
                not minimum_mass <= proof.get("uniform_weight", 0) <= 1):
            raise ValueError("Exact replay needs certified W/D/L and representable mass on every optimal move")
    fields = {name: [] for name in ("teacher", "teacher_v", "chosen", "better", "gain", "difficulty", "mismatch")}
    for start in range(0, len(x), batch_size):
        s = slice(start, start + batch_size)
        root = states[s].to(device).float()
        legal = root[:, 162:243] > 0
        tl, tv = teacher(root)
        sl, sv = student(root)
        a, b = probability(tl, legal), probability(sl, legal)
        target = normalize(policies[s].to(device).float())
        choices = torch.stack((b.argmax(1), a.argmax(1), target.argmax(1)), 1).cpu().numpy()
        y = target.cpu().numpy()
        if exact:
            q = np.zeros(choices.shape, np.float32)
            terminal = np.zeros_like(q, bool)
        else:
            q, terminal = child_scores(x[s], choices, teacher, device)
        better, gain = ranking_evidence(y, choices, q, terminal, exact)
        m = (a + b) * .5
        js = .5 * ((a * (a.clamp_min(1e-20).log() - m.clamp_min(1e-20).log())).sum(1) +
                   (b * (b.clamp_min(1e-20).log() - m.clamp_min(1e-20).log())).sum(1))
        rows = np.arange(len(y))
        missed = y[rows, choices[:, 2]] - y[rows, choices[:, 0]]
        difficult = (1.7 * gain + 1.2 * js.cpu().numpy() + (.0 if exact else .65) * missed +
                     .45 * np.abs(sv.flatten().cpu().numpy() - (v[s] if exact else tv.flatten().cpu().numpy())))
        for name, values_ in zip(fields, (a.half().cpu().numpy(), tv.flatten().cpu().numpy(), choices[:, 0],
                                          better, gain, difficult, choices[:, 0] != choices[:, 1])):
            fields[name].append(values_)
    fields = {k: np.concatenate(parts) for k, parts in fields.items()}
    difficulty = fields.pop("difficulty")
    hard = stratified_hardness(x, difficulty)
    metrics = {"rows": len(x), "teacherStudentDisagreement": float(fields.pop("mismatch").mean()),
               "supportedRegretFraction": float(np.mean(fields["gain"] > 0)),
               "meanSupportedRegret": float(fields["gain"].mean()), "identity": identity}
    fields["chosen"] = fields["chosen"].astype(np.uint8)
    fields["better"] = fields["better"].astype(np.uint8)
    return dict(x=x, p=p, v=v, hard=hard, **fields), metrics


def mine(args, device):
    if args.output.exists() and (args.output / "manifest.json").exists():
        raise ValueError("A completed mined set already occupies the requested output directory")
    _, student, _ = student_from(args.input, args.base, device)
    teacher = Stage2Net(compact=False)
    load_weights(teacher, args.teacher)
    teacher.eval().requires_grad_(False).to(device)
    report = {"format": "uttt-regret-replay-v1", "seed": args.seed, "student": file_hash(args.input),
              "base": file_hash(args.base), "teacher": file_hash(args.teacher), "sources": []}
    if len(set(row[0] for row in args.source)) != len(args.source):
        raise ValueError("Duplicate source name")
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (name, directory, limit) in enumerate(args.source):
        if not name.isalnum() or int(limit) <= 0:
            raise ValueError("Use an alphanumeric source name and positive sample cap")
        dest = args.output / (name + ".npz")
        arrays, metrics = mine_source(Path(directory), int(limit), args.seed+i, teacher, student,
                                      name, args.batch_size, device, args.proof_manifest_sha256)
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("Mined replay contains non-finite values")
        immutable(dest, lambda path: np.savez_compressed(path, **arrays))
        info = {"name": name, "sha256": file_hash(dest), **metrics}
        emit(info)
        report["sources"].append(info)
    immutable(args.output / "manifest.json", lambda p: p.write_text(json.dumps(report, indent=2) + "\n"))



def excluded_rows(x, encoded):
    """Return root and successor exclusions; search-mined rows can use child-value labels."""
    blocked = np.unique(np.ascontiguousarray(encoded).view("V35"))
    max_ply = np.unpackbits(encoded, axis=1)[:, :162].sum(1).max()
    def contains(states):
        keys = np.ascontiguousarray(np.packbits(states, axis=1)).view("V35").reshape(-1)
        return np.isin(keys, blocked)
    ply = x[:, :162].sum(1)
    root = np.zeros(len(x), bool)
    near = np.flatnonzero(ply <= max_ply)
    root[near] = contains(x[near])
    child = np.zeros(len(x), bool)
    for rows in np.array_split(np.flatnonzero((ply < max_ply) & ~root), max(1, len(near)//256 + 1)):
        if len(rows):
            r, actions = np.nonzero(x[rows, 162:243])
            successors, _, _ = children(x, rows[r], actions)
            child[rows[r[contains(successors)]]] = True
    return root, child


def filter_mined(args):
    original = json.loads((args.data / "manifest.json").read_bytes())
    if original["format"] != "uttt-regret-replay-v1" or args.output.exists():
        raise ValueError("Expected a mined source and a fresh output directory")
    if args.published_suite:
        from .suite_exclusion import Exclusion
        orbit = Exclusion(args.published_suite)
        encoded = np.frombuffer(b"".join(sorted(orbit.features)), np.uint8).reshape(-1, 35)
        exclusion_file_hash = file_hash(args.published_suite)
    else:
        encoded = np.load(args.excluded, allow_pickle=False)
        exclusion_file_hash = file_hash(args.excluded)
    if encoded.dtype != np.uint8 or encoded.ndim != 2 or encoded.shape[1] != 35 or not len(encoded):
        raise ValueError("Expected N x 35 uint8 packed excluded states with numpy's big-endian bit order")
    report = {k:v for k,v in original.items() if k != "sources"}
    report.update(sources=[], excludedSetSha256=exclusion_file_hash,
                  excludedPackedSha256=hashlib.sha256(encoded.tobytes()).hexdigest(), excludedStates=len(encoded))
    args.output.mkdir(parents=True)
    for source in original["sources"]:
        f = args.data/(source["name"]+".npz")
        if file_hash(f) != source["sha256"]:
            raise ValueError("Source changed before applying exclusion")
        with np.load(f, allow_pickle=False) as stored:
            all_rows = {k:stored[k] for k in stored.files}
        root, child = excluded_rows(all_rows["x"], encoded)
        keep = ~(root | child)
        rows = {k:v[keep] for k,v in all_rows.items()}
        # Original rank is monotonic within a stratum. Removing rows, then
        # reranking within that stratum, restores the intended sample quantiles.
        rows["hard"] = stratified_hardness(rows["x"], rows["hard"])
        dest = args.output/f.name
        immutable(dest, lambda p: np.savez_compressed(p, **rows))
        number = {"name":source["name"], "rows":int(keep.sum()), "excludedRoots":int(root.sum()),
                  "excludedParents":int(child.sum()), "sourceSha256":source["sha256"], "sha256":file_hash(dest)}
        report["sources"].append(number)
        emit(number)
    immutable(args.output/"manifest.json", lambda p:p.write_text(json.dumps(report,indent=2)+"\n"))


def orbit_holdout(x, seed, fraction=20):
    """Assign all D4 copies of a root to the same held-out fold across sources."""
    held = np.zeros(len(x), bool)
    input_maps = [indices for indices, _ in _symmetry_maps()]
    for start in range(0, len(x), 2048):
        rows = x[start:start+2048]
        identity = np.full(len(rows), np.iinfo(np.uint32).max, dtype=np.uint32)
        for input_map in input_maps:
            packed = np.ascontiguousarray(np.packbits(rows[:, input_map], axis=1))
            checks = np.fromiter((zlib.crc32(row, seed) for row in packed), np.uint32, len(rows))
            identity = np.minimum(identity, checks)
        held[start:start+len(rows)] = identity % fraction == 0
    return held

def load_pools(paths, mixture, device, seed, orbit_seed):
    source_meta, manifests = {}, []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_bytes())
        if manifest["format"] != "uttt-regret-replay-v1":
            raise ValueError("Invalid replay format")
        if manifests and any(manifest.get(k) != manifests[0].get(k)
                             for k in ("base", "student", "teacher", "excludedSetSha256")):
            raise ValueError("Replay sources have different teacher, student, or excluded openings")
        for source in manifest["sources"]:
            name = source["name"]
            if name in source_meta:
                raise ValueError("Duplicated source across mined manifests")
            source_meta[name] = (path/(name+".npz"), source["sha256"])
        manifests.append(manifest)
    pools = []
    if len(set(name for name, _ in mixture)) != len(mixture):
        raise ValueError("Duplicate mixture source")
    for name, fraction in mixture:
        probability_ = float(fraction)
        if not 0 < probability_ <= 1:
            raise ValueError("Invalid source fraction")
        f, sha256 = source_meta[name]
        if file_hash(f) != sha256:
            raise ValueError("Replay sample digest changed")
        with np.load(f, allow_pickle=False) as blob:
            data = {k: torch.from_numpy(blob[k]).to(device) for k in blob.files}
        n = len(data["x"])
        if orbit_seed is None:
            # Retain the original per-source diagnostics split for reproducibility.
            rng = np.random.default_rng(seed + len(pools))
            order = rng.permutation(n)
            holdout = min(max(1, n // 20), 2048)
            train_ids, valid_ids = order[holdout:], order[:holdout]
        else:
            excluded = orbit_holdout(data["x"].cpu().numpy(), orbit_seed)
            train_ids, valid_ids = np.flatnonzero(~excluded), np.flatnonzero(excluded)
            if len(valid_ids) > 2048:
                valid_ids = np.random.default_rng(orbit_seed).choice(valid_ids, 2048, replace=False)
        if not len(train_ids) or not len(valid_ids):
            raise ValueError("Replay source too short for a disjoint diagnostics split")
        valid = torch.as_tensor(valid_ids, device=device)
        train = torch.as_tensor(train_ids, device=device)
        hard = train[data["hard"][train] >= .8]
        if not len(hard):
            hard = train
        pools.append((name, probability_, data, train, hard, valid))
    if not math.isclose(sum(p[1] for p in pools), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Source fractions must sum to one")
    return pools, manifests[0]



class AdaptiveBook(torch.nn.Module):
    """Accumulate straight-through weight updates between discrete reassignment steps."""
    def __init__(self, book, size):
        super().__init__()
        self.centers = book.centers
        self.register_buffer("codes", book.codes)
        self.register_buffer("initial_codes", book.codes.to(torch.uint8))
        self.residual = torch.nn.Parameter(torch.zeros_like(book.codes, dtype=torch.float32))
        self.size = size

    def forward(self, original):
        rounded = self.centers.half().float().gather(1, self.codes)
        # The expression in parentheses has exact zero forward value. Its
        # derivative accumulates the update each individual weight would take.
        dense = rounded + (self.residual - self.residual.detach())
        return dense.flatten()[:self.size].reshape_as(original)

    @staticmethod
    def strongest(scores, limit):
        selected = torch.zeros_like(scores, dtype=torch.bool)
        if limit > 0:
            top, ids = torch.topk(scores.flatten(), min(limit, scores.numel()), sorted=False)
            selected.flatten()[ids[top > 0]] = True
        return selected

    @torch.no_grad()
    def reassign(self, step_fraction=1.0, total_fraction=1.0):
        width = self.codes.shape[1]
        candidates = self.codes.clone()
        advantage = torch.zeros_like(self.residual)
        for start in range(0, len(self.codes), 128):
            groups = slice(start, start + 128)
            c = self.centers[groups].half().float()
            code, residual = self.codes[groups], self.residual[groups]
            before = c.gather(1, code)
            goal = before + residual
            nearest = (c[:, None, :] - goal[:, :, None]).abs().argmin(2)
            next_value = c.gather(1, nearest)
            distance = (next_value - before).abs()
            improvement = (residual.abs() - (goal-next_value).abs()) / distance.clamp_min(1e-8)
            advantage[groups] = torch.where((distance > 1e-8) & (improvement > .02), improvement, 0.)
            candidates[groups] = nearest
        advantage.flatten()[self.size:] = 0
        changed = self.codes != self.initial_codes
        room = max(0, int(self.size*total_fraction) - int(changed.sum()))
        new = self.strongest(advantage.masked_fill(changed, 0), room)
        chosen = self.strongest(advantage.masked_fill(~new & ~changed, 0), int(self.size*step_fraction))
        rounded = self.centers.half().float()
        before = rounded.gather(1, self.codes)
        self.codes.copy_(torch.where(chosen, candidates, self.codes))
        remaining = self.residual + before - rounded.gather(1, self.codes)
        span = (rounded.max(1).values - rounded.min(1).values)[:,None].clamp_min(1e-8)
        self.residual.copy_(torch.clamp(remaining, min=-.35*span, max=.35*span))
        return chosen.sum()

    def packed_bytes(self):
        bits = self.centers.shape[1].bit_length() - 1
        codes = self.codes.detach().to(torch.uint8).cpu().numpy().reshape(-1, 8//bits)
        shifts = np.arange(8//bits, dtype=np.uint8)*bits
        return np.bitwise_or.reduce(codes << shifts, axis=1).tobytes()


def adapt_assignments(model, books):
    by_id = {id(book): key for key, book in books.items()}
    for module in list(model.modules()):
        if not hasattr(module, "parametrizations") or "weight" not in module.parametrizations:
            continue
        chain = module.parametrizations.weight
        name = by_id.get(id(chain[0]))
        if name is not None:
            adjusted = AdaptiveBook(chain[0], chain.original.numel())
            chain[0] = adjusted
            books[name] = adjusted
    if not all(isinstance(x, AdaptiveBook) for x in books.values()):
        raise ValueError("An assignment table was not attached to its model weight")

def stored_affines(packed, model):
    """Map direct serialized float32 biases and offsets to their training parameters."""
    tables = {t.name: t for t in packed.graph.initializer}
    for name, parameter in model.named_parameters():
        if name not in tables or not name.endswith((".bias", ".offset")):
            continue
        array = numpy_helper.to_array(tables[name])
        if array.dtype != np.float32 or tuple(array.shape) != tuple(parameter.shape) or not np.isfinite(array).all():
            raise ValueError(f"Invalid stored bias or offset for {name}")
        yield name, parameter, array


def trainable_biases(packed, model):
    params = {name: p.requires_grad_(True) for name, p, _ in stored_affines(packed, model)}
    if not params:
        raise ValueError("No supported exact float32 biases found")
    return params


def policy_head_parameters(model, books, biases):
    """Restrict updates to the policy branch after the last shared activation."""
    modules = ('layers.policy', 'layers.upsample', 'layers.logits', 'layers.logit_out')
    def belongs(name):
        return any(name == prefix or name.startswith(prefix + '.') for prefix in modules)
    selected = {id(layer.parametrizations.weight[0]) for name, layer in model.named_modules()
                if belongs(name) and hasattr(layer, 'parametrizations') and 'weight' in layer.parametrizations}
    kept = {}
    for key, book in books.items():
        enabled = id(book) in selected
        book.centers.requires_grad_(enabled)
        if enabled:
            kept[key] = book
    if not kept or len(kept) == len(books):
        raise ValueError('Expected separately encoded policy and trunk/value parameters')
    for name, parameter in biases.items():
        parameter.requires_grad_(belongs(name))
    return kept, {name: p for name, p in biases.items() if belongs(name)}


def value_fingerprint(payload, fixtures):
    """Hash production value outputs on both singleton and batched parity inputs."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(payload, options, providers=['CPUExecutionProvider'])
    hashed = hashlib.sha256()
    for batch in (fixtures[:1], fixtures):
        values = session.run(['value'], {'state': batch})[0]
        if values.shape != (len(batch), 1) or not np.isfinite(values).all():
            raise ValueError('Invalid value output in the policy-only control')
        hashed.update(values.astype('<f4').tobytes())
    return hashed.hexdigest()


def objective(logits, value, x, p, v, teacher, teacher_v, chosen, better, gain, mix, exact, margin):
    legal = x[:, 162:243] > 0
    logits = logits.float().masked_fill(~legal, -1e9)
    logp = logits.log_softmax(1)
    target = torch.lerp(normalize(p.float()), normalize(teacher.float()), mix)
    target_v = torch.lerp(v, teacher_v, mix)
    log_support = torch.logsumexp(logp.masked_fill(p <= 0, -1e9), dim=1)
    # Encourages total mass on *all* optimal actions in tied exact positions.
    exact_ce = .75 * (-log_support) - .25 * (normalize(p.float()) * logp).sum(1)
    ce = torch.where(exact, exact_ce, -(target * logp).sum(1))
    rows = torch.arange(len(x), device=x.device)
    gap = logits[rows, better] - logits[rows, chosen]
    rank_loss = (F.softplus(.65 - gap) * gain).mean()
    target_v = torch.where(exact, v.flatten(), target_v.flatten())
    return ce.mean(), F.mse_loss(value.float().flatten(), target_v), rank_loss * margin


@torch.inference_mode()
def validation(model, pools):
    result = {}
    for name, _, d, _, _, valid in pools:
        rows_ = []
        for ix in valid.split(256):
            x, p = d["x"][ix].float(), normalize(d["p"][ix].float())
            lp, value = model(x)
            logp = lp.masked_fill(x[:, 162:243] == 0, -1e9).log_softmax(1)
            best = logp.argmax(1)
            gain = d["gain"][ix]
            rows = torch.arange(len(ix), device=x.device)
            r = (lp[rows, d["better"][ix].long()] > lp[rows, d["chosen"][ix].long()])
            rows_.append(torch.stack((-(p * logp).sum(1), (value.flatten() - d["v"][ix])**2,
                                      p[rows, best], gain * r.float(), gain), 1))
        scores = torch.cat(rows_).mean(0).cpu().tolist()
        result[name] = {"searchCE": scores[0], "valueMSE": scores[1], "topMoveTargetMass": scores[2],
                        "supportedPairAccuracy": scores[3]/scores[4] if scores[4] > 0 else None}
    return result


def fit(args, device):
    if args.output.exists():
        raise ValueError("Use a fresh root for immutable candidates")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    packed, student, books = student_from(args.input, args.base, device)
    biases = trainable_biases(packed, student) if args.biases else {}
    if args.policy_head_only:
        books, biases = policy_head_parameters(student, books, biases)
    if args.recode_every:
        adapt_assignments(student, books)
    codes = {k.removesuffix("centers")+"bytes":book for k,book in books.items()} if args.recode_every else {}
    permitted = set(books) | set(biases) | set(codes)
    fingerprint = layout_hash(packed, permitted)
    fixtures = torch.tensor([c["encoded"] for c in json.loads(args.fixtures.read_bytes())["cases"][:12]],
                            dtype=torch.float32)
    check_parity(packed.SerializeToString(), student, fixtures, 1)
    original_value = (value_fingerprint(TokenGraph(packed).build().SerializeToString(), fixtures.numpy())
                      if args.policy_head_only else None)
    student.to(device)
    anchor = copy.deepcopy(student).eval().requires_grad_(False) if args.anchor else None
    pools, manifest = load_pools([args.data, *args.extra_data], args.mix, device, args.seed, args.orbit_seed)
    if manifest["base"] != file_hash(args.base):
        raise ValueError("Mined replay base differs from training base")
    faults = {}
    if args.proof_error_fraction:
        for name, _, d, train, _, _ in pools:
            if name not in ("exact","proof"):
                continue
            candidates = train[d["gain"][train] > 0]
            eligible = []
            for ix in candidates.split(2048):
                if not len(ix):
                    continue
                x = d["x"][ix]
                open_boards = ~x[:,243:270].reshape(-1,3,9).bool().any(1)
                empty = ~(x[:,:81] | x[:,81:162]).bool().reshape(-1,9,9)
                remaining = (empty & open_boards[:,:,None]).sum((1,2))
                eligible.append(ix[remaining > 14])
            if eligible:
                faults[name] = torch.cat(eligible)
    state_ix, policy_ix = [torch.as_tensor(np.argsort(np.stack(ax), axis=1), device=device)
                           for ax in zip(*_symmetry_maps())]
    policy_dest = policy_ix.argsort(1)
    counts = np.array([pool[1] * args.batch_size for pool in pools])
    sizes = np.floor(counts).astype(int)
    sizes[np.argsort(-(counts - sizes))[:args.batch_size - sizes.sum()]] += 1
    if (sizes < 1).any():
        raise ValueError("Batch too small for source mixture")
    parameters = [book.centers for book in books.values()] + list(biases.values())
    groups = [{"params": parameters, "lr":args.lr}]
    if args.recode_every:
        residuals = [book.residual for book in books.values()]
        groups.append({"params":residuals, "lr":args.assignment_lr})
        parameters = parameters + residuals
    optimizer = torch.optim.Adam(groups, lr=args.lr)
    steps = sorted(set(args.at))
    horizon = args.schedule_steps or steps[-1]
    if horizon < steps[-1]:
        raise ValueError("The learning-rate schedule cannot end before the final snapshot")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, horizon, eta_min=args.lr/8)
    settings = {"inputSha256":file_hash(args.input), "baseSha256":file_hash(args.base),
                "dataManifestSha256":[file_hash(p/"manifest.json") for p in [args.data,*args.extra_data]], "seed":args.seed,
                "mixture": {name:float(fraction) for name, fraction in args.mix}, "orbitSeed":args.orbit_seed,
                "steps":steps, "scheduleSteps":horizon, "batchSize":args.batch_size, "learningRate":args.lr,
                "hardFraction":args.hard_fraction, "proofErrorFraction":args.proof_error_fraction, "rankingWeight":args.margin,
                "teacherScale":args.teacher_scale, "valueWeight":args.value_weight, "anchor":args.anchor,
                "trainBiases":args.biases, "recodeEvery":args.recode_every, "assignmentLR":args.assignment_lr,
                "stepCodeFraction":args.step_code_fraction, "totalCodeFraction":args.total_code_fraction}
    if args.policy_head_only:
        settings.update(policyHeadOnly=True, inputValueFingerprint=original_value)
    initial_metrics = validation(student, pools)
    emit({"initialDiagnostics": initial_metrics, "trainingRows": sum(len(p[3]) for p in pools),
          "heldOutRows": sum(len(p[5]) for p in pools), "sampleCounts": sizes.tolist(),
          "hardFraction": args.hard_fraction, "rankingWeight": args.margin, "biasTables": len(biases),
          "provedErrorsBeyondSolver": {k:len(value) for k,value in faults.items()}})
    for step in range(1, steps[-1] + 1):
        batches = []
        for (name, _, d, train, hard, _), n in zip(pools, sizes):
            mistakes = faults.get(name)
            ne = round(n*args.proof_error_fraction) if mistakes is not None and len(mistakes) else 0
            nh = round((n-ne)*args.hard_fraction)
            samples = [train[torch.randint(len(train),(n-nh-ne,),device=device)],
                       hard[torch.randint(len(hard),(nh,),device=device)]]
            if ne:
                samples.append(mistakes[torch.randint(len(mistakes),(ne,),device=device)])
            ix = torch.cat(samples)
            batch = {k: t[ix] for k, t in d.items() if k != "hard"}
            batch["exact"] = torch.full((n,), name in ("exact","proof"), device=device, dtype=torch.bool)
            # Replay game outcomes are noisier than fully searched root targets.
            base_mix = {"exact": 0, "proof": 0, "search": .32, "deep": .32, "paper": .44,
                        "selfplay": .52, "replay": .65, "oracle": .45, "action": .12}.get(name, .45)
            batch["mix"] = torch.full((n, 1), min(1, base_mix * args.teacher_scale), device=device)
            batches.append(batch)
        d = {k: torch.cat([part[k] for part in batches]) for k in batches[0]}
        sym = torch.randint(8, (args.batch_size,), device=device)
        x = d["x"].gather(1, state_ix[sym]).float()
        p, ref = [d[k].gather(1, policy_ix[sym]) for k in ("p", "teacher")]
        chosen, better = [policy_dest[sym, d[k].long()] for k in ("chosen", "better")]
        logits, value = student(x)
        ce, mse, rank = objective(logits, value, x, p, d["v"][:, None], ref, d["teacher_v"][:, None],
                                  chosen, better, d["gain"], d["mix"], d["exact"], args.margin)
        loss = ce + args.value_weight * mse + rank
        if anchor is not None:
            # Protect the starting policy on all legal moves while the curated
            # Q targets change the relative preference of supported actions.
            with torch.no_grad():
                initial_logits, initial_value = anchor(x)
                initial_p = probability(initial_logits, x[:,162:243] > 0)
            logp = logits.float().masked_fill(x[:,162:243] == 0, -1e9).log_softmax(1)
            anchor_kl = (initial_p * (initial_p.clamp_min(1e-20).log() - logp)).sum(1).mean()
            anchor_loss = anchor_kl + args.value_weight * F.mse_loss(value, initial_value)
            loss = torch.lerp(loss, anchor_loss, args.anchor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if args.recode_every and (step % args.recode_every == 0 or step in steps):
            changed = sum(book.reassign(args.step_code_fraction, args.total_code_fraction) for book in books.values()).item()
            if step % (4*args.recode_every) == 0 or step in steps:
                emit({"recodingStep":step, "changedAssignments":changed})
        if step == 1 or step % 250 == 0:
            emit({"step":step, "policy":float(ce.detach()), "value":float(mse.detach()), "ranking":float(rank.detach())})
        if step in steps:
            update_centroids(packed, books)
            for t in packed.graph.initializer:
                if t.name in biases:
                    t.raw_data = biases[t.name].detach().cpu().numpy().astype("<f4").tobytes()
                elif t.name in codes:
                    updated = codes[t.name].packed_bytes()
                    if len(updated) != len(t.raw_data):
                        raise ValueError("Integer assignment storage length changed")
                    t.raw_data = updated
            if layout_hash(packed, permitted) != fingerprint:
                raise ValueError("Graph changed beyond allowed initializer bytes")
            payload = packed.SerializeToString()
            parity = check_parity(payload, copy.deepcopy(student), fixtures, 1)
            optimized = TokenGraph(packed).build().SerializeToString()
            if len(optimized) > args.max_bytes:
                raise ValueError(f"Optimized model needs {len(optimized)} bytes")
            graph_parity = verify(payload, optimized, fixtures.numpy(), 1)
            if args.policy_head_only and value_fingerprint(optimized, fixtures.numpy()) != original_value:
                raise ValueError('The policy-only update changed the production value output')
            name = f"{args.name}-{step}"
            immutable(args.output/(name+".onnx"), lambda p: p.write_bytes(optimized))
            metrics = {"training":settings, "step":step, "bytes":len(optimized), "sha256":hashlib.sha256(optimized).hexdigest(),
                       "torchParity":parity, "graphParity":graph_parity, "diagnostics":validation(student,pools)}
            if codes:
                changed = sum(torch.count_nonzero(book.codes.flatten()[:book.size] !=
                                     book.initial_codes.flatten()[:book.size]) for book in codes.values())
                metrics["assignmentFractionChanged"] = changed.item() / sum(book.size for book in codes.values())
            if args.save_raw:
                immutable(args.output/(name+"-raw.onnx"), lambda p:p.write_bytes(payload))
            emit(metrics)
            immutable(args.output/(name+".json"), lambda p: p.write_text(json.dumps(metrics, indent=2)+"\n"))
            if args.guard and "deep" in initial_metrics:
                ref, current = initial_metrics["deep"], metrics["diagnostics"]["deep"]
                if (current["valueMSE"] > 1.6*ref["valueMSE"] + .01 or
                        current["searchCE"] > ref["searchCE"] + .12):
                    raise ValueError("Candidate failed its frozen deep-search validation guard")
            if args.candidate_dir:
                args.candidate_dir.mkdir(parents=True, exist_ok=True)
                os.link(args.output/(name+".onnx"), args.candidate_dir/(name+".onnx"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exclusion = sub.add_parser("filter", help="Exclude protected roots and every one-move predecessor")
    exclusion.add_argument("--data", required=True, type=Path)
    sets = exclusion.add_mutually_exclusive_group(required=True)
    sets.add_argument("--excluded", type=Path, help="N × 35 uint8 packed states")
    sets.add_argument("--published-suite", type=Path, help="Pinned upstream evaluation_uttt_states.py")
    exclusion.add_argument("--output", required=True, type=Path)
    m = sub.add_parser("mine")
    m.add_argument("--source", nargs=3, action="append", required=True, metavar=("NAME","DIR","ROWS"))
    m.add_argument("--teacher", type=Path, required=True)
    m.add_argument("--proof-manifest-sha256", help="Pinned all-action alternatives manifest for a source named proof")
    f = sub.add_parser("fit")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--extra-data", type=Path, action="append", default=[], help="Additional matched mined source directories")
    f.add_argument("--mix", nargs=2, action="append", required=True, metavar=("SOURCE","FRACTION"))
    f.add_argument("--name", default="regret-student")
    f.add_argument("--candidate-dir", type=Path, help="Optional local directory to link each verified export")
    f.add_argument("--at", nargs="+", type=int, required=True)
    f.add_argument("--schedule-steps", type=int, help="Match a longer cosine schedule while ending at an early snapshot")
    f.add_argument("--orbit-seed", type=int, help="Use a common symmetry-invariant validation partition across sources")
    f.add_argument("--proof-error-fraction", type=float, default=0, help="Within each proof batch, fraction devoted to proved incumbent errors beyond 14 playable squares")
    f.add_argument("--hard-fraction", type=float, default=.45)
    f.add_argument("--anchor", type=float, default=0, help="Mixture weight for frozen initial-model policy KL and value MSE")
    f.add_argument("--teacher-scale", type=float, default=1.0)
    f.add_argument("--margin", type=float, default=.25)
    f.add_argument("--lr", type=float, default=8e-5)
    f.add_argument("--value-weight", type=float, default=.9)
    f.add_argument("--biases", action="store_true")
    f.add_argument("--policy-head-only", action="store_true",
                   help="Freeze the shared trunk and value path and verify unchanged production value outputs")
    f.add_argument("--recode-every", type=int, default=0, help="Period between weight-index updates; zero fixes original indices")
    f.add_argument("--assignment-lr", type=float, default=.00001)
    f.add_argument("--step-code-fraction", type=float, default=.0005, help="Maximum reassigned fraction of each tensor per update")
    f.add_argument("--total-code-fraction", type=float, default=.035, help="Maximum fraction differing from the input tensor's original codes")
    f.add_argument("--guard", action="store_true", help="Stop before candidate publication if deep-search diagnostics collapse")
    f.add_argument("--save-raw", action="store_true", help="Retain raw packed graph for follow-on refinement")
    f.add_argument("--max-bytes", type=int, default=1_480_935)
    f.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("parity.json"))
    for cmd in (m, f):
        cmd.add_argument("--base", type=Path, required=True)
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--seed", type=int, default=741091)
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        cmd.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "filter":
        filter_mined(args)
        return
    if args.batch_size <= 0:
        parser.error("A positive batch size is required")
    if args.command == "fit" and (not all(math.isfinite(n) for n in (args.lr, args.value_weight,
            args.teacher_scale, args.margin, args.hard_fraction, args.assignment_lr, args.anchor)) or
            not args.name.replace("-", "").isalnum() or min(args.at) <= 0 or
            min(args.lr, args.value_weight, args.max_bytes) <= 0 or not 0 <= args.hard_fraction <= 1 or
            not 0 <= args.teacher_scale <= 2 or not 0 <= args.anchor < 1 or args.margin < 0 or args.recode_every < 0 or args.assignment_lr <= 0 or
            not 0 <= args.proof_error_fraction <= 1 or not 0 < args.step_code_fraction <= 1 or not 0 < args.total_code_fraction <= 1 or
            (args.orbit_seed is not None and not 0 <= args.orbit_seed < 2**32)):
        parser.error("Invalid training settings")
    torch.set_num_threads(2)
    (mine if args.command == "mine" else fit)(args, torch.device(args.device))


if __name__ == "__main__":
    main()
