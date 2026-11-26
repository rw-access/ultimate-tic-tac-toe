"""Pack a proof specialist while preserving every frozen Tiny kernel's original INT8 groups."""

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from training.pack_weights import eligible, make_graph, quantize_int8, verify_int8_reconstruction


def shorten_internal_names(model):
    graph = model.graph
    if graph.sparse_initializer or graph.quantization_annotation or model.functions:
        raise ValueError("Expected a dense single-graph model")
    if any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in graph.node for attr in node.attribute):
        raise ValueError("Subgraph tensor names require separate namespaces")
    public = {value.name for value in (*graph.input, *graph.output)}
    renamed = {}

    def shorten(name):
        if not name or name in public:
            return name
        if name not in renamed:
            new = f"z{len(renamed)}"
            if new in public:
                raise ValueError("Public tensor name conflicts with internal name")
            renamed[name] = new
        return renamed[name]

    for initializer in graph.initializer:
        initializer.name = shorten(initializer.name)
    for node in graph.node:
        node.ClearField("name")
        for values in (node.input, node.output):
            rewritten = [shorten(name) for name in values]
            del values[:]
            values.extend(rewritten)
    for value in graph.value_info:
        value.name = shorten(value.name)
    onnx.checker.check_model(model)


def mixed_weights(model, family="proof"):
    tensors = eligible(model)
    specialist_inputs = {name for node in model.graph.node if "late_policy" in node.name or "late_value" in node.name
                          for name in node.input}
    expected = {t.name: t for t in copy.deepcopy(model).graph.initializer}
    arrays = {}
    groups = {}
    for tensor in tensors:
        weights = numpy_helper.to_array(tensor)
        group = 8
        if tensor.name in specialist_inputs:
            # A divisor leaves the decoder's final reshape independent of group size.
            group = next((g for g in (128, 96, 64, 48, 32, 24, 16, 8) if weights.size % g == 0), None)
            if group is None:
                raise ValueError(f"Cannot group specialist matrix {tensor.name}")
            groups[tensor.name] = group
        codes, scales, reconstructed = quantize_int8(weights, group)
        if tensor.name in groups and codes.size != weights.size:
            raise ValueError("Unexpected padded specialist matrix")
        arrays[tensor.name] = codes, scales
        expected[tensor.name].CopyFrom(numpy_helper.from_array(reconstructed, tensor.name))
    expected_count = {"proof": 4, "routing": 5}[family]
    if len(groups) != expected_count:
        raise ValueError(f"Expected {expected_count} {family} specialist matrices")
    candidate = make_graph(model, tensors, 8, 8, arrays)
    verify_int8_reconstruction(model, candidate, tensors, expected)
    return candidate, groups


def example_states(directory, count, seed, family="proof"):
    rng = np.random.default_rng(seed)
    early, late = [], []
    for path in rng.permutation(sorted(directory.glob("*.npz"))).tolist():
        with np.load(path, allow_pickle=False) as shard:
            x = shard["x"]
        if x.dtype != np.uint8 or x.ndim != 2 or x.shape[1] != 280:
            raise ValueError("Expected compact state shards")
        open_boards = 1 - x[:, 243:270].reshape(-1, 3, 9).sum(1)
        empties = (1 - x[:, :81] - x[:, 81:162]).reshape(-1, 9, 9).sum(2)
        remaining = (empties * open_boards).sum(1)
        if family == "routing":
            ply = x[:, :162].sum(1)
            unchanged = (ply <= 7) | (ply >= 44) | (remaining <= 18)
        else:
            unchanged = remaining >= 48
        early.extend(x[np.flatnonzero(unchanged)[:40]])
        late.extend(x[np.flatnonzero(~unchanged)[:40]])
        if len(early) >= count and len(late) >= count:
            return np.stack(early[:count]).astype(np.float32), np.stack(late[:count]).astype(np.float32)
    raise ValueError("Not enough early and late positions for export verification")


def max_error(first, second, x):
    config = ort.SessionOptions()
    config.intra_op_num_threads = config.inter_op_num_threads = 1
    config.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sessions = [ort.InferenceSession(blob, config, providers=["CPUExecutionProvider"]) for blob in (first, second)]
    error = np.zeros(2)
    for start in range(0, len(x), 24):
        feeds = {"state": x[start:start + 24]}
        a, b = [engine.run(["policy", "value"], feeds) for engine in sessions]
        for i in (0, 1):
            if a[i].shape != b[i].shape or not np.isfinite(a[i]).all() or not np.isfinite(b[i]).all():
                raise ValueError("Invalid model output")
            error[i] = max(error[i], np.max(np.abs(a[i] - b[i])))
    return error.tolist()


def pack(args):
    if args.output.exists() or args.report.exists() or args.output.resolve() == args.report.resolve():
        raise ValueError("Expected separate fresh model and report paths")
    if args.count < 1 or args.max_bytes < 1:
        raise ValueError("Sample and byte limits must be positive")
    source = args.input.read_bytes()
    incumbent = args.incumbent.read_bytes()
    if hashlib.sha256(incumbent).hexdigest() != args.incumbent_sha256:
        raise ValueError("Incumbent hash mismatch")
    model, groups = mixed_weights(onnx.load_model_from_string(source), args.family)
    before = model.SerializeToString()
    shorten_internal_names(model)
    after = model.SerializeToString()
    if len(after) > args.max_bytes:
        raise ValueError(f"Packed specialist exceeds cap ({len(after):,} bytes)")
    early, late = example_states(args.data, args.count, args.seed, args.family)
    rename_error = max_error(before, after, np.concatenate((early, late)))
    early_error = max_error(incumbent, after, early)
    if any(rename_error) or any(early_error):
        raise ValueError(f"Exact output verification failed: graph {rename_error}, incumbent {early_error}")
    result = dict(bytes=len(after), sha256=hashlib.sha256(after).hexdigest(), frozenGroup=8,
                  specialistGroups=groups, verifiedEarlyStates=len(early), verifiedLateStates=len(late),
                  incumbentSha256=args.incumbent_sha256, inputSha256=hashlib.sha256(source).hexdigest())
    created = []
    try:
        for destination, payload in [(args.output, after), (args.report, (json.dumps(result, indent=2) + "\n").encode())]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.link(temporary.name, destination)
                created.append(destination)
    except OSError:
        for destination in reversed(created):
            destination.unlink()
        raise
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "incumbent", "data", "output", "report"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--family", choices=("proof", "routing"), default="proof")
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=26316511)
    parser.add_argument("--max-bytes", type=int, default=477280)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
