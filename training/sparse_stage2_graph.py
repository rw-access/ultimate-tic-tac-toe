"""Evaluate the Stage 2 policy decoder on playable boards or legal cells only.

Accepts position-major, factored Stage 2 exports from training.fast_stage2_graph.
All weights, the value function and legal policy logits are retained. Skipped
logits are zero. Terminal states retain the complete policy to avoid empty
matrix multiplications. The compressed weights keep their original payloads.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .pack_weights import load_states


class SparseDecoder:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self.source = list(self.graph.node)
        self.arrays = {t.name: t for t in self.graph.initializer}
        self.producers = {out: node for node in self.source for out in node.output}
        self.constants = {}
        self.counter = 0

    def parent(self, value, kind):
        node = self.producers.get(value)
        if node is None or node.op_type != kind or node.domain:
            raise ValueError(f"Expected {kind} producer for {value}")
        return node

    def dimensions(self, reshape):
        return numpy_helper.to_array(self.arrays[reshape.input[1]]).tolist()

    def linear_input(self, name):
        bias = f"layers.{name}.bias"
        matches = [n for n in self.source if n.op_type == "Add" and n.input[1] == bias]
        if len(matches) != 1:
            raise ValueError(f"Expected one position-major affine layer for {name}")
        return self.parent(matches[0].input[0], "MatMul").input[0]

    def node(self, kind, *inputs, output=None, **attrs):
        self.counter += 1
        output = output or f"sparse/{self.counter}"
        self.graph.node.append(helper.make_node(kind, list(inputs), [output], **attrs))
        return output

    def constant(self, value, dtype=np.int64):
        array = np.asarray(value, dtype=dtype)
        key = (array.shape, array.dtype.str, array.tobytes())
        if key not in self.constants:
            name = f"sparse/constant/{len(self.constants)}"
            self.graph.initializer.append(numpy_helper.from_array(array, name))
            self.constants[key] = name
        return self.constants[key]

    def reshape(self, value, shape, output=None):
        return self.node("Reshape", value, self.constant(shape), output=output)

    def indices(self, flags):
        return self.reshape(self.node("NonZero", self.reshape(flags, [-1])), [-1])

    def convert(self, mode):
        if mode not in ("board", "cell"):
            raise ValueError("Expected board or cell mode")
        if ([x.name for x in self.graph.input] != ["state"]
                or [x.name for x in self.graph.output] != ["policy", "value"]):
            raise ValueError("Expected the project state/policy/value interface")
        context = self.parent(self.linear_input("policy"), "Concat")
        logits = self.parent(self.linear_input("logits"), "Concat")
        cells = self.parent(logits.input[0], "Concat")
        if len(context.input) != 3 or len(logits.input) != 2 or len(cells.input) != 4:
            raise ValueError("Unrecognized Stage 2 policy features")
        legal = self.parent(self.parent(cells.input[3], "Reshape").input[0], "Slice")
        if (legal.input[0] != "state" or [numpy_helper.to_array(self.arrays[k]).tolist()
                for k in legal.input[1:4]] != [[162], [243], [1]]):
            raise ValueError("Expected project legal action mask in features 162:243")
        action = self.parent(logits.input[1], "Elu")
        post_offset = self.parent(action.input[0], "Reshape")
        offset = self.parent(post_offset.input[0], "Add")
        pre_offset = self.parent(offset.input[0], "Reshape")
        expand = self.parent(pre_offset.input[0], "MatMul")
        pre_expand = self.parent(expand.input[0], "Reshape")
        cell_shape = self.dimensions(pre_expand)
        if (len(cell_shape) != 3 or cell_shape[:2] != [-1, 81] or cell_shape[2] <= 0
                or self.dimensions(pre_offset) != [-1, 9, 9, 256]
                or self.dimensions(post_offset) != [-1, 81, 256]
                or list(self.arrays[offset.input[1]].dims) != [9, 256]):
            raise ValueError("Expected the factored 3-by-3 Stage 2 decoder with a per-cell offset")
        policy = self.parent("policy", "Reshape")
        if self.dimensions(policy) != [-1, 81]:
            raise ValueError("Unexpected policy output shape")
        legal_input = legal.output[0]
        # Keep the complete constant weight-decompression branch ahead of the
        # runtime graph. ORT folds it once when a session opens.
        fixed = set(self.arrays)
        del self.graph.node[:]
        for node in self.source:
            if all(not name or name in fixed for name in node.input):
                self.graph.node.append(node)
                fixed.update(node.output)
        cell_mode = mode == "cell"
        for node in self.source:
            out = node.output[0]
            if out in fixed:
                continue
            if node is context:
                dense = self.reshape(legal_input, [-1, 9])
                active = self.node("ReduceMax", legal_input, axes=[1], keepdims=1)
                terminal = self.node("Sub", self.constant(1, np.float32), active)
                mask = self.node("Add", legal_input, terminal)
                board_mask = self.reshape(mask, [-1, 9])
                boards = self.indices(self.node("ReduceMax", board_mask, axes=[1], keepdims=0))
                all_context = self.node("Concat", *node.input, axis=2)
                chosen = self.node("Gather", self.reshape(all_context, [-1, 640]), boards, axis=0)
                self.reshape(chosen, [1, -1, 640], output=out)
                features = self.node("Gather", self.reshape(cells.output[0], [-1, 36]), boards, axis=0)
                if cell_mode:
                    selected_mask = self.node("Gather", board_mask, boards, axis=0)
                    selected_cells = self.indices(selected_mask)
                    output_cells = self.indices(mask)
                    destinations = self.node("Mod", selected_cells, self.constant(9))
                    features = self.node("Gather", self.reshape(features, [-1, 4]), selected_cells, axis=0)
                cell_features = self.reshape(features, [1, -1, 4])
            elif node is pre_expand:
                decoded = self.reshape(node.input[0], [-1, cell_shape[2]])
                if cell_mode:
                    decoded = self.node("Gather", decoded, selected_cells, axis=0)
                self.reshape(decoded, [1, -1, cell_shape[2]], output=out)
            elif node is pre_offset:
                self.reshape(node.input[0], [-1, 256] if cell_mode else [-1, 9, 256], output=out)
            elif node is offset and cell_mode:
                selected_bias = self.node("Gather", node.input[1], destinations, axis=0)
                self.node("Add", node.input[0], selected_bias, output=out)
            elif node is post_offset:
                self.reshape(node.input[0], [1, -1, 256], output=out)
            elif node is logits:
                self.node("Concat", cell_features, node.input[1], axis=2, output=out)
            elif node is policy:
                if cell_mode:
                    values = self.node("ScatterND", self.reshape(legal_input, [-1]),
                                       self.reshape(output_cells, [-1, 1]), self.reshape(node.input[0], [-1]))
                else:
                    values = self.node("ScatterND", dense, self.reshape(boards, [-1, 1]),
                                       self.reshape(node.input[0], [-1, 9]))
                self.reshape(values, [-1, 81], output=out)
            else:
                self.graph.node.append(node)
        self.compact()
        onnx.checker.check_model(self.model)
        return self.model

    def compact(self):
        # Drop superseded shapes and shorten private tensor names to fit the
        # source byte budget. Public ONNX names and tensor payloads are stable.
        needed = {x.name for x in self.graph.output}
        nodes = []
        for node in reversed(self.graph.node):
            if needed.intersection(node.output):
                needed.update(node.input)
                nodes.append(node)
        names = {}
        public = {x.name for x in self.graph.input} | {x.name for x in self.graph.output}

        def rename(name):
            if not name or name in public:
                return name
            if name not in names:
                names[name] = f"a{len(names):x}"
            return names[name]

        arrays = [t for t in self.graph.initializer if t.name in needed]
        if any(k.endswith(("/bytes", "/centers", "/fp16")) and k not in needed for k in self.arrays):
            raise ValueError("An original packed parameter was disconnected")
        del self.graph.initializer[:]
        self.graph.initializer.extend(arrays)
        for tensor in self.graph.initializer:
            tensor.name = rename(tensor.name)
        del self.graph.node[:]
        self.graph.node.extend(reversed(nodes))
        del self.graph.value_info[:]
        for node in self.graph.node:
            node.name = ""
            node.input[:] = [rename(k) for k in node.input]
            node.output[:] = [rename(k) for k in node.output]


def verify(source, candidate, states, mode, threads=1):
    import onnxruntime as ort

    if len(states) < 12:
        raise ValueError("At least 12 fixture positions required")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(m, options, providers=["CPUExecutionProvider"])
                for m in (source, candidate)]
    maximum = np.zeros(2)
    batches = [states[:1], states[:12]] + [states[i:i + 32] for i in range(12, len(states), 32)]
    terminal = states[~states[:, 162:243].any(1)]
    if len(terminal):
        batches.append(terminal[:1])
    for batch in batches:
        (p, v), (q, w) = [s.run(["policy", "value"], {"state": batch}) for s in sessions]
        if (p.shape != q.shape or p.shape != (len(batch), 81) or v.shape != w.shape
                or v.shape != (len(batch), 1) or not all(np.isfinite(x).all() for x in (p, q, v, w))):
            raise ValueError("Invalid policy/value outputs")
        legal = batch[:, 162:243].astype(bool)
        retained = legal | ~legal.any(1, keepdims=True)
        if mode == "board":
            retained = np.repeat(retained.reshape(-1, 9, 9).any(2), 9, axis=1)
        maximum = np.maximum(maximum, [np.max(np.abs(p[retained] - q[retained])), np.max(np.abs(v - w))])
        np.testing.assert_array_equal(q[~retained], 0)
        np.testing.assert_array_equal(np.where(legal, p, -np.inf).argmax(1), np.where(legal, q, -np.inf).argmax(1))
    if np.any(maximum >= 1e-4):
        raise ValueError(f"Policy/value drift above tolerance: {maximum}")
    return {"positions": len(states), "maxRetainedLogitError": maximum[0], "maxValueError": maximum[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cell", "board"), default="cell")
    parser.add_argument("--states", "--parity", type=Path, default=Path(__file__).with_name("parity.json"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=713741)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=1_480_935)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        parser.error("Choose a fresh output file")
    if min(args.samples, args.threads, args.max_bytes) < 1:
        parser.error("Threads, sample count and maximum bytes must be positive")
    source = args.input.read_bytes()
    candidate = SparseDecoder(onnx.load_from_string(source)).convert(args.mode).SerializeToString()
    if len(candidate) > args.max_bytes:
        raise ValueError(f"Output {len(candidate):,} exceeds cap {args.max_bytes:,}")
    states = load_states(args.states, args.data, args.samples, args.seed)
    metrics = verify(source, candidate, states, args.mode, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "candidate.onnx"
        staged.write_bytes(candidate)
        os.link(staged, args.output)
    print(json.dumps({"mode": args.mode, "inputBytes": len(source), "outputBytes": len(candidate),
                      "inputSha256": hashlib.sha256(source).hexdigest(),
                      "outputSha256": hashlib.sha256(candidate).hexdigest(), "parity": metrics}))


if __name__ == "__main__":
    main()
