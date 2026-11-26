"""Rewrite the project's exported Stage 2 models in board-major token order.

Convolution weights and packed code assignments are preserved. Static kernel
transposes/reshapes are eligible for constant folding. The run-time graph uses
matrix multiplications for 1x1 convolutions and nonoverlapping 3x3 lifts.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .pack_weights import load_states


class TokenGraph:
    def __init__(self, source):
        self.source = source
        graph = source.graph
        self.arrays = {t.name: t for t in graph.initializer}
        self.definitions = {o: n for n in graph.node for o in n.output}
        self.layers = {n.input[1]: n for n in graph.node if n.op_type in ("Conv", "ConvTranspose")}
        self.nodes = []
        self.constants = {}
        self.counter = 0
        self.used_layers = set()

    def node(self, kind, *inputs, output=None, **attrs):
        self.counter += 1
        value = output or f"z/{self.counter}"
        self.nodes.append(helper.make_node(kind, list(inputs), [value], **attrs))
        return value

    def constant(self, value, dtype=np.int64):
        a = np.asarray(value, dtype=dtype)
        key = (a.dtype.str, a.shape, a.tobytes())
        if key not in self.constants:
            name = f"zc/{len(self.constants)}"
            self.constants[key] = numpy_helper.from_array(a, name)
        return self.constants[key].name

    def reshape(self, value, shape, output=None):
        return self.node("Reshape", value, self.constant(shape), output=output)

    def shape(self, value):
        if value in self.arrays:
            return tuple(self.arrays[value].dims)
        n = self.definitions[value]
        if n.op_type == "Reshape" and n.input[1] in self.arrays:
            return tuple(numpy_helper.to_array(self.arrays[n.input[1]]).tolist())
        if n.op_type in ("Cast", "Identity"):
            return self.shape(n.input[0])
        raise ValueError(f"No fixed Stage 2 kernel shape for {value}")

    def convolution(self, name, kind):
        weight = "layers." + name + ".weight"
        n = self.layers.get(weight)
        if n is None or n.op_type != kind or n.domain not in ("", "ai.onnx"):
            raise ValueError(f"Missing expected {kind} {weight}")
        dims = self.shape(weight)
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        stride = 3 if name in ("stem", "upsample", "upsample.lift") else 1
        group = dims[0] if name.startswith(("pool", "vpool")) else 1
        if (attrs.get("strides", [1, 1]) != [stride, stride] or attrs.get("group", 1) != group
                or any(attrs.get("pads", [])) or any(attrs.get("output_padding", []))
                or attrs.get("dilations", [1, 1]) != [1, 1] or "output_shape" in attrs
                or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"):
            raise ValueError(f"Unexpected stride, padding, or grouping in {name}")
        self.used_layers.add(weight)
        return n, weight, dims

    def add_bias(self, value, conv):
        return self.node("Add", value, conv.input[2]) if len(conv.input) > 2 and conv.input[2] else value

    def point(self, x, name):
        if f"layers.{name}.reduce.weight" in self.layers:
            return self.point(self.point(x, name + ".reduce"), name + ".expand")
        n, kernel, (m, d, h, w) = self.convolution(name, "Conv")
        if (h, w) != (1, 1):
            raise ValueError(f"Expected a pointwise kernel for {name}")
        matrix = self.node("Transpose", self.reshape(kernel, [m, d]), perm=[1, 0])
        return self.add_bias(self.node("MatMul", x, matrix), n)

    def pool(self, x, name):
        n, kernel, (d, inner, h, w) = self.convolution(name, "Conv")
        if (inner, h, w) != (1, 3, 3):
            raise ValueError(f"Expected depthwise mini-board pool for {name}")
        factors = self.node("Transpose", self.reshape(kernel, [d, 9]), perm=[1, 0])
        return self.add_bias(self.node("ReduceSum", self.node("Mul", x, factors),
                                       self.constant([1]), keepdims=1), n)

    def packed_offset(self, name):
        def initial_name(key):
            while key in self.definitions and self.definitions[key].op_type == "Identity":
                key = self.definitions[key].input[0]
            return key if key in self.arrays else None

        key = f"layers.{name}.offset"
        direct = initial_name(key)
        if direct:
            a = numpy_helper.to_array(self.arrays[direct])
        else:
            # ONNX exporters can fold the tile/unsqueeze to an unnamed initializer.
            # Follow the data branch from this specific expand convolution to its Add.
            expand = self.layers[f"layers.{name}.expand.weight"].output[0]
            queue = [expand]
            offsets = set()
            while queue:
                value = queue.pop()
                for n in self.source.graph.node:
                    if n.op_type == "Add" and value in n.input:
                        for s in n.input:
                            target = initial_name(s)
                            if s != value and target:
                                offsets.add(target)
                    elif n.op_type in ("Reshape", "Identity", "Transpose") and n.input[0] == value:
                        queue.append(n.output[0])
            if len(offsets) != 1:
                raise ValueError(f"Cannot identify a single static offset for {name}: {offsets}")
            a = numpy_helper.to_array(self.arrays[offsets.pop()])
        if a.dtype != np.float32 or a.shape[0:2] != (1, 256):
            raise ValueError(f"Unexpected {name} offset type or shape {a.dtype} {a.shape}")
        if a.size == 256 * 81 and name == "upsample":
            image = a.reshape(1, 256, 9, 9)
            tile = image[:, :, :3, :3]
            if not np.array_equal(image, np.tile(tile, (1, 1, 3, 3))):
                raise ValueError("Decoder offset is spatially varying across mini-boards")
        elif a.size == 256 * 9:
            tile = a.reshape(1, 256, 3, 3)
        else:
            raise ValueError(f"Unexpected spatial offset for {name}: {a.shape}")
        return self.constant(tile.reshape(256, 9).T, dtype=np.float32)

    def lift(self, x, name, boards):
        factored = f"layers.{name}.reduce.weight" in self.layers
        if factored:
            x = self.point(x, name + ".reduce")
        n, kernel, (d, m, h, w) = self.convolution(name + ".lift" if factored else name, "ConvTranspose")
        if (h, w) != (3, 3):
            raise ValueError(f"Unexpected transposed kernel for {name}")
        matrix = self.reshape(self.node("Transpose", kernel, perm=[0, 2, 3, 1]), [d, 9 * m])
        decoded = self.reshape(self.node("MatMul", x, matrix), [-1, boards * 9, m])
        decoded = self.add_bias(decoded, n)
        if factored:
            decoded = self.point(decoded, name + ".expand")
            if boards == 9:
                decoded = self.reshape(decoded, [-1, 9, 9, 256])
            decoded = self.node("Add", decoded, self.packed_offset(name))
            if boards == 9:
                decoded = self.reshape(decoded, [-1, 81, 256])
        return decoded

    def build(self):
        own, other, legal = [self.node("Slice", "state", self.constant([i]), self.constant([i + 81]),
                                       self.constant([1])) for i in (0, 81, 162)]
        turn = self.node("ReduceSum", self.node("Sub", own, other), self.constant([1]), keepdims=1)
        turn = self.node("Add", self.node("Mul", turn, self.constant(2, np.float32)),
                         self.constant(1, np.float32))
        turn = self.node("Expand", turn, self.node("Shape", own))
        cells = self.node("Concat", *(self.reshape(t, [-1, 81, 1]) for t in (own, other, turn, legal)), axis=2)
        n, kernel, dims = self.convolution("stem", "Conv")
        if dims != (128, 4, 3, 3):
            raise ValueError(f"Unexpected Stage 2 stem geometry: {dims}")
        stem_w = self.reshape(self.node("Transpose", kernel, perm=[2, 3, 1, 0]), [36, 128])
        stem = self.node("Elu", self.add_bias(self.node("MatMul", self.reshape(cells, [-1, 9, 36]), stem_w), n))
        trunk = self.node("Relu", self.point(self.node("Elu", self.point(stem, "early")), "trunk"))
        final = trunk
        for i in range(3):
            gate = self.node("Elu", self.point(final, f"gate{i}"))
            pooled = self.node("Elu", self.pool(gate, f"pool{i}"))
            final = self.node("Relu", self.node("Add", final, self.lift(pooled, f"lift{i}", 1)))
        context = self.node("Elu", self.point(self.node("Concat", stem, trunk, final, axis=2), "policy"))
        action = self.node("Elu", self.lift(context, "upsample", 9))
        hidden = self.node("Elu", self.point(self.node("Concat", cells, action, axis=2), "logits"))
        logits = self.node("Clip", self.point(hidden, "logit_out"), self.constant(-32, np.float32),
                           self.constant(32, np.float32))
        self.reshape(logits, [-1, 81], output="policy")
        v1 = self.node("Elu", self.pool(self.point(trunk, "v1"), "vpool1"))
        v2 = self.node("Elu", self.pool(self.point(final, "v2"), "vpool2"))
        v = self.node("Elu", self.point(self.node("Concat", v1, v2, axis=2), "value"))
        self.reshape(self.node("Tanh", self.point(v, "value_out")), [-1, 1], output="value")
        if self.used_layers != self.layers.keys():
            raise ValueError(f"Unrecognized extra convolutions: {self.layers.keys() - self.used_layers}")

        # Include only source nodes that define required static weights and biases.
        needed = {s for n in self.nodes for s in n.input}
        prefix = []
        for original in reversed(self.source.graph.node):
            if needed.intersection(original.output):
                if original.op_type in ("Conv", "ConvTranspose") or "state" in original.input:
                    raise ValueError("Source weight transform depends on game state")
                prefix.append(original)
                needed.update(s for s in original.input if s)
        keep = [t for name, t in self.arrays.items() if name in needed]
        for tensor in self.arrays.values():
            if tensor.name.endswith(("/bytes", "/centers", "/fp16")) and tensor.name not in needed:
                raise ValueError(f"Packed parameter removed: {tensor.name}")
        model = copy.deepcopy(self.source)
        graph = model.graph
        del graph.node[:]
        del graph.initializer[:]
        del graph.value_info[:]
        graph.initializer.extend(keep + list(self.constants.values()))
        graph.node.extend(list(reversed(prefix)) + self.nodes)
        onnx.checker.check_model(model)
        return model


def verify(source, candidate, states, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(m, options, providers=["CPUExecutionProvider"])
                for m in (source, candidate)]
    if len(states) < 12:
        raise ValueError("Expected at least 12 parity positions")
    max_errors = [0.0, 0.0]
    same_best = live = 0
    # Distinct single and 12-position runs verify dynamic batching. Remaining
    # data, when supplied, expands parity to independent gameplay states.
    for batch in [states[:1], states[:12]] + [states[i:i + 32] for i in range(12, len(states), 32)]:
        (p, v), (q, w) = [s.run(["policy", "value"], {"state": batch}) for s in sessions]
        for i, (a, b, width) in enumerate(((p, q, 81), (v, w, 1))):
            if (a.shape != (len(batch), width) or b.shape != a.shape
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError("Invalid native policy/value outputs")
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=5e-4)
            max_errors[i] = max(max_errors[i], float(np.max(np.abs(a - b))))
        legal = batch[:, 162:243].astype(bool)
        rows = np.where(legal.any(1))[0]
        if rows.size:
            expected = np.where(legal[rows], p[rows], -np.inf).argmax(1)
            chosen = np.where(legal[rows], q[rows], -np.inf).argmax(1)
            np.testing.assert_allclose(p[rows, expected], p[rows, chosen], rtol=4e-4, atol=1e-3)
            same_best += int(np.sum(expected == chosen))
            live += len(rows)
    return {"positions": len(states), "maxAbsPolicyValueErrors": max_errors,
            "legalBestMatches": same_best, "legalBestProbesIncludingSingleton": live}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=1_492_760)
    parser.add_argument("--parity", type=Path, default=Path(__file__).with_name("parity.json"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=713741)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        parser.error("Choose a fresh output path")
    if min(args.threads, args.samples, args.max_bytes) < 1:
        parser.error("Positive threads, sample count, and byte cap required")
    source = args.input.read_bytes()
    model = onnx.load_model_from_string(source)
    if [x.name for x in model.graph.input] != ["state"] or [x.name for x in model.graph.output] != ["policy", "value"]:
        parser.error("Expected the native project Stage 2 state/policy/value graph")
    candidate = TokenGraph(model).build().SerializeToString()
    if len(candidate) > args.max_bytes:
        raise ValueError(f"Rewritten graph exceeds cap: {len(candidate):,} > {args.max_bytes:,} bytes")
    states = load_states(args.parity, args.data, args.samples, args.seed)
    metrics = verify(source, candidate, states, args.threads)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "candidate.onnx"
        staged.write_bytes(candidate)
        os.link(staged, args.output)
    print(json.dumps({"inputBytes": len(source), "outputBytes": len(candidate),
                      "inputSha256": hashlib.sha256(source).hexdigest(),
                      "outputSha256": hashlib.sha256(candidate).hexdigest(), "parity": metrics}))


if __name__ == "__main__":
    main()
