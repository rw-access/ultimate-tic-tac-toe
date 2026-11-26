"""Pack frozen ONNX convolution/matrix weights, retaining float32 activations.

Store float16 weights, signed int8 blocks with float32 scales, or 2-bit and
4-bit indices with float16 codebooks. Standard ONNX constant operations materialize the float32
kernels when the inference session optimizes the graph. The storage cap refers to the serialized, standalone ONNX file.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def byte_lookup(bits):
    shifts = np.arange(8 // bits) * bits
    return ((np.arange(256)[:, None] >> shifts) & ((1 << bits) - 1)).astype(np.int32)


def layout(size, group_size, bits):
    factor = 8 // bits
    if group_size <= 0 or group_size % factor:
        raise ValueError(f"group size must be a positive multiple of {factor}")
    width = min(group_size, math.ceil(size / factor) * factor)
    return math.ceil(size / width), width


def nearest(values, centers):
    # Sorted thresholds allow exact nearest-centroid assignment, including ties.
    thresholds = (centers[:, :-1] + centers[:, 1:]) * 0.5
    return np.sum(values[:, :, None] > thresholds[:, None, :], axis=2, dtype=np.uint8)


def check_float16_range(values):
    if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
        raise ValueError("Cannot encode nonfinite weights or centroids beyond float16 range")


def quantize(weight, bits, group_size, iterations):
    values = np.asarray(weight, dtype=np.float32).ravel()
    check_float16_range(values)
    groups, width = layout(len(values), group_size, bits)
    levels = 1 << bits
    packed = np.empty(groups * width // (8 // bits), dtype=np.uint8)
    book = np.empty((groups, levels), dtype=np.float16)
    reconstructed = np.empty(len(values), dtype=np.float32)
    shifts = np.arange(8 // bits) * bits
    for start in range(0, groups, 128):
        stop = min(start + 128, groups)
        segment = values[start * width:stop * width]
        valid = np.arange((stop - start) * width).reshape(-1, width) < segment.size
        chunk = np.pad(segment, (0, valid.size - segment.size), mode="edge").reshape(-1, width)
        ordered = np.sort(chunk, axis=1)
        # For a partial final group, padded elements must not affect the fit.
        lengths = valid.sum(axis=1)
        if lengths[-1] < width:
            ordered[-1, :lengths[-1]] = np.sort(chunk[-1, :lengths[-1]])
        first = ordered[:, :1]
        last = ordered[np.arange(len(chunk)), lengths - 1, None]
        affine = first + (last - first) * np.linspace(0, 1, levels, dtype=np.float32)
        quantiles = ((np.arange(levels) + 0.5) / levels * lengths[:, None]).astype(int)
        starts = (affine, np.take_along_axis(ordered, quantiles, axis=1))
        offsets = (np.arange(len(chunk)) * levels)[:, None]
        best_err = np.full(len(chunk), np.inf)
        best_table = np.empty((len(chunk), levels), np.float16)
        best_codes = np.empty(chunk.shape, np.uint8)
        for centers in starts:
            for step in range(iterations + 1):
                rounded = centers.astype(np.float16).astype(np.float32)
                codes = nearest(chunk, rounded)
                decoded = np.take_along_axis(rounded, codes, axis=1)
                errors = np.sum(np.square(chunk - decoded) * valid, axis=1, dtype=np.float64)
                better = errors < best_err
                best_err[better] = errors[better]
                best_table[better] = rounded[better]
                best_codes[better] = codes[better]
                if step == iterations:
                    break
                tags = (offsets + codes)[valid]
                counts = np.bincount(tags, minlength=centers.size).reshape(centers.shape)
                sums = np.bincount(tags, weights=chunk[valid], minlength=centers.size).reshape(centers.shape)
                updated = np.divide(sums, counts, out=centers.astype(np.float64), where=counts > 0)
                updated = np.sort(updated.astype(np.float32), axis=1)
                if np.array_equal(rounded, updated.astype(np.float16).astype(np.float32)):
                    break
                centers = updated
        book[start:stop] = best_table
        first_byte = start * width // len(shifts)
        octets = (best_codes.reshape(-1, len(shifts)).astype(np.int32) << shifts).sum(axis=1)
        packed[first_byte:first_byte + octets.size] = octets
        decoded = np.take_along_axis(best_table.astype(np.float32), best_codes, axis=1)
        reconstructed[start * width:stop * width] = decoded.ravel()[:segment.size]
    return packed, book, reconstructed.reshape(weight.shape)


def quantize_int8(weight, group_size):
    """Symmetric [-127,127] per block, with the scale stored exactly as float32."""
    weight = np.asarray(weight, dtype=np.float32)
    if not np.isfinite(weight).all():
        raise ValueError("Cannot encode nonfinite int8 weights")
    groups, width = layout(weight.size, group_size, 8)
    blocks = np.pad(weight.ravel(), (0, groups * width - weight.size)).reshape(groups, width)
    maxima = np.max(np.abs(blocks), axis=1, keepdims=True)
    # Round scales to float32 before assigning codes. Very small subnormal blocks
    # can round down to zero or too far down for their largest value to fit.
    scales = (maxima.astype(np.float64) / 127).astype(np.float32)
    scales[maxima == 0] = 1
    smallest = np.nextafter(np.float32(0), np.float32(1))
    scales = np.maximum(scales, smallest)
    round_up = maxima.astype(np.float64) >= 127.5 * scales.astype(np.float64)
    scales[round_up] = np.nextafter(scales[round_up], np.float32(np.inf))
    rounded = np.rint(blocks.astype(np.float64) / scales)
    if np.max(np.abs(rounded)) > 127:
        raise ValueError("Weight outside signed int8 code range")
    codes = rounded.astype(np.int8)
    with np.errstate(over="ignore"):
        decoded = codes.astype(np.float32) * scales
    if not np.isfinite(decoded).all():
        raise ValueError("Int8 reconstruction exceeds float32 range")
    return codes, scales, decoded.ravel()[:weight.size].reshape(weight.shape)


def eligible(model):
    import onnx
    names = set()
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            continue
        if node.op_type in ("Conv", "ConvTranspose", "Gemm"):
            names.add(node.input[1])
        elif node.op_type == "MatMul":
            names.update(node.input)
    names -= {item.name for item in model.graph.input}
    return [tensor for tensor in model.graph.initializer
            if tensor.name in names and tensor.data_type == onnx.TensorProto.FLOAT
            and len(tensor.dims) >= 2 and math.prod(tensor.dims) >= 32]


def make_graph(source, tensors, bits, group_size, arrays=None):
    import onnx
    from onnx import helper, numpy_helper

    model = copy.deepcopy(source)
    graph = model.graph
    eligible_names = {t.name for t in tensors}
    keep = [t for t in graph.initializer if t.name not in eligible_names]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    existing = set(eligible_names)
    existing.update(t.name for t in keep)
    existing.update(x.name for x in list(graph.input) + list(graph.output))
    existing.update(x for node in graph.node for x in list(node.input) + list(node.output))
    prefix = "wpack/"
    while any(x.startswith(prefix) for x in existing):
        prefix = "_" + prefix
    nodes = []

    def init(name, value):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name=prefix + name))
        return prefix + name

    def node(op, inputs, output, **kwargs):
        nodes.append(helper.make_node(op, inputs, [output], **kwargs))
        return output

    lookup = init("lookup", byte_lookup(bits)) if bits in (2, 4) else None
    flat = begin = None
    for i, tensor in enumerate(tensors):
        name = f"{i}/"
        out = prefix + name
        if bits == 16:
            values = np.zeros(tensor.dims, np.float16) if arrays is None else arrays[tensor.name]
            storage = init(name + "fp16", values)
            node("Cast", [storage], tensor.name, to=onnx.TensorProto.FLOAT)
            continue
        n = math.prod(tensor.dims)
        groups, width = layout(n, group_size, bits)
        if bits == 8:
            if arrays is None:
                codes = np.zeros((groups, width), np.int8)
                scales = np.ones((groups, 1), np.float32)
            else:
                codes, scales = arrays[tensor.name]
            storage = init(name + "i8", codes)
            scale = init(name + "scale", scales)
            codes32 = node("Cast", [storage], out + "f32", to=onnx.TensorProto.FLOAT)
            values = node("Mul", [codes32, scale], out + "values")
        else:
            if arrays is None:
                packed = np.zeros(groups * width // (8 // bits), np.uint8)
                table = np.zeros((groups, 1 << bits), np.float16)
            else:
                packed, table = arrays[tensor.name]
            octets = init(name + "bytes", packed)
            centers = init(name + "centers", table)
            shape = init(name + "groups", np.array([groups, width], np.int64))
            addresses = node("Cast", [octets], out + "addresses", to=onnx.TensorProto.INT32)
            unpacked = node("Gather", [lookup, addresses], out + "indices", axis=0)
            indices = node("Reshape", [unpacked, shape], out + "rows")
            table32 = node("Cast", [centers], out + "f32", to=onnx.TensorProto.FLOAT)
            values = node("GatherElements", [table32, indices], out + "values", axis=1)
        original = init(name + "shape", np.array(tensor.dims, np.int64))
        if n != groups * width:
            if flat is None:
                flat = init("flat", np.array([-1], np.int64))
                begin = init("begin", np.array([0], np.int64))
            values = node("Reshape", [values, flat], out + "padded")
            end = init(name + "end", np.array([n], np.int64))
            values = node("Slice", [values, begin, end], out + "trimmed")
        node("Reshape", [values, original], tensor.name)
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + original_nodes)
    return model


def choose_group(source, tensors, bits, requested, cap):
    sizes = [32, 48, 64, 80, 96, 112, 128, 160, 192,
             224, 256, 320, 384, 512, 768, 1024, 2048, 4096]
    if bits == 8:
        sizes = [4, 8, 12, 16, 24] + sizes
    if requested is not None:
        sizes = [requested]
    best = None
    for size in sizes:
        size_bytes = make_graph(source, tensors, bits, size).ByteSize()
        if cap is None or size_bytes <= cap:
            return size
        best = size_bytes if best is None else min(best, size_bytes)
    raise ValueError(f"No group size fits {cap} bytes; smallest measured size was {best}")


def load_states(parity, data, samples, seed):
    cases = json.loads(parity.read_text())["cases"]
    arrays = [np.array([case["encoded"] for case in cases], dtype=np.float32)]
    if data is not None:
        rng = np.random.default_rng(seed)
        files = sorted(data.glob("*.npz")) if data.is_dir() else [data]
        if not files:
            raise ValueError("No NPZ files in data directory")
        # Uniformly sample rows over shards, even when shards vary in size.
        lengths = []
        for file in files:
            with np.load(file, allow_pickle=False) as shard:
                lengths.append(len(shard["v"]))
        starts = np.cumsum([0] + lengths)
        chosen = np.sort(rng.choice(starts[-1], min(samples, starts[-1]), replace=False))
        for i, file in enumerate(files):
            local = chosen[(chosen >= starts[i]) & (chosen < starts[i + 1])] - starts[i]
            if local.size:
                with np.load(file, allow_pickle=False) as shard:
                    arrays.append(shard["x"][local].astype(np.float32))
    states = np.concatenate(arrays)
    if states.ndim != 2 or states.shape[1] != 280 or not len(cases):
        raise ValueError("Expected project state features [batch,280] and nonempty parity suite")
    return states


def evaluate_models(source, candidate, expected, states, batch_size):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def infer(model):
        session = ort.InferenceSession(model.SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        # Exercise both singleton and multi-position batches.
        batches = [states[:1]] + [states[i:i + batch_size] for i in range(1, len(states), batch_size)]
        rows = [session.run(["policy", "value"], {name: x}) for x in batches]
        result = [np.concatenate(parts) for parts in zip(*rows)]
        p, v = result
        if p.shape != (len(states), 81) or v.size != len(states) or not all(np.isfinite(a).all() for a in result):
            raise ValueError("Invalid native policy or value outputs")
        return p, v.reshape(-1)

    reference_p, reference_v = infer(source)
    packed_p, packed_v = infer(candidate)
    dense_p, dense_v = infer(expected)
    np.testing.assert_allclose(packed_p, dense_p, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(packed_v, dense_v, rtol=1e-4, atol=1e-4)
    legal = states[:, 162:243].astype(bool)
    live = legal.any(axis=1)
    if not live.any():
        raise ValueError("Drift evaluation requires live positions")
    p = np.where(legal[live], reference_p[live], -np.inf)
    q = np.where(legal[live], packed_p[live], -np.inf)

    def log_probs(a):
        shifted = a.astype(np.float64) - a.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    a, b = log_probs(p), log_probs(q)
    valid = legal[live]
    kl = (np.exp(a[valid]) * (a[valid] - b[valid])).sum() / len(a)
    return {"positions": len(states), "live_positions": int(live.sum()),
            "legal_top1_agreement": float(np.mean(p.argmax(1) == q.argmax(1))),
            "legal_policy_kl": float(kl), "policy_logit_rmse": float(np.mean((reference_p - packed_p)**2)**0.5),
            "value_mae": float(np.mean(np.abs(reference_v - packed_v))),
            "value_max_error": float(np.max(np.abs(reference_v - packed_v))),
            "graph_dense_max_logit_error": float(np.max(np.abs(dense_p - packed_p))),
            "graph_dense_max_value_error": float(np.max(np.abs(dense_v - packed_v)))}


def verify_int8_reconstruction(source, candidate, tensors, expected):
    """Run the serialized Cast/Mul/reshape decoder and compare every float32 kernel."""
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort

    decoder = candidate.graph.node[:len(candidate.graph.node) - len(source.graph.node)]
    used = {name for node in decoder for name in node.input}
    stored = [t for t in candidate.graph.initializer if t.name in used]
    outputs = [helper.make_tensor_value_info(t.name, onnx.TensorProto.FLOAT, list(t.dims)) for t in tensors]
    graph = helper.make_graph(decoder, "int8_kernel_check", [], outputs, initializer=stored)
    model = helper.make_model(graph, opset_imports=list(candidate.opset_import), ir_version=candidate.ir_version)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    for tensor, decoded in zip(tensors, session.run(None, {}), strict=True):
        np.testing.assert_array_equal(decoded, numpy_helper.to_array(expected[tensor.name]), err_msg=tensor.name)


def pack(args):
    import onnx
    from onnx import numpy_helper

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ")
    if args.report and args.report.resolve() in (args.input.resolve(), args.output.resolve()):
        raise ValueError("Report must be a separate file")
    if args.iterations < 0 or args.samples < 1 or args.batch_size < 1:
        raise ValueError("Invalid iterations, samples, or batch size")
    source = onnx.load(str(args.input))
    onnx.external_data_helper.convert_model_from_external_data(source)
    if not any(op.domain in ("", "ai.onnx") and op.version >= 11 for op in source.opset_import):
        raise ValueError("Standard ONNX opset 11 or higher is required")
    tensors = eligible(source)
    if not tensors:
        raise ValueError("No eligible frozen float32 weights")
    if args.bits == 16 and args.group_size is not None:
        raise ValueError("Float16 weights have no groups")
    size = None if args.bits == 16 else choose_group(source, tensors, args.bits, args.group_size, args.max_bytes)
    expected = copy.deepcopy(source)
    slots = {t.name: t for t in expected.graph.initializer}
    arrays = {}
    squared_error = squared_weight = 0.0
    count = 0
    for tensor in tensors:
        weight = numpy_helper.to_array(tensor)
        if args.bits == 16:
            check_float16_range(weight)
            half = weight.astype(np.float16)
            arrays[tensor.name] = half
            decoded = half.astype(np.float32)
        elif args.bits == 8:
            codes, scales, decoded = quantize_int8(weight, size)
            arrays[tensor.name] = (codes, scales)
        else:
            octets, table, decoded = quantize(weight, args.bits, size, args.iterations)
            arrays[tensor.name] = (octets, table)
        slots[tensor.name].CopyFrom(numpy_helper.from_array(decoded, name=tensor.name))
        squared_error += np.square(weight.astype(np.float64) - decoded).sum()
        squared_weight += np.square(weight, dtype=np.float64).sum()
        count += weight.size
    candidate = make_graph(source, tensors, args.bits, size, arrays)
    onnx.checker.check_model(candidate)
    payload = candidate.SerializeToString()
    if args.max_bytes is not None and len(payload) > args.max_bytes:
        raise ValueError(f"Candidate exceeds cap: {len(payload)} bytes")
    if args.bits == 8:
        verify_int8_reconstruction(source, candidate, tensors, slots)
    states = load_states(args.parity, args.data, args.samples, args.seed)
    drift = evaluate_models(source, candidate, expected, states, args.batch_size)
    report = {"bits": args.bits, "group_size": size, "iterations": args.iterations if args.bits in (2, 4) else 0, "seed": args.seed,
              "packed_tensors": len(tensors), "packed_weights": count,
              "input_bytes": args.input.stat().st_size, "packed_bytes": len(payload),
              "weight_rmse": math.sqrt(squared_error / count),
              "relative_weight_rmse": math.sqrt(squared_error / squared_weight) if squared_weight else 0,
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "packed_sha256": hashlib.sha256(payload).hexdigest(), "drift": drift}
    if args.bits == 8:
        report["exactly_reconstructed_tensors"] = len(tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as fp:
        temporary = Path(fp.name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--group-size", type=int, help="fixed block length (default: smallest fitting the cap)")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--parity", type=Path, default=Path(__file__).with_name("parity.json"))
    parser.add_argument("--data", type=Path, help="optional compact NPZ shard or directory of shards")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7241)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path)
    print(json.dumps(pack(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
