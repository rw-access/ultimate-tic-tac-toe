"""Adapt the dynamic uttt.ai ONNX teacher to the project's fixed-work arena."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def wrap(source: Path) -> onnx.ModelProto:
    model = onnx.load(source)
    graph = model.graph
    initializers = {tensor.name for tensor in graph.initializer}
    inputs = [field for field in graph.input if field.name not in initializers]
    if [field.name for field in inputs] != ["input"] or {field.name for field in graph.output} != {
        "policy_logits", "state_value"
    }:
        raise ValueError("Expected uttt.ai input and policy_logits/state_value outputs")
    shape = inputs[0].type.tensor_type.shape.dim
    if len(shape) != 4 or shape[0].HasField("dim_value") or [d.dim_value for d in shape[1:]] != [4, 9, 9]:
        raise ValueError("The source model needs dynamic batch size and four 9x9 input planes")

    model = onnx.compose.add_prefix(model, "utttai/")
    graph = model.graph
    # Transposing board-column and cell-row maps board-major to spatial order.
    # The transpose is its own inverse, so it also maps logits back to move IDs.
    order = np.arange(81).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(-1)
    turn_weights = np.zeros((280, 1), dtype=np.float32)
    turn_weights[:81] = 2
    turn_weights[81:162] = -2
    constants = {
        "own_indices": order, "opponent_indices": order + 81, "legal_indices": order + 162,
        "turn_weights": turn_weights, "bias": np.asarray(1, dtype=np.float32),
        "tile": [1, 81], "image_shape": [-1, 4, 9, 9], "value_shape": [-1, 1],
    }
    for name, array in constants.items():
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))

    node = helper.make_node
    pre = [
        node("Gather", ["state", "own_indices"], ["own"], axis=1),
        node("Gather", ["state", "opponent_indices"], ["opponent"], axis=1),
        node("Gather", ["state", "legal_indices"], ["legal"], axis=1),
        # Counts are equal on X's turn; on O's turn the opponent has one extra mark.
        node("MatMul", ["state", "turn_weights"], ["count_difference"]),
        node("Add", ["count_difference", "bias"], ["turn"]),
        node("Tile", ["turn", "tile"], ["turn_plane"]),
        node("Concat", ["own", "opponent", "turn_plane", "legal"], ["planes"], axis=1),
        node("Reshape", ["planes", "image_shape"], ["utttai/input"]),
    ]
    post = [
        node("Flatten", ["utttai/policy_logits"], ["spatial_logits"], axis=1),
        node("Gather", ["spatial_logits", "own_indices"], ["policy"], axis=1),
        node("Reshape", ["utttai/state_value", "value_shape"], ["value"]),
    ]
    nodes = pre + list(graph.node) + post
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.input[:]
    del graph.output[:]
    for fields, name, width in [(graph.input, "state", 280), (graph.output, "policy", 81), (graph.output, "value", 1)]:
        fields.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width]))
    onnx.checker.check_model(model)
    return model


def reference_planes(state):
    """Independently encode the JSON board in spatial order like utttpy/game/helpers.py."""
    planes = np.zeros((4, 9, 9), dtype=np.float32)
    planes[2] = 1 if state["turn"] == "X" else -1
    for row in range(9):
        for col in range(9):
            board = 3 * (row // 3) + col // 3
            cell = 3 * (row % 3) + col % 3
            mark = state["boards"][board][cell]
            if mark is not None:
                planes[int(mark != state["turn"]), row, col] = 1
            if (state["winner"] is None and state["results"][board] is None
                    and state["forced"] in (board, None) and mark is None):
                planes[3, row, col] = 1
    return planes


def verify(source: Path, candidate: Path, fixtures: Path):
    cases = json.loads(fixtures.read_text())["cases"]
    if {case["state"]["turn"] for case in cases} != {"X", "O"}:
        raise ValueError("Parity fixtures must exercise both seats")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                for path in (source, candidate)]
    states = np.array([case["encoded"] for case in cases], dtype=np.float32)
    planes = np.stack([reference_planes(case["state"]) for case in cases])
    for size in (1, len(cases)):
        spatial, reference_value = sessions[0].run(["policy_logits", "state_value"], {"input": planes[:size]})
        actual, value = sessions[1].run(["policy", "value"], {"state": states[:size]})
        if spatial.shape != (size, 9, 9) or reference_value.shape != (size,):
            raise ValueError("Unexpected shape from the raw teacher")
        if actual.shape != (size, 81) or value.shape != (size, 1):
            raise ValueError("Unexpected shape from the wrapped teacher")
        for output in (spatial, reference_value, actual, value):
            if not np.isfinite(output).all():
                raise ValueError("Nonfinite output from teacher parity run")
        np.testing.assert_allclose(value[:, 0], reference_value, rtol=1e-5, atol=1e-5)
        for index, case in enumerate(cases[:size]):
            expected = []
            for move in range(81):
                row = 3 * (move // 27) + (move % 9) // 3
                col = 3 * ((move // 9) % 3) + move % 3
                expected.append(spatial[index, row, col])
            np.testing.assert_allclose(actual[index], expected, rtol=1e-5, atol=2e-5)
            legal = planes[index, 3].astype(bool)
            if legal.any():
                row, col = np.unravel_index(np.where(legal, spatial[index], -np.inf).argmax(), (9, 9))
                reference_move = (row // 3 * 3 + col // 3) * 9 + (row % 3 * 3 + col % 3)
                best = int(np.where(states[index, 162:243] > 0, actual[index], -np.inf).argmax())
                # Equal logits may resolve differently in the two move orders.
                if expected[best] != expected[reference_move] or not states[index, 162 + reference_move]:
                    raise ValueError(f"Best legal move mismatch for {case['name']}")
    print(f"CPU parity passed on {len(cases)} fixture positions, batch sizes 1 and {len(cases)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity", type=Path, default=Path("training/parity.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Use a separate path for the adapted model")
    model = wrap(args.input)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        candidate = Path(directory) / "teacher.onnx"
        onnx.save(model, candidate)
        verify(args.input, candidate, args.parity)
        candidate.replace(args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
