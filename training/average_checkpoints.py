"""Uniformly average policy/value checkpoints descended from a common initialization.

This implements the uniform model soup of Wortsman et al. (ICML 2022). The
training records identify a common initialization; a state dictionary alone
does not encode that lineage. Only learned floating-point parameters are averaged.
"""

import argparse
import json
from pathlib import Path

import torch

from .fit_search_student import export, file_hash, load_weights
from .line_student import LineStudent
from .stage2_student import Stage2Net


@torch.no_grad()
def average(model, paths):
    if len(paths) < 2 or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Provide at least two distinct checkpoint paths")
    parameters = dict(model.named_parameters())
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    totals = {name: torch.zeros_like(p, dtype=torch.float64) for name, p in parameters.items()}

    # load_state_dict casts incoming tensors to the destination dtype. Inspect
    # the source values before that cast can conceal a changed integer buffer.
    def check_source(module, state, *unused):
        for name, expected in module.state_dict().items():
            tensor = state[name]
            if tensor.dtype != expected.dtype:
                raise ValueError(f"{path}: unexpected dtype for {name}: {tensor.dtype}")
            if name in buffers and not torch.equal(tensor, buffers[name]):
                raise ValueError(f"{path}: structural buffer {name} differs from the canonical tier")

    hook = model.register_load_state_dict_pre_hook(check_source)
    try:
        for path in paths:
            load_weights(model, path)
            for name, parameter in parameters.items():
                if not parameter.is_floating_point() or not torch.isfinite(parameter).all():
                    raise ValueError(f"{path}: expected finite floating-point parameter {name}")
                totals[name].add_(parameter)
    finally:
        hook.remove()
    for name, parameter in parameters.items():
        parameter.copy_(totals[name].div_(len(paths)))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("stage3_teacher", "line_student"), required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh .onnx path; also writes an averaged .pt")
    args = parser.parse_args()
    if args.output.suffix != ".onnx" or args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Choose a fresh .onnx path and sibling .pt")
    model = Stage2Net(compact=False) if args.tier == "stage3_teacher" else LineStudent()
    average(model, args.inputs)
    cases = json.loads(Path(__file__).with_name("parity.json").read_text())["cases"]
    positions = torch.tensor([case["encoded"] for case in cases], dtype=torch.float32)
    print(json.dumps({"tier": args.tier, "method": "uniform_parameter_mean",
                      "inputs": [{"path": str(p), "sha256": file_hash(p)} for p in args.inputs]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, positions, args.output, args.tier)


if __name__ == "__main__":
    main()
