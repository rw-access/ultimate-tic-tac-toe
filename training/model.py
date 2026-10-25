"""The compact PyTorch policy/value network and ONNX exporter."""

import tempfile
from pathlib import Path

import torch
from torch import nn

MAX_ONNX_MODEL_BYTES = 1_500_000
ONNX_SIZE_ESTIMATE_OVERHEAD_BYTES = 16 * 1024


def estimate_onnx_model_bytes(hidden_widths: tuple[int, int]) -> int:
    """Conservatively estimate the float32 ONNX artifact before training."""
    if (len(hidden_widths) != 2
            or any(not isinstance(width, int) or width < 1 for width in hidden_widths)):
        raise ValueError("hidden_widths must contain two positive integers")
    first_width, second_width = hidden_widths
    parameters = (
        (280 + 1) * first_width
        + (first_width + 1) * second_width
        + (second_width + 1) * 81
        + (second_width + 1)
    )
    return parameters * 4 + ONNX_SIZE_ESTIMATE_OVERHEAD_BYTES


def validate_onnx_model_size(hidden_widths: tuple[int, int],
                             actual_bytes: int | None = None) -> int:
    """Enforce the preflight estimate and exact exported ONNX byte limit."""
    estimated_bytes = estimate_onnx_model_bytes(hidden_widths)
    if estimated_bytes > MAX_ONNX_MODEL_BYTES:
        raise ValueError(
            f"estimated ONNX model size {estimated_bytes:,} bytes exceeds the "
            f"{MAX_ONNX_MODEL_BYTES:,}-byte maximum"
        )
    if actual_bytes is not None and actual_bytes > MAX_ONNX_MODEL_BYTES:
        raise ValueError(
            f"exported ONNX model size {actual_bytes:,} bytes exceeds the "
            f"{MAX_ONNX_MODEL_BYTES:,}-byte maximum"
        )
    return estimated_bytes


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_widths: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        if (len(hidden_widths) != 2
                or any(not isinstance(width, int) or width < 1 for width in hidden_widths)):
            raise ValueError("hidden_widths must contain two positive integers")
        self.hidden_widths = tuple(hidden_widths)
        first_width, second_width = self.hidden_widths
        self.trunk = nn.Sequential(
            nn.Linear(280, first_width), nn.ReLU(),
            nn.Linear(first_width, second_width), nn.ReLU(),
        )
        self.policy_head = nn.Linear(second_width, 81)
        self.value_head = nn.Linear(second_width, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        return self.policy_head(hidden), torch.tanh(self.value_head(hidden))


def export_onnx(model: PolicyValueNet, output: str | Path) -> None:
    validate_onnx_model_size(model.hidden_widths)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".onnx", dir=output.parent, delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        torch.onnx.export(
            model, torch.zeros((2, 280), dtype=torch.float32), str(temporary),
            input_names=["state"], output_names=["policy", "value"],
            dynamic_axes={"state": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        validate_onnx_model_size(model.hidden_widths, temporary.stat().st_size)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
