"""Train and export a policy/value model from TypeScript-generated examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import (
    MAX_ONNX_MODEL_BYTES,
    PolicyValueNet,
    export_onnx,
    validate_onnx_model_size,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIDDEN_WIDTHS = (256, 128)
WIDENING_NOISE_SCALE = 1e-4


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads not in (1, 2):
        raise ValueError("training threads must be 1 or 2")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def _count_positions(path: Path) -> list[int]:
    counts = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                counts.append(len(json.loads(line)["samples"]))
    return counts


def _game_content_sha256(record: dict, game_id: int) -> str:
    """Hash a game's ordered states and outcomes, excluding IDs and policy labels."""
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"game {game_id}: samples must be a list")
    digest = hashlib.sha256()
    digest.update(b"ultimate-tic-tac-toe-game-content-v1\0")
    digest.update(len(samples).to_bytes(8, "little"))
    for sample in samples:
        try:
            features, _policy, value = sample
            features = np.asarray(features, dtype="<f4")
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"game {game_id}: malformed sample for overlap check") from error
        if features.shape != (280,) or not np.isfinite(features).all() or not math.isfinite(value):
            raise ValueError(f"game {game_id}: invalid state/value for overlap check")
        digest.update(features.tobytes(order="C"))
        digest.update(np.asarray([value], dtype="<f4").tobytes())
    return digest.hexdigest()


def _scan_dataset(path: Path, identify_games: bool = True) -> tuple[list[int], str, list[str]]:
    """Read counts/SHA-256 and optionally index-independent game identities."""
    counts: list[int] = []
    fingerprints: list[str] = []
    file_digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            file_digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            samples = record.get("samples")
            if not isinstance(samples, list):
                raise ValueError(f"game {len(counts)}: samples must be a list")
            counts.append(len(samples))
            if identify_games:
                fingerprints.append(_game_content_sha256(record, len(counts) - 1))
    return counts, file_digest.hexdigest(), fingerprints


def _require_disjoint_games(training_fingerprints: list[str],
                            validation_fingerprints: list[str]) -> None:
    overlap = set(training_fingerprints) & set(validation_fingerprints)
    if overlap:
        raise ValueError(
            f"external validation overlaps training data in {len(overlap)} exact game(s)"
        )


def _split_games(counts: list[int], seed: int, fraction: float) -> tuple[set[int], set[int]]:
    order = list(range(len(counts)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, int(round(len(order) * fraction))) if len(order) > 1 else 0
    validation = set(order[:validation_count])
    return set(order[validation_count:]), validation


def _transform_coord(index: int, symmetry: int) -> int:
    row, column = divmod(index, 3)
    if symmetry >= 4:
        column = 2 - column
    for _ in range(symmetry % 4):
        row, column = column, 2 - row
    return row * 3 + column


def _symmetry_maps() -> list[tuple[np.ndarray, np.ndarray]]:
    """Build source-to-destination permutations for input features and moves."""
    result = []
    for symmetry in range(8):
        feature_map = np.arange(280)
        policy_map = np.arange(81)
        for action in range(81):
            board, cell = divmod(action, 9)
            destination = _transform_coord(board, symmetry) * 9 + _transform_coord(cell, symmetry)
            policy_map[action] = destination
            for channel in (0, 81, 162):
                feature_map[channel + action] = channel + destination
        for board in range(9):
            destination = _transform_coord(board, symmetry)
            for channel in (243, 252, 261, 270):
                feature_map[channel + board] = channel + destination
        result.append((feature_map, policy_map))
    return result


def _augment_batch(features: np.ndarray, policies: np.ndarray,
                   symmetries: np.ndarray,
                   maps: tuple[np.ndarray, np.ndarray] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Apply one independently selected symmetry to each item in a batch."""
    if maps is None:
        raw_maps = _symmetry_maps()
        maps = (np.stack([item[0] for item in raw_maps]),
                np.stack([item[1] for item in raw_maps]))
    feature_maps, policy_maps = maps
    feature_destinations = feature_maps[symmetries]
    policy_destinations = policy_maps[symmetries]
    rows = np.arange(len(features))[:, None]
    augmented_features = np.empty_like(features)
    augmented_policies = np.empty_like(policies)
    augmented_features[rows, feature_destinations] = features
    augmented_policies[rows, policy_destinations] = policies
    return augmented_features, augmented_policies


def parse_hidden_widths(value: str) -> tuple[int, int]:
    try:
        parts = value.split(",")
        if len(parts) != 2 or any(not part.strip() for part in parts):
            raise ValueError
        widths = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden widths must be two positive integers, e.g. 512,256") from error
    if any(width < 1 for width in widths):
        raise argparse.ArgumentTypeError("hidden widths must be two positive integers, e.g. 512,256")
    return widths


def _load_onnx_state(model_path: Path) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    import onnx
    from onnx.numpy_helper import to_array

    graph = onnx.load(str(model_path))
    weights = {item.name: torch.from_numpy(to_array(item).copy())
               for item in graph.graph.initializer}
    try:
        first_width, input_width = weights["trunk.0.weight"].shape
        second_width, first_width_from_second = weights["trunk.2.weight"].shape
        policy_width, second_width_from_policy = weights["policy_head.weight"].shape
        value_width, second_width_from_value = weights["value_head.weight"].shape
    except (KeyError, ValueError) as error:
        raise ValueError("initial ONNX does not contain the expected policy/value MLP weights") from error
    if (input_width != 280 or first_width_from_second != first_width
            or second_width_from_policy != second_width
            or second_width_from_value != second_width
            or policy_width != 81 or value_width != 1):
        raise ValueError("initial ONNX has incompatible input, hidden, or output dimensions")
    return weights, (int(first_width), int(second_width))


def _onnx_hidden_widths(model_path: Path) -> tuple[int, int]:
    return _load_onnx_state(model_path)[1]


def _copy_map(source_width: int, target_width: int,
              rng: np.random.Generator) -> np.ndarray:
    mapping = np.empty(target_width, dtype=np.int64)
    mapping[:source_width] = np.arange(source_width)
    if target_width > source_width:
        mapping[source_width:] = rng.integers(source_width, size=target_width - source_width)
    return mapping


def _net2wider_state(source: dict[str, torch.Tensor],
                     target_widths: tuple[int, int], seed: int,
                     noise_scale: float = WIDENING_NOISE_SCALE
                     ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Expand both hidden layers while preserving the represented function."""
    old_first = int(source["trunk.0.weight"].shape[0])
    old_second = int(source["trunk.2.weight"].shape[0])
    new_first, new_second = target_widths
    if new_first < old_first or new_second < old_second:
        raise ValueError(
            f"hidden-width shrinking is unsupported (parent {(old_first, old_second)}, "
            f"requested {target_widths})"
        )
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("widening noise scale must be finite and nonnegative")
    if target_widths == (old_first, old_second):
        return {key: value.detach().clone() for key, value in source.items()}, {
            "first_map": np.arange(old_first), "second_map": np.arange(old_second),
        }

    rng = np.random.default_rng(seed)
    first_map = _copy_map(old_first, new_first, rng)
    second_map = _copy_map(old_second, new_second, rng)
    first_counts = np.bincount(first_map, minlength=old_first)
    second_counts = np.bincount(second_map, minlength=old_second)

    # Distribute each parent's outgoing contribution across its copies. Small
    # per-source zero-sum perturbations break symmetry without changing their sum.
    second_source = source["trunk.2.weight"].detach().cpu().numpy().astype(np.float64)
    second_expanded = second_source[:, first_map] / first_counts[first_map][None, :]
    if noise_scale and new_first > old_first:
        for source_index in range(old_first):
            copies = np.flatnonzero(first_map == source_index)
            if len(copies) > 1:
                noise = rng.normal(0.0, noise_scale, size=(old_second, len(copies)))
                noise -= noise.mean(axis=1, keepdims=True)
                second_expanded[:, copies] += noise

    policy_source = source["policy_head.weight"].detach().cpu().numpy().astype(np.float64)
    value_source = source["value_head.weight"].detach().cpu().numpy().astype(np.float64)
    policy_expanded = policy_source[:, second_map] / second_counts[second_map][None, :]
    value_expanded = value_source[:, second_map] / second_counts[second_map][None, :]
    if noise_scale and new_second > old_second:
        for source_index in range(old_second):
            copies = np.flatnonzero(second_map == source_index)
            if len(copies) > 1:
                policy_noise = rng.normal(0.0, noise_scale, size=(81, len(copies)))
                policy_noise -= policy_noise.mean(axis=1, keepdims=True)
                value_noise = rng.normal(0.0, noise_scale, size=(1, len(copies)))
                value_noise -= value_noise.mean(axis=1, keepdims=True)
                policy_expanded[:, copies] += policy_noise
                value_expanded[:, copies] += value_noise

    def float_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())

    widened = {
        "trunk.0.weight": source["trunk.0.weight"][torch.as_tensor(first_map)].detach().clone(),
        "trunk.0.bias": source["trunk.0.bias"][torch.as_tensor(first_map)].detach().clone(),
        "trunk.2.weight": float_tensor(second_expanded[second_map]),
        "trunk.2.bias": source["trunk.2.bias"][torch.as_tensor(second_map)].detach().clone(),
        "policy_head.weight": float_tensor(policy_expanded),
        "policy_head.bias": source["policy_head.bias"].detach().clone(),
        "value_head.weight": float_tensor(value_expanded),
        "value_head.bias": source["value_head.bias"].detach().clone(),
    }
    return widened, {"first_map": first_map, "second_map": second_map}


def _load_initial_model_with_info(model: PolicyValueNet, model_path: Path, seed: int,
                                  noise_scale: float = WIDENING_NOISE_SCALE
                                  ) -> tuple[str, dict[str, object]]:
    weights, parent_widths = _load_onnx_state(model_path)
    expected = model.state_dict()
    missing = set(expected) - set(weights)
    target_widths = model.hidden_widths
    if any(target < parent for target, parent in zip(target_widths, parent_widths)):
        raise ValueError(
            f"hidden-width shrinking is unsupported (parent {parent_widths}, requested {target_widths})"
        )
    if target_widths == parent_widths:
        if missing:
            raise ValueError(f"initial ONNX is missing model weights: {sorted(missing)}")
        model.load_state_dict({key: weights[key] for key in expected}, strict=True)
        method = "exact_load"
        widening_details = {"first_map": np.arange(parent_widths[0]),
                            "second_map": np.arange(parent_widths[1])}
    else:
        widened, widening_details = _net2wider_state(weights, target_widths, seed, noise_scale)
        model.load_state_dict(widened, strict=True)
        method = "net2wider_zero_sum_perturbation"
    info: dict[str, object] = {
        "method": method,
        "parent_hidden_widths": list(parent_widths),
        "hidden_widths": list(target_widths),
        "seed": seed if method != "exact_load" else None,
        "noise_scale": noise_scale if method != "exact_load" else 0.0,
        "preserves_function": True,
        # Kept private to callers/tests; not copied into published run metadata.
        "_first_map": widening_details["first_map"],
        "_second_map": widening_details["second_map"],
    }
    return hashlib.sha256(model_path.read_bytes()).hexdigest(), info


def _load_initial_model(model: PolicyValueNet, model_path: Path) -> str:
    """Backwards-compatible exact/widening loader used by lightweight callers."""
    digest, _info = _load_initial_model_with_info(model, model_path, seed=0)
    return digest


def _read_sample(game_id: int, sample) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        features, policy, value = sample
    except (TypeError, ValueError) as error:
        raise ValueError(f"game {game_id}: sample must be [features, policy, value]") from error
    features = np.asarray(features, dtype=np.float32)
    policy = np.asarray(policy, dtype=np.float32)
    value = float(value)
    if features.shape != (280,) or policy.shape != (81,):
        raise ValueError(f"game {game_id}: expected feature[280], policy[81]")
    if not np.isfinite(features).all() or not np.isfinite(policy).all() or not math.isfinite(value):
        raise ValueError(f"game {game_id}: sample contains non-finite values")
    if np.any(features < 0) or np.any(features > 1) or np.any(policy < -1e-7):
        raise ValueError(f"game {game_id}: features/policy must be nonnegative and bounded")
    if value < -1 or value > 1:
        raise ValueError(f"game {game_id}: value target must be in [-1, 1]")
    legal = features[162:243] > 0.5
    if not legal.any():
        raise ValueError(f"game {game_id}: sampled position has no legal moves")
    if float(policy[~legal].sum()) > 1e-6:
        raise ValueError(f"game {game_id}: policy target assigns mass to an illegal move")
    mass = float(policy.sum())
    if mass <= 0 or not math.isfinite(mass):
        raise ValueError(f"game {game_id}: policy target has no mass")
    policy = np.maximum(policy, 0)
    policy /= policy.sum()
    return features, policy, value


def _load_arrays(path: Path, counts: list[int], train_ids: set[int],
                 validation_ids: set[int], augment: bool,
                 validation_path: Path | None = None,
                 validation_counts: list[int] | None = None,
                 expected_training_sha256: str | None = None,
                 expected_validation_sha256: str | None = None):
    train_positions = sum(counts[index] for index in train_ids)
    if validation_path is None:
        validation_counts = counts
    elif validation_counts is None:
        raise ValueError("validation_counts are required with a separate validation_path")
    validation_positions = sum(validation_counts[index] for index in validation_ids)
    multiplier = 8 if augment else 1
    train = (
        np.empty((train_positions * multiplier, 280), dtype=np.float32),
        np.empty((train_positions * multiplier, 81), dtype=np.float32),
        np.empty(train_positions * multiplier, dtype=np.float32),
    )
    validation = (
        np.empty((validation_positions, 280), dtype=np.float32),
        np.empty((validation_positions, 81), dtype=np.float32),
        np.empty(validation_positions, dtype=np.float32),
    )
    maps = _symmetry_maps()
    train_row = validation_row = game_id = 0

    def load_training(record: dict, source_game_id: int) -> None:
        nonlocal train_row
        active_maps = maps if augment else maps[:1]
        for sample in record["samples"]:
            source_x, source_p, value = _read_sample(source_game_id, sample)
            for feature_map, policy_map in active_maps:
                train[0][train_row, feature_map] = source_x
                train[1][train_row, policy_map] = source_p
                train[2][train_row] = float(value)
                train_row += 1

    def load_validation(record: dict, source_game_id: int) -> None:
        nonlocal validation_row
        for sample in record["samples"]:
            source_x, source_p, value = _read_sample(source_game_id, sample)
            validation[0][validation_row] = source_x
            validation[1][validation_row] = source_p
            validation[2][validation_row] = float(value)
            validation_row += 1

    training_digest = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            training_digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            if game_id in train_ids:
                load_training(record, game_id)
            if validation_path is None and game_id in validation_ids:
                load_validation(record, game_id)
            game_id += 1
    if expected_training_sha256 and training_digest.hexdigest() != expected_training_sha256:
        raise ValueError("training dataset changed after its hash and overlap scan")
    if validation_path is None and expected_validation_sha256:
        if training_digest.hexdigest() != expected_validation_sha256:
            raise ValueError("internal validation dataset hash does not match training dataset")
    if validation_path is not None:
        game_id = 0
        validation_digest = hashlib.sha256()
        with validation_path.open("rb") as source:
            for line in source:
                validation_digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                if game_id in validation_ids:
                    load_validation(record, game_id)
                game_id += 1
        if (expected_validation_sha256
                and validation_digest.hexdigest() != expected_validation_sha256):
            raise ValueError("validation dataset changed after its hash and overlap scan")
    if train_row != len(train[2]) or validation_row != len(validation[2]):
        raise RuntimeError("dataset changed while loading game-level split")
    if not len(train[2]):
        raise ValueError("training split contains no positions")
    return train, validation


def _validation_diagnostics(policies: np.ndarray, values: np.ndarray,
                            training_values: np.ndarray,
                            policy_cross_entropy: float) -> dict[str, float | None]:
    """Summarize target entropy/KL and simple value baselines on one holdout."""
    if not len(values):
        return {
            "policy_target_entropy": None,
            "policy_cross_entropy": float(policy_cross_entropy),
            "policy_kl": None,
            "value_target_mean": None,
            "value_target_variance": None,
            "value_zero_mse": None,
            "value_empirical_mean_mse": None,
            "value_train_mean_mse": None,
        }
    # Diagnostics are small compared with network training, and float64
    # aggregation avoids rounding a constant-baseline MSE at float32 precision.
    policies_64 = np.asarray(policies, dtype=np.float64)
    values_64 = np.asarray(values, dtype=np.float64)
    training_values_64 = np.asarray(training_values, dtype=np.float64)
    log_policy = np.zeros_like(policies_64)
    np.log(policies_64, out=log_policy, where=policies_64 > 0)
    target_entropy = float(-(policies_64 * log_policy).sum(axis=1).mean())
    target_mean = float(values_64.mean())
    training_mean = float(training_values_64.mean())
    return {
        "policy_target_entropy": target_entropy,
        "policy_cross_entropy": float(policy_cross_entropy),
        "policy_kl": float(policy_cross_entropy - target_entropy),
        "value_target_mean": target_mean,
        "value_target_variance": float(values_64.var()),
        "value_zero_mse": float(np.mean(np.square(values_64))),
        # Descriptive lower bound; its mean is estimated from validation labels.
        "value_empirical_mean_mse": float(np.mean(np.square(values_64 - target_mean))),
        # Fair constant-predictor baseline: estimate the mean on training labels.
        "value_train_mean_mse": float(np.mean(np.square(values_64 - training_mean))),
    }


def _losses(model: PolicyValueNet, features: torch.Tensor, policies: torch.Tensor,
            values: torch.Tensor, batch_size: int) -> tuple[float, float]:
    if not len(features):
        return 0.0, 0.0
    model.eval()
    policy_sum = value_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = features[start:start + batch_size]
            target_p = policies[start:start + batch_size]
            target_v = values[start:start + batch_size]
            logits, prediction = model(x)
            logits = logits.masked_fill(x[:, 162:243] < 0.5, -1.0e9)
            policy_loss = -(target_p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(prediction[:, 0], target_v)
            policy_sum += float(policy_loss) * len(x)
            value_sum += float(value_loss) * len(x)
            count += len(x)
    return policy_sum / count, value_sum / count


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def compare_torch_onnx(model_path: Path, sample_x: np.ndarray) -> dict[str, float]:
    """Rebuild the exported weights and check the trained graph numerically."""
    import onnx
    import onnxruntime as ort
    from onnx.numpy_helper import to_array

    graph = onnx.load(str(model_path))
    weights = {initializer.name: torch.from_numpy(to_array(initializer).copy())
               for initializer in graph.graph.initializer}
    first_weight = weights.get("trunk.0.weight")
    second_weight = weights.get("trunk.2.weight")
    if first_weight is None or second_weight is None:
        raise ValueError("ONNX is missing hidden-layer weights")
    reference = PolicyValueNet((int(first_weight.shape[0]), int(second_weight.shape[0])))
    reference.load_state_dict(weights)
    reference.eval()
    with torch.no_grad():
        torch_policy, torch_value = reference(torch.from_numpy(sample_x))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    ort_policy, ort_value = session.run(["policy", "value"], {"state": sample_x})
    policy_error = float(np.max(np.abs(ort_policy - torch_policy.numpy())))
    value_error = float(np.max(np.abs(ort_value - torch_value.numpy())))
    if policy_error >= 1e-4 or value_error >= 1e-4:
        raise RuntimeError(f"Torch/ONNX mismatch: policy={policy_error}, value={value_error}")
    return {"policy_max_abs_error": policy_error, "value_max_abs_error": value_error,
            "tolerance": 1e-4}


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.validation_fraction)
            or args.validation_fraction < 0 or args.validation_fraction >= 1):
        raise SystemExit("learning rate must be positive and validation fraction in [0, 1)")
    if (not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.value_weight) or args.value_weight < 0):
        raise SystemExit("weight decay and value weight must be finite and nonnegative")
    if args.threads not in (1, 2):
        raise SystemExit("--threads must be 1 or 2")
    if args.patience < 1:
        raise SystemExit("--patience must be at least 1")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise SystemExit("--max-seconds must be positive")
    if args.max_updates is not None and args.max_updates < 1:
        raise SystemExit("--max-updates must be positive")
    hidden_widths = tuple(args.hidden_widths)
    if len(hidden_widths) != 2 or any(width < 1 for width in hidden_widths):
        raise SystemExit("--hidden-widths must contain two positive integers")
    try:
        validate_onnx_model_size(hidden_widths)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    initial_model_path = Path(args.initial_model) if args.initial_model else None
    parent_widths = None
    if initial_model_path is not None:
        if not initial_model_path.is_file():
            raise SystemExit(f"initial model does not exist: {initial_model_path}")
        try:
            parent_widths = _onnx_hidden_widths(initial_model_path)
        except (ValueError, KeyError) as error:
            raise SystemExit(str(error)) from error
        try:
            validate_onnx_model_size(parent_widths, initial_model_path.stat().st_size)
        except ValueError as error:
            raise SystemExit(f"initial model rejected: {error}") from error
        if any(target < parent for target, parent in zip(hidden_widths, parent_widths)):
            raise SystemExit(
                f"hidden-width shrinking is unsupported (parent {parent_widths}, "
                f"requested {hidden_widths}); no output was written"
            )
    set_seed(args.seed, args.threads)
    start = time.perf_counter()
    dataset_path = Path(args.data)
    validation_path = Path(args.validation_data) if args.validation_data else None
    counts, dataset_sha256, training_fingerprints = _scan_dataset(
        dataset_path, identify_games=validation_path is not None)
    if not counts:
        raise SystemExit(f"no games found in {dataset_path}")
    if validation_path is None:
        train_ids, validation_ids = _split_games(counts, args.seed, args.validation_fraction)
        validation_counts = counts
        validation_sha256 = dataset_sha256
        validation_fingerprints = []
        validation_mode = "internal_game_split"
    else:
        if not validation_path.is_file():
            raise SystemExit(f"validation dataset does not exist: {validation_path}")
        validation_counts, validation_sha256, validation_fingerprints = _scan_dataset(validation_path)
        if not validation_counts or not sum(validation_counts):
            raise SystemExit(f"no positions found in validation dataset {validation_path}")
        _require_disjoint_games(training_fingerprints, validation_fingerprints)
        train_ids = set(range(len(counts)))
        validation_ids = set(range(len(validation_counts)))
        validation_mode = "external_game_holdout"
    # Split complete games before applying any of the eight board symmetries.
    augmentation_mode = args.augmentation_mode
    train_arrays, validation_arrays = _load_arrays(
        dataset_path, counts, train_ids, validation_ids,
        augment=augmentation_mode == "all", validation_path=validation_path,
        validation_counts=validation_counts if validation_path is not None else None,
        expected_training_sha256=dataset_sha256,
        expected_validation_sha256=validation_sha256)
    train_x, train_p, train_v = train_arrays
    val_x, val_p, val_v = validation_arrays
    print(f"loaded {len(train_ids)} train / {len(validation_ids)} validation games; "
          f"{len(train_x):,} train positions/epoch ({augmentation_mode} augmentation), "
          f"{len(val_x):,} validation positions",
          flush=True)

    model = PolicyValueNet(hidden_widths)
    initial_model_sha256 = None
    initialization = {
        "method": "random_initialization" if initial_model_path is None else None,
        "parent_hidden_widths": list(parent_widths) if parent_widths is not None else None,
        "hidden_widths": list(hidden_widths),
        "seed": None,
        "noise_scale": 0.0,
        "preserves_function": False,
    }
    if initial_model_path is not None:
        initial_model_sha256, initialization = _load_initial_model_with_info(
            model, initial_model_path, seed=args.seed,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    if augmentation_mode == "all":
        x_tensor, p_tensor, v_tensor = map(torch.from_numpy, train_arrays)
    val_tensors = tuple(map(torch.from_numpy, validation_arrays))
    best_score = math.inf
    best_state = None
    best_epoch = 0
    initial_validation = None
    if args.initial_model:
        initial_policy, initial_value = _losses(model, *val_tensors, args.batch_size)
        initial_validation = {"policy_cross_entropy": initial_policy,
                              "value_mse": initial_value}
        initial_validation["diagnostics"] = _validation_diagnostics(
            val_p, val_v, train_v, initial_policy)
        best_score = initial_policy + args.value_weight * initial_value
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    epochs_actual = 0
    stopped_early = False
    stop_reason = "max_epochs"
    history: list[dict[str, float]] = []
    validation_target_entropy = _validation_diagnostics(
        val_p, val_v, train_v, 0.0)["policy_target_entropy"] or 0.0
    if augmentation_mode == "random":
        raw_maps = _symmetry_maps()
        maps = (np.stack([item[0] for item in raw_maps]),
                np.stack([item[1] for item in raw_maps]))
    else:
        maps = None
    augmentation_rng = np.random.default_rng(args.seed)
    shuffle_generator = None
    if args.shuffle_seed is not None:
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(args.shuffle_seed)
    optimizer_updates = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=shuffle_generator)
        for begin in range(0, len(order), args.batch_size):
            indices = order[begin:begin + args.batch_size]
            if augmentation_mode == "random":
                row_indices = indices.numpy()
                batch_x = train_x[row_indices]
                batch_p = train_p[row_indices]
                selected_symmetries = augmentation_rng.integers(0, 8, size=len(row_indices))
                batch_x, batch_p = _augment_batch(batch_x, batch_p, selected_symmetries, maps)
                x, target_p = torch.from_numpy(batch_x), torch.from_numpy(batch_p)
                target_v = torch.from_numpy(train_v[row_indices])
            else:
                x, target_p, target_v = x_tensor[indices], p_tensor[indices], v_tensor[indices]
            logits, prediction = model(x)
            logits = logits.masked_fill(x[:, 162:243] < 0.5, -1.0e9)
            policy_loss = -(target_p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(prediction[:, 0], target_v)
            loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            if args.max_updates is not None and optimizer_updates >= args.max_updates:
                break
        policy_val, value_val = _losses(model, *val_tensors, args.batch_size)
        score = policy_val + args.value_weight * value_val
        history.append({"epoch": epoch + 1, "optimizer_updates": optimizer_updates,
                        "validation_policy_loss": policy_val,
                        "validation_policy_kl": policy_val - validation_target_entropy,
                        "validation_value_mse": value_val, "validation_loss": score})
        epochs_actual = epoch + 1
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        print(f"epoch {epoch + 1}/{args.epochs}: policy_ce={policy_val:.4f} "
              f"value_mse={value_val:.4f}", flush=True)
        if args.max_updates is not None and optimizer_updates >= args.max_updates:
            stopped_early = True
            stop_reason = "max_updates"
            break
        if stale_epochs >= args.patience:
            stopped_early = True
            stop_reason = "patience"
            break
        if args.max_seconds is not None and time.perf_counter() - start >= args.max_seconds:
            stopped_early = True
            stop_reason = "max_seconds"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    output = Path(args.output)
    export_onnx(model, output)
    # Check the output interface and compare the trained PyTorch model numerically.
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    sample_x = val_x[:2] if len(val_x) else train_x[:2]
    policy_out, value_out = session.run(["policy", "value"], {"state": sample_x})
    if policy_out.shape != (len(sample_x), 81) or value_out.shape != (len(sample_x), 1):
        raise RuntimeError("exported ONNX has unexpected output shapes")
    if not np.isfinite(policy_out).all() or not np.isfinite(value_out).all():
        raise RuntimeError("exported ONNX produced non-finite values")
    export_parity = compare_torch_onnx(output, sample_x)
    val_policy, val_value = _losses(model, *val_tensors, args.batch_size)
    validation_diagnostics = _validation_diagnostics(val_p, val_v, train_v, val_policy)
    sidecar_path = dataset_path.with_suffix(".metadata.json")
    generation = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else None
    metadata = {
        "format": "onnx",
        "architecture": f"280-{hidden_widths[0]}-ReLU-{hidden_widths[1]}-ReLU-policy81-value1",
        "hidden_widths": list(hidden_widths),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input": {"name": "state", "shape": ["batch", 280], "dtype": "float32"},
        "outputs": [
            {"name": "policy", "shape": ["batch", 81], "meaning": "raw move logits"},
            {"name": "value", "shape": ["batch", 1], "meaning": "tanh estimate for the player to move"},
        ],
        "encoding": {"own_marks": [0, 81], "opponent_marks": [81, 162],
                     "legal_moves": [162, 243], "own_local_wins": [243, 252],
                     "opponent_local_wins": [252, 261], "drawn_local_boards": [261, 270],
                     "forced_board_onehot": [270, 280], "no_forced_board_index": 9},
        "training": {
            "dataset": _relative(dataset_path), "dataset_sha256": dataset_sha256,
            "validation_dataset": _relative(validation_path or dataset_path),
            "validation_dataset_sha256": validation_sha256,
            "validation_mode": validation_mode,
            "validation_game_identity": (
                "sha256 of ordered float32 features and value labels; excludes game index and policy"
                if validation_path is not None else None
            ),
            "games": len(counts),
            "train_games": len(train_ids), "validation_games": len(validation_ids),
            "train_positions_before_augmentation": sum(counts[index] for index in train_ids),
            "train_positions_after_augmentation": len(train_x),
            "train_positions_per_epoch": len(train_x),
            "validation_positions": len(val_x), "seed": args.seed,
            "shuffle_seed": args.shuffle_seed,
            "epochs": epochs_actual, "epochs_requested": args.epochs,
            "epochs_actual": epochs_actual,
            "max_updates": args.max_updates,
            "optimizer_updates": optimizer_updates,
            "selected_optimizer_updates": next(
                (item["optimizer_updates"] for item in history if item["epoch"] == best_epoch), 0),
            "batch_size": args.batch_size, "threads": args.threads,
            "best_epoch": best_epoch,
            "initial_model": _relative(Path(args.initial_model)) if args.initial_model else None,
            "initial_model_sha256": initial_model_sha256,
            "initialization": {
                key: value for key, value in initialization.items() if not key.startswith("_")
            },
            "initial_validation": initial_validation,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_weight,
            "augmentation_mode": augmentation_mode,
            "dihedral_augmentation": True,
            "patience": args.patience, "max_seconds": args.max_seconds,
            "stopped_early": stopped_early, "stop_reason": stop_reason,
            "epochs_without_improvement_at_stop": stale_epochs,
            "elapsed_seconds": round(time.perf_counter() - start, 2),
            "validation_policy_cross_entropy": val_policy,
            "validation_value_mse": val_value,
            "validation_diagnostics": validation_diagnostics,
            "history": history,
            "export_parity": export_parity,
        },
        "generation": generation,
        "artifact_bytes": output.stat().st_size,
        "artifact_size_limit_bytes": MAX_ONNX_MODEL_BYTES,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": _relative(output), "bytes": output.stat().st_size,
                      "sha256": metadata["sha256"], "validation_policy_ce": val_policy,
                      "validation_value_mse": val_value}, indent=2), flush=True)


def verify_onnx(args: argparse.Namespace) -> None:
    features = []
    with Path(args.data).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                game = json.loads(line)
                features.extend(sample[0] for sample in game["samples"][:2])
                break
    if not features:
        raise SystemExit("dataset has no positions to verify")
    errors = compare_torch_onnx(Path(args.model), np.asarray(features, dtype=np.float32))
    metadata_path = Path(args.metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = metadata.setdefault("training", {})
    training["export_parity"] = errors
    history = training.get("history", [])
    if history:
        training["best_epoch"] = min(history, key=lambda item: item["validation_loss"])["epoch"]
    metadata["sha256"] = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    metadata["artifact_bytes"] = Path(args.model).stat().st_size
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(errors, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train from TypeScript-generated game examples")
    fit.add_argument("--data", default=str(ROOT / "training" / "data" / "baseline.jsonl"))
    fit.add_argument("--validation-data",
                     help="separate immutable holdout JSONL; all --data games are used for training")
    fit.add_argument("--output", default=str(ROOT / "public" / "models" / "policy-value.onnx"))
    fit.add_argument("--metadata", default=str(ROOT / "public" / "models" / "metadata.json"))
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--seed", type=int, default=20261024)
    fit.add_argument("--shuffle-seed", type=int,
                     help="independent minibatch-order seed for controlled architecture comparisons")
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.add_argument("--weight-decay", type=float, default=0.0001)
    fit.add_argument("--value-weight", type=float, default=1.0)
    fit.add_argument("--validation-fraction", type=float, default=0.1)
    fit.add_argument("--initial-model", help="warm-start from an existing policy/value ONNX file")
    fit.add_argument("--hidden-widths", type=parse_hidden_widths, default=DEFAULT_HIDDEN_WIDTHS,
                     metavar="H1,H2", help="hidden layer widths (default: 256,128); warm starts may widen only")
    fit.add_argument("--threads", type=int, choices=(1,), default=1)
    fit.add_argument("--patience", type=int, default=2)
    fit.add_argument("--max-seconds", type=float,
                     help="stop after the current epoch once this elapsed time is reached")
    fit.add_argument("--max-updates", type=int,
                     help="stop after this many optimizer steps, validating the final partial epoch")
    fit.add_argument("--augmentation-mode", choices=("random", "all"), default="all")
    fit.set_defaults(run=train)
    verify = commands.add_parser("verify-onnx", help="compare the exported graph with its Torch weights")
    verify.add_argument("--data", default=str(ROOT / "training" / "data" / "baseline.jsonl"))
    verify.add_argument("--model", default=str(ROOT / "public" / "models" / "policy-value.onnx"))
    verify.add_argument("--metadata", default=str(ROOT / "public" / "models" / "metadata.json"))
    verify.set_defaults(run=verify_onnx)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
