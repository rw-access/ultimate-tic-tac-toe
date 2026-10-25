"""Focused tests for the Python data boundary and feature augmentation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.model import (
    MAX_ONNX_MODEL_BYTES,
    PolicyValueNet,
    estimate_onnx_model_bytes,
    export_onnx,
    validate_onnx_model_size,
)
from training.pipeline import (
    _augment_batch,
    _load_arrays,
    _load_initial_model,
    _load_initial_model_with_info,
    _net2wider_state,
    _read_sample,
    _require_disjoint_games,
    _scan_dataset,
    _split_games,
    _symmetry_maps,
    _validation_diagnostics,
    compare_torch_onnx,
    parse_hidden_widths,
    parser,
    train,
)


def sample():
    features = np.zeros(280, dtype=np.float32)
    features[0] = features[81 + 20] = 1
    features[162 + 0] = features[162 + 20] = features[162 + 80] = 1
    features[270 + 9] = 1
    policy = np.zeros(81, dtype=np.float32)
    policy[20] = 0.75
    policy[80] = 0.25
    return features.tolist(), policy.tolist(), -1


def game_record(game_id, marker, value=-1):
    features, policy, _ = sample()
    features[marker] = 1
    return {"game": game_id, "winner": "X", "moves": 1,
            "samples": [[features, policy, value]]}


def write_games(path, games):
    with path.open("w", encoding="utf-8") as sink:
        for game in games:
            sink.write(json.dumps(game) + "\n")


class TrainingInputTests(unittest.TestCase):
    def test_model_size_budget_accepts_current_network_and_rejects_oversize(self):
        self.assertEqual(MAX_ONNX_MODEL_BYTES, 1_500_000)
        current_estimate = estimate_onnx_model_bytes((512, 256))
        self.assertLessEqual(current_estimate, MAX_ONNX_MODEL_BYTES)
        self.assertGreater(estimate_onnx_model_bytes((600, 300)), MAX_ONNX_MODEL_BYTES)
        with self.assertRaisesRegex(ValueError, "exceeds the 1,500,000-byte maximum"):
            validate_onnx_model_size((600, 300))
        with self.assertRaisesRegex(ValueError, "exported ONNX model size"):
            validate_onnx_model_size((512, 256), MAX_ONNX_MODEL_BYTES + 1)

    def test_onnx_export_rejects_an_oversized_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oversized.onnx"

            def write_oversized(_model, _inputs, output_path, **_options):
                Path(output_path).write_bytes(b"x" * (MAX_ONNX_MODEL_BYTES + 1))

            with patch("training.model.torch.onnx.export", side_effect=write_oversized):
                with self.assertRaisesRegex(ValueError, "exported ONNX model size"):
                    export_onnx(PolicyValueNet((8, 4)), output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_train_rejects_oversized_architecture_before_loading_data(self):
        args = parser().parse_args([
            "train", "--data", "missing-training-data.jsonl",
            "--output", "unused.onnx", "--hidden-widths", "600,300",
        ])
        with self.assertRaisesRegex(SystemExit, "estimated ONNX model size"):
            train(args)

    def test_policy_must_only_assign_mass_to_legal_moves(self):
        features, policy, value = sample()
        policy[1] = 0.1
        with self.assertRaisesRegex(ValueError, "illegal move"):
            _read_sample(4, [features, policy, value])

    def test_symmetry_permutations_keep_legal_moves_and_policy_aligned(self):
        features, policy, value = sample()
        maps = _symmetry_maps()
        for feature_map, policy_map in maps:
            transformed = np.empty(280, dtype=np.float32)
            transformed_policy = np.empty(81, dtype=np.float32)
            transformed[feature_map] = features
            transformed_policy[policy_map] = policy
            legal = transformed[162:243] > 0.5
            self.assertTrue(np.all((transformed_policy > 0) <= legal))
            self.assertEqual(float(transformed_policy.sum()), 1.0)
            self.assertEqual(value, -1)

    def test_split_partitions_complete_game_ids(self):
        train, validation = _split_games([2, 4, 3, 5, 1, 8, 6, 7, 4, 2], 123, 0.2)
        self.assertEqual(train & validation, set())
        self.assertEqual(train | validation, set(range(10)))
        self.assertEqual(len(validation), 2)

    def test_random_symmetries_are_applied_per_item(self):
        features, policy, _ = sample()
        maps = _symmetry_maps()
        lookups = (np.stack([item[0] for item in maps]),
                   np.stack([item[1] for item in maps]))
        batch_x = np.stack((features, features)).astype(np.float32)
        batch_p = np.stack((policy, policy)).astype(np.float32)
        actual_x, actual_p = _augment_batch(
            batch_x, batch_p, np.asarray([0, 3]), lookups,
        )
        np.testing.assert_array_equal(actual_x[0], batch_x[0])
        np.testing.assert_array_equal(actual_p[0], batch_p[0])
        expected_x, expected_p = np.empty_like(batch_x[1]), np.empty_like(batch_p[1])
        expected_x[maps[3][0]] = batch_x[1]
        expected_p[maps[3][1]] = batch_p[1]
        np.testing.assert_array_equal(actual_x[1], expected_x)
        np.testing.assert_array_equal(actual_p[1], expected_p)

    def test_initial_onnx_loads_as_exact_warm_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.onnx"
            source = PolicyValueNet()
            export_onnx(source, path)
            restored = PolicyValueNet()
            digest = _load_initial_model(restored, path)
        self.assertEqual(len(digest), 64)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_net2wider_preserves_outputs_for_both_expanded_layers(self):
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parent.onnx"
            parent = PolicyValueNet().eval()
            export_onnx(parent, path)
            widened = PolicyValueNet((512, 256)).eval()
            digest, info = _load_initial_model_with_info(widened, path, seed=812)
            self.assertEqual(len(digest), 64)
            self.assertEqual(info["method"], "net2wider_zero_sum_perturbation")
            self.assertEqual(info["parent_hidden_widths"], [256, 128])
            self.assertEqual(info["hidden_widths"], [512, 256])
            self.assertEqual(info["seed"], 812)
            self.assertTrue(info["preserves_function"])
            rng = np.random.default_rng(13)
            states = rng.normal(0.15, 0.5, size=(24, 280)).astype(np.float32)
            with torch.no_grad():
                parent_outputs = parent(torch.from_numpy(states))
                widened_outputs = widened(torch.from_numpy(states))
            for original, expanded in zip(parent_outputs, widened_outputs):
                self.assertLess(float(torch.max(torch.abs(original - expanded))), 1e-5)

    def test_net2wider_duplicate_units_receive_distinct_gradients(self):
        torch.manual_seed(71)
        parent = PolicyValueNet((5, 3))
        widened = PolicyValueNet((9, 6))
        widened_state, maps = _net2wider_state(parent.state_dict(), (9, 6), seed=93)
        widened.load_state_dict(widened_state)
        rng = np.random.default_rng(7)
        states = torch.from_numpy(rng.normal(size=(16, 280)).astype(np.float32))
        target_policy = torch.softmax(torch.randn(16, 81), dim=1)
        target_value = torch.linspace(-1, 1, 16)
        logits, values = widened(states)
        loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss = loss + (values[:, 0] - target_value).square().mean()
        loss.backward()
        first_map, second_map = maps["first_map"], maps["second_map"]
        first_duplicate = next(
            np.flatnonzero(first_map == source_index)[:2]
            for source_index in range(5)
            if np.count_nonzero(first_map == source_index) > 1
        )
        second_duplicate = next(
            np.flatnonzero(second_map == source_index)[:2]
            for source_index in range(3)
            if np.count_nonzero(second_map == source_index) > 1
        )
        grad1 = widened.trunk[0].weight.grad
        grad2 = widened.trunk[2].weight.grad
        self.assertGreater(float(torch.max(torch.abs(grad1[first_duplicate[0]] - grad1[first_duplicate[1]]))), 1e-9)
        self.assertGreater(float(torch.max(torch.abs(grad2[second_duplicate[0]] - grad2[second_duplicate[1]]))), 1e-9)

    def test_onnx_parity_supports_default_and_widened_architectures(self):
        with tempfile.TemporaryDirectory() as temporary:
            rng = np.random.default_rng(108)
            states = rng.uniform(-0.2, 1.0, size=(5, 280)).astype(np.float32)
            for widths in ((256, 128), (384, 192)):
                model = PolicyValueNet(widths)
                path = Path(temporary) / f"{widths[0]}-{widths[1]}.onnx"
                export_onnx(model, path)
                errors = compare_torch_onnx(path, states)
                self.assertLess(errors["policy_max_abs_error"], 1e-4)
                self.assertLess(errors["value_max_abs_error"], 1e-4)

    def test_hidden_width_parser_and_warm_start_reject_shrinking(self):
        self.assertEqual(parse_hidden_widths("512, 256"), (512, 256))
        for invalid in ("512", "512,0", "512,-1", "a,256", "512,256,128"):
            with self.assertRaises(Exception):
                parse_hidden_widths(invalid)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parent.onnx"
            export_onnx(PolicyValueNet((8, 4)), path)
            with self.assertRaisesRegex(ValueError, "shrinking is unsupported"):
                _load_initial_model_with_info(PolicyValueNet((8, 3)), path, seed=1)
            output_path = Path(temporary) / "should-not-exist.onnx"
            args = parser().parse_args([
                "train", "--initial-model", str(path), "--hidden-widths", "8,3",
                "--data", str(Path(temporary) / "missing.jsonl"), "--output", str(output_path),
            ])
            with self.assertRaisesRegex(SystemExit, "shrinking is unsupported"):
                train(args)
            self.assertFalse(output_path.exists())

    def test_replay_defaults_keep_all_symmetries_but_cap_cpu_to_one(self):
        args = parser().parse_args(["train"])
        self.assertEqual(args.augmentation_mode, "all")
        self.assertEqual(args.threads, 1)
        self.assertEqual(args.patience, 2)
        self.assertIsNone(args.validation_data)
        self.assertEqual(args.hidden_widths, (256, 128))
        self.assertIsNone(args.shuffle_seed)
        self.assertIsNone(args.max_updates)

    def test_optimizer_update_limit_stops_mid_epoch_and_exports_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "train.jsonl"
            validation = directory / "validation.jsonl"
            write_games(data, [game_record(index, 30 + index) for index in range(5)])
            write_games(validation, [game_record(99, 45, value=1)])
            for limit, epochs, expected_updates, expected_epochs in ((4, 10, 4, 2), (None, 2, 6, 2)):
                output = directory / f"model-{limit}.onnx"
                metadata = directory / f"model-{limit}.json"
                command = [
                    sys.executable, "-m", "training.pipeline", "train",
                    "--data", str(data), "--validation-data", str(validation),
                    "--hidden-widths", "8,4", "--augmentation-mode", "random",
                    "--epochs", str(epochs), "--batch-size", "2", "--patience", "100",
                    "--output", str(output), "--metadata", str(metadata),
                ]
                if limit is not None:
                    command.extend(["--max-updates", str(limit)])
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                model_metadata = json.loads(metadata.read_text())
                training = model_metadata["training"]
                self.assertEqual(training["optimizer_updates"], expected_updates)
                self.assertEqual(training["epochs_actual"], expected_epochs)
                self.assertEqual(training["stop_reason"], "max_updates" if limit else "max_epochs")
                self.assertEqual([item["optimizer_updates"] for item in training["history"]],
                                 [3, 4] if limit else [3, 6])
                self.assertGreater(output.stat().st_size, 0)
                self.assertLessEqual(output.stat().st_size, MAX_ONNX_MODEL_BYTES)
                self.assertEqual(model_metadata["artifact_bytes"], output.stat().st_size)
                self.assertEqual(model_metadata["artifact_size_limit_bytes"], MAX_ONNX_MODEL_BYTES)
                self.assertLess(training["export_parity"]["policy_max_abs_error"], 1e-4)
            args = parser().parse_args(["train", "--max-updates", "0"])
            with self.assertRaisesRegex(SystemExit, "--max-updates must be positive"):
                train(args)

    def test_array_loader_augments_only_training_games(self):
        features, policy, value = sample()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small.jsonl"
            with path.open("w", encoding="utf-8") as sink:
                for game in range(4):
                    sink.write(json.dumps({"game": game, "samples": [[features, policy, value]]}) + "\n")
            train, validation = _load_arrays(path, [1] * 4, {0, 1}, {2, 3}, True)
        self.assertEqual(train[0].shape, (16, 280))
        self.assertEqual(validation[0].shape, (2, 280))
        np.testing.assert_array_equal(validation[0][0], features)
        np.testing.assert_array_equal(validation[1][0], policy)

    def test_external_validation_loads_all_training_games_without_augmenting_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            write_games(training_path, [game_record(i, i + 1) for i in range(3)])
            holdout = [game_record(i + 10, i + 4, i % 2) for i in range(2)]
            write_games(validation_path, holdout)
            train_counts, train_hash, train_fingerprints = _scan_dataset(training_path)
            validation_counts, validation_hash, validation_fingerprints = _scan_dataset(validation_path)
            _require_disjoint_games(train_fingerprints, validation_fingerprints)
            train, validation = _load_arrays(
                training_path, train_counts, set(range(3)), set(range(2)), True,
                validation_path=validation_path, validation_counts=validation_counts,
                expected_training_sha256=train_hash,
                expected_validation_sha256=validation_hash,
            )
        self.assertEqual(len(train_hash), 64)
        self.assertEqual(len(validation_hash), 64)
        self.assertEqual(train[0].shape, (24, 280))
        self.assertEqual(validation[0].shape, (2, 280))
        np.testing.assert_array_equal(validation[0][0], holdout[0]["samples"][0][0])
        np.testing.assert_array_equal(validation[1][1], holdout[1]["samples"][0][1])

    def test_external_overlap_ignores_game_index_and_policy_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            training_game = game_record(1, 4, -1)
            validation_game = game_record(999, 4, -1)
            validation_game["samples"][0][1][20] = 0.25
            validation_game["samples"][0][1][80] = 0.75
            write_games(training_path, [training_game])
            write_games(validation_path, [validation_game])
            _, _, training_fingerprints = _scan_dataset(training_path)
            _, _, validation_fingerprints = _scan_dataset(validation_path)
        self.assertEqual(training_fingerprints, validation_fingerprints)
        with self.assertRaisesRegex(ValueError, "overlaps training data in 1 exact game"):
            _require_disjoint_games(training_fingerprints, validation_fingerprints)

    def test_dataset_hash_is_rechecked_while_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            write_games(path, [game_record(1, 4)])
            counts, data_hash, _ = _scan_dataset(path)
            with path.open("a", encoding="utf-8") as sink:
                sink.write("\n")
            with self.assertRaisesRegex(ValueError, "changed after its hash"):
                _load_arrays(path, counts, {0}, set(), False,
                             expected_training_sha256=data_hash)

    def test_validation_diagnostics_separate_target_entropy_and_value_baselines(self):
        policies = np.zeros((2, 81), dtype=np.float32)
        policies[0, 0] = 1
        policies[1, :2] = 0.5
        entropy = np.log(2) / 2
        values = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
        training_values = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
        diagnostics = _validation_diagnostics(
            policies, values, training_values, policy_cross_entropy=entropy + 0.2,
        )
        self.assertAlmostEqual(diagnostics["policy_target_entropy"], entropy)
        self.assertAlmostEqual(diagnostics["policy_kl"], 0.2)
        self.assertAlmostEqual(diagnostics["value_target_mean"], 2 / 3)
        self.assertAlmostEqual(diagnostics["value_target_variance"], 2 / 9)
        self.assertAlmostEqual(diagnostics["value_zero_mse"], 2 / 3)
        self.assertAlmostEqual(diagnostics["value_empirical_mean_mse"], 2 / 9)
        self.assertAlmostEqual(diagnostics["value_train_mean_mse"], 11 / 9)


if __name__ == "__main__":
    unittest.main()
