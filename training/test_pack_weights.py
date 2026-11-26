"""Regression checks for packed kernels, native output parity, and size caps."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.pack_weights import byte_lookup, check_float16_range, layout, pack, quantize, quantize_int6, quantize_int8


class CodebookTests(unittest.TestCase):
    def test_bit_order(self):
        self.assertEqual(byte_lookup(2)[0b11100100].tolist(), [0, 1, 2, 3])
        self.assertEqual(byte_lookup(4)[0xB3].tolist(), [3, 11])

    def test_six_bit_two_stream_round_trip(self):
        weights = np.r_[np.arange(-32, 32), np.zeros(64), -1.2, .6, 3.].astype(np.float32)
        high, low, scales, decoded = quantize_int6(weights, 64)
        upper = byte_lookup(4)[high].ravel()
        lower = byte_lookup(2)[low].ravel()
        signed = (upper * 4 + lower - 32).reshape(len(scales), -1)
        np.testing.assert_array_equal((signed.astype(np.float32) * scales).ravel()[:len(weights)], decoded)
        np.testing.assert_array_equal(decoded[:128], weights[:128])
        self.assertEqual(high.nbytes + low.nbytes, signed.size * 3 // 4)
        with self.assertRaises(ValueError):
            quantize_int6(np.array([np.nan], dtype=np.float32), 64)

    def test_real_tail_constant_and_skewed_groups(self):
        rng = np.random.default_rng(53)
        # One constant group; skewed, non-symmetric weights; odd tail.
        values = np.r_[np.full(80, -0.625), rng.lognormal(-2, 1.5, 319)].astype(np.float32)
        err = {}
        for bits in (2, 4):
            octets, table, dense = quantize(values, bits, 80, 6)
            codes = byte_lookup(bits)[octets].reshape(len(table), -1)
            independent = np.take_along_axis(table, codes, axis=1).flatten()[:len(values)]
            np.testing.assert_array_equal(dense, independent)
            np.testing.assert_array_equal(dense[:80], values[:80])
            np.testing.assert_array_less(-1e-9, np.diff(table.astype(float), axis=1) + 1e-10)
            _, _, initial = quantize(values, bits, 80, 0)
            err[bits] = np.square(dense - values).sum()
            self.assertLessEqual(err[bits], float(np.square(initial - values).sum()) + 1e-9)
        self.assertLess(err[4], err[2])

    def test_signed_blocks_round_and_trim_exactly(self):
        # Zero, symmetric extrema, and a final two-value block with a different scale.
        x = np.array([0, 0, 0, 0, -127, -63.5, 63.5, 127, -1e-5, 1e-5], np.float32)
        codes, scales, decoded = quantize_int8(x.reshape(2, 5), 4)
        np.testing.assert_array_equal(codes, [[0, 0, 0, 0], [-127, -64, 64, 127], [-127, 127, 0, 0]])
        self.assertEqual(codes.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float32)
        self.assertEqual(scales.shape, (3, 1))
        self.assertEqual(decoded.shape, (2, 5))
        np.testing.assert_array_equal(decoded.ravel(), (codes.astype(np.float32) * scales).ravel()[:x.size])
        smallest = np.nextafter(np.float32(0), np.float32(1))
        tiny = np.array([smallest * 129, -smallest], np.float32)
        c, s, y = quantize_int8(tiny, 8)
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.max(np.abs(tiny - y))), float(s[0, 0] / 2))
        for value in [np.inf, np.nan, np.finfo(np.float32).max]:
            with self.assertRaises(ValueError):
                quantize_int8(np.array([value], np.float32), 8)

    def test_extreme_or_non_finite_rejected(self):
        for value in [1e6, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                check_float16_range(np.array([value], np.float32))
            with self.assertRaises(ValueError):
                quantize(np.array([value], np.float32), 2, 32, 0)
        with self.assertRaises(ValueError):
            layout(100, 11, 2)


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "onnx and onnxruntime required for graph integration")
class NativeGraphTests(unittest.TestCase):
    def test_standalone_native_model_and_budget(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.default_rng(231)
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 280])]
        outputs = [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 81]),
                   helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch", 1])]
        weights = [numpy_helper.from_array(rng.normal(0, .02, (280, n)).astype(np.float32), name)
                   for n, name in [(81, "p_weight"), (1, "v_weight")]]
        nodes = [helper.make_node("MatMul", ["x", "p_weight"], ["policy"]),
                 helper.make_node("MatMul", ["x", "v_weight"], ["v_raw"]),
                 helper.make_node("Tanh", ["v_raw"], ["value"])]
        model = helper.make_model(helper.make_graph(nodes, "test", inputs, outputs, weights),
                                  opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d / "teacher.onnx"
            onnx.save(model, path)
            src_bytes = path.read_bytes()
            for bits in (2, 4, 6, 8, 16):
                dest = d / f"{bits}.onnx"
                args = argparse.Namespace(input=path, output=dest, bits=bits, max_bytes=len(src_bytes) * (3 if bits == 16 else 2) // 4,
                                          group_size=None if bits == 16 else 80, iterations=3, parity=Path(__file__).with_name("parity.json"),
                                          data=None, samples=10, seed=83, batch_size=16, report=None)
                report = pack(args)
                self.assertEqual(report["packed_bytes"], dest.stat().st_size)
                self.assertEqual(report["packed_weights"], 280 * 82)
                self.assertGreater(report["drift"]["live_positions"], 1)
                self.assertLess(report["drift"]["graph_dense_max_logit_error"], 1e-5)
                packed_bytes = dest.read_bytes()
                if bits == 6:
                    self.assertEqual(report["exactly_reconstructed_tensors"], len(weights))
                    self.assertLess(report["relative_weight_rmse"], .04)
                if bits == 16:
                    graph = onnx.load(dest).graph
                    self.assertEqual([n.op_type for n in graph.node], ["Cast", "Cast", "MatMul", "MatMul", "Tanh"])
                    self.assertTrue(all(n.attribute[0].i == TensorProto.FLOAT for n in graph.node[:2]))
                    for src, stored in zip(weights, graph.initializer, strict=True):
                        self.assertEqual(stored.data_type, TensorProto.FLOAT16)
                        np.testing.assert_array_equal(numpy_helper.to_array(stored), numpy_helper.to_array(src).astype(np.float16))
                    self.assertIsNone(report["group_size"])
                    self.assertEqual(report["iterations"], 0)
                    self.assertLess(report["relative_weight_rmse"], 0.0003)
                    self.assertLess(report["drift"]["policy_logit_rmse"], 0.0002)
                    self.assertLess(report["drift"]["value_max_error"], 0.0002)
                if bits == 8:
                    self.assertEqual(report["exactly_reconstructed_tensors"], len(weights))
                    graph = onnx.load(dest).graph
                    self.assertEqual(sum(n.op_type == "Mul" for n in graph.node), len(weights))
                    self.assertFalse(any(n.op_type == "Gather" for n in graph.node))
                    self.assertFalse({t.name for t in weights} & {t.name for t in graph.initializer})
                    codes = [t for t in graph.initializer if t.data_type == TensorProto.INT8]
                    scales = [t for t in graph.initializer if t.data_type == TensorProto.FLOAT]
                    self.assertEqual(len(codes), len(weights))
                    self.assertEqual(len(scales), len(weights))  # every float32 initializer is a compact scale
                    for src, c, s in zip(weights, codes, scales, strict=True):
                        expected_codes, expected_scales, _ = quantize_int8(numpy_helper.to_array(src), 80)
                        np.testing.assert_array_equal(numpy_helper.to_array(c), expected_codes)
                        np.testing.assert_array_equal(numpy_helper.to_array(s), expected_scales)
                    self.assertEqual(report["iterations"], 0)
                    self.assertLess(report["relative_weight_rmse"], .009)
                    self.assertLess(report["drift"]["policy_logit_rmse"], .002)
                    self.assertLess(report["drift"]["graph_dense_max_value_error"], 1e-5)
                args.max_bytes = report["packed_bytes"] - 1
                with self.assertRaisesRegex(ValueError, "Candidate exceeds cap" if bits == 16 else "No group size fits"):
                    pack(args)
                self.assertEqual(dest.read_bytes(), packed_bytes)
            self.assertEqual(path.read_bytes(), src_bytes)


if __name__ == "__main__":
    unittest.main()
