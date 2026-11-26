import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from training.utttai_stage2 import convert, parse_record


def drawn_board_record(value):
    # X has played five marks, O four. The finished top-left board is drawn.
    # O must play the middle board; move indexes 36..44 have 1..9 visits.
    state = "121122211" + "0" * 72 + "300000000" + "240"
    actions = ",".join(f"2 {i} {i-35} 0" for i in range(36, 45))
    return f"evaluatedState{{{state} 46 {value}}} evaluatedActions{{{actions}}}".encode()


class Stage2Tests(unittest.TestCase):
    def test_encoding_and_perspective(self):
        x, p, v = parse_record(drawn_board_record(-0.73))
        self.assertEqual(v, -0.73)
        self.assertEqual(x.shape, (280,))
        self.assertEqual(set(np.flatnonzero(x[:81])), {1, 4, 5, 6})
        self.assertEqual(set(np.flatnonzero(x[81:162])), {0, 2, 3, 7, 8})
        self.assertEqual(list(np.flatnonzero(x[162:243])), list(range(36, 45)))
        self.assertEqual(list(np.flatnonzero(x[243:270])), [18])
        self.assertEqual(list(np.flatnonzero(x[270:])), [4])
        np.testing.assert_allclose(p[36:45], np.arange(1, 10) / 45)
        self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_free_board_and_invalid_records(self):
        state = "0" * 90 + "190"
        actions = ",".join(f"1 {i} 1 0" for i in range(81))
        record = f"evaluatedState{{{state} 82 0}} evaluatedActions{{{actions}}}".encode()
        x, _, _ = parse_record(record)
        self.assertTrue(x[162:243].all())
        self.assertEqual(list(np.flatnonzero(x[270:])), [9])
        for invalid in (
            record.replace(b"82 0}", b"82 nan}"), record.replace(b"1 0 1 0", b"2 0 1 0"),
            record.replace(b"1 1 1 0", b"1 0 1 0"), record.replace(b"1 0 1 0,", b""),
            record.replace(b"1 0 1 0", b"1 0 -1 0"), record.replace(b"190 82", b"191 82"),
            drawn_board_record(0).replace(b"300000000240", b"300000000200"),
            drawn_board_record(0).replace(b"2 36 1 0", b"2 45 1 0"),
        ):
            with self.subTest(invalid=invalid[:100]), self.assertRaises(ValueError):
                parse_record(invalid)

    def test_deterministic_depth_sampling_and_atomic_failure(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            data.mkdir()
            for i, size in enumerate((12, 8)):
                depth = data / str(i)
                depth.mkdir()
                (depth / "depth.txt").write_bytes(b"\n".join(drawn_board_record(n / 20) for n in range(size)))
            for name in ("a", "b"):
                result = convert([str(data)], base / name, maximum=10, seed=3, shard_size=4)
                self.assertEqual([s["sampled"] for s in result["sources"]], [6, 4])
                self.assertEqual([s["positions"] for s in result["shards"]], [4, 4, 2])
                self.assertEqual(json.loads((base / name / "manifest.json").read_text()), result)
            for name in sorted((base / "a").glob("*.npz")):
                with np.load(name) as a, np.load(base / "b" / name.name) as b:
                    self.assertEqual((a["x"].dtype, a["p"].dtype, a["v"].dtype),
                                     (np.uint8, np.float16, np.float32))
                    np.testing.assert_array_equal(a["v"], b["v"])
            (data / "zz-bad.txt").write_bytes(b"malformed")
            with self.assertRaisesRegex(ValueError, "zz-bad.txt:1"):
                convert([str(data)], base / "bad", shard_size=2)
            self.assertFalse((base / "bad").exists())
