import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.arena_scorecard import opening_state
from training.suite_exclusion import Exclusion
from training.utttai_stage2 import parse_record


# Legal board-major histories recovered independently from the 50 published positions.
HISTORIES = [
    [0], [1], [2], [4], [5], [8], [9], [10], [12], [13], [15], [16], [36], [37], [40],
    [40, 44, 80, 78, 60], [40, 42, 60, 54, 8], [40, 38, 20, 24, 54], [40, 42, 60, 56, 20],
    [40, 36, 8, 72, 5], [40, 42, 56, 24, 54], [40, 44, 72, 2, 20], [40, 36, 8, 78, 56],
    [40, 42, 54, 0, 8], [40, 36, 2, 24, 56], [40, 38, 23], [40, 44, 76], [40, 38, 21],
    [10, 16, 70, 66, 32, 48], [38, 19, 11, 18, 5, 51], [73, 10, 14, 47, 25, 66],
    [4, 40, 36, 2, 19, 12], [18, 4, 42, 54, 8, 75], [39, 32, 46, 15, 60, 58],
    [62, 72, 4, 40, 36, 2], [17, 80, 73, 14, 50, 48], [42, 58, 44, 78, 55, 11],
    [18, 0, 6, 56, 26, 76], [50, 48, 31, 40, 37, 10], [40, 41, 48, 35, 80, 72],
    [37, 9, 3, 33, 59, 52], [79, 65, 19, 11, 18, 0], [24, 58, 43, 64, 15, 59],
    [16, 65, 24, 59, 48, 27], [60, 58, 43, 64, 17, 80], [44, 74, 18, 2, 24, 59],
    [69, 59, 49, 39, 29, 19], [64, 16, 68, 48, 34, 65], [58, 38, 24, 54, 4, 40],
    [43, 66, 29, 25, 67, 44],
]


def fixture_bytes():
    headings = {0: 'all unique states at depth 1', 15: 'top opening states', 28: 'random positions at depth 6'}
    lines = ['EVALUATION_UTTT_STATES = [']
    for i, moves in enumerate(HISTORIES):
        if i in headings:
            lines.append(f'    # {headings[i]}:')
        lines.append(f'    "{opening_state(moves)}",')
    return ('\n'.join(lines) + '\n]\n').encode()


def feature_row(moves):
    state = opening_state(moves)
    forced = int(state[91])
    moves = [i for i in range(81) if state[i] == '0' and forced in (9, i // 9) and state[81 + i // 9] == '0']
    actions = ','.join(f'{state[90]} {i} 1 0' for i in moves)
    return parse_record(f'evaluatedState{{{state} {len(moves)} 0}} evaluatedActions{{{actions}}}'.encode())[0]


class SuiteExclusionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'evaluation_uttt_states.py'
        self.source.write_bytes(fixture_bytes())
        self.exclusion = Exclusion(self.source)

    def test_complete_recovered_suite_and_rotations(self):
        rows = []
        for moves in HISTORIES:
            for flip in (False, True):
                for turns in range(4):
                    def mapped(index):
                        r, c = divmod(index, 3)
                        for _ in range(turns):
                            r, c = 2 - c, r
                        return 3 * r + (2 - c if flip else c)
                    rows.append(feature_row([9 * mapped(i // 9) + mapped(i % 9) for i in moves]))
        ids = self.exclusion.identities(np.stack(rows).astype(np.float32))
        np.testing.assert_array_equal(ids, np.repeat(np.arange(50), 8))
        self.assertEqual(len(self.exclusion.features), 353)

    def test_keeps_unrelated_states_and_checks_full_features(self):
        row = feature_row(HISTORIES[-1]).copy()
        row[270:280] = 0
        row[279] = 1
        batch = np.stack([feature_row([]), row, feature_row([40, 44]), feature_row([40])])
        np.testing.assert_array_equal(self.exclusion.keep(batch), [True, True, True, False])
        self.assertEqual(self.exclusion.keep(batch[:0]).shape, (0,))

    def test_rejects_changed_source_and_nonbinary_inputs(self):
        self.source.write_bytes(fixture_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            Exclusion(self.source)
        for invalid in [np.zeros((2, 279)), np.full((1, 280), .5), np.full((1, 280), 2, np.uint8)]:
            with self.assertRaisesRegex(ValueError, 'binary'):
                self.exclusion.identities(invalid)


if __name__ == '__main__':
    unittest.main()
