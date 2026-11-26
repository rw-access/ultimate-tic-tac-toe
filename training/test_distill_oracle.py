import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

from training.distill_oracle import children, distribution, symmetry_evaluator, targets, validate


def zero_teacher(x, value_only=False):
    value = np.zeros(len(x), np.float32)
    return value if value_only else (np.zeros((len(x), 81), np.float32), value)


class OracleTests(unittest.TestCase):
    def test_eight_view_probabilities_child_values_and_equivariance(self):
        from training.pipeline import _symmetry_maps

        # After X plays board 0, cell 1, O has all nine cells of board 1.
        x = np.zeros((1, 280), np.uint8)
        x[0, 81 + 1] = x[0, 270 + 1] = 1
        x[0, 162 + 9:162 + 18] = 1
        original = x[:, 162:243].astype(np.float32) / 9
        batch_lengths = []

        def directional_teacher(states, value_only=False):
            batch_lengths.append(len(states))
            root = states[:, :81].sum(1) == 0
            value = np.where(root, .125 + .5 * states[:, 81 + 1],
                             -.5 + .25 * states[:, 81 + 15]).astype(np.float32)
            if value_only:
                return value
            logits = np.zeros((len(states), 81), dtype=np.float32)
            logits[:, 15] = 2 * states[:, 81 + 1]
            logits[states[:, 162:243] == 0] = 10_000  # must be masked per view
            return logits, value

        ensemble = symmetry_evaluator(directional_teacher, batch_size=3)
        logits, root_v = ensemble(x)
        self.assertTrue(np.isfinite(logits).all())
        self.assertAlmostEqual(root_v.item(), .125 + .5 / 8)
        # Exactly one orientation of this asymmetric root triggers the boosted prior.
        single = np.ones(9)
        single[6] = np.exp(2.)
        expected = (7 / 9 + single / single.sum()) / 8
        prior = distribution(logits, x[:, 162:243].astype(bool))
        np.testing.assert_allclose(prior[0, 9:18], expected, rtol=1e-6)
        self.assertEqual(prior[0, x[0, 162:243] == 0].sum(), 0)

        p, v, solved = targets(x, original, np.zeros(1), ensemble, 3, 1, .25, .75, 0.)
        # Move 15 and its reflected counterpart 17 each hit the teacher's
        # child-value feature in one orientation. Convert to the root perspective.
        q = np.full(9, .5)
        q[[6, 8]] -= .25 / 8
        expected *= np.exp(q / .25)
        expected /= expected.sum()
        np.testing.assert_allclose(p[0, 9:18], expected, atol=6e-5, rtol=0)
        self.assertAlmostEqual(v.item(), .25 * root_v.item() + .75 * np.dot(expected, q), places=6)
        self.assertEqual(solved, 0)
        for feature_map, move_map in _symmetry_maps():
            transformed = np.empty_like(x)
            transformed[:, feature_map] = x
            transformed_p = np.empty_like(original)
            transformed_p[:, move_map] = original
            tp, tv, _ = targets(transformed, transformed_p, np.zeros(1), ensemble, 3, 1, .25, .75, 0.)
            np.testing.assert_array_equal(tp[:, move_map], p)
            np.testing.assert_allclose(tv, v, atol=1e-7)
        self.assertEqual(max(batch_lengths), 3)

    def test_play_and_encoding_match_project_for_complete_seeded_games(self):
        # Use the actual browser rules and encoder to generate expected successors.
        script = """
            import {createGame, play} from './src/game.ts';
            import {encodeState, legalMoves} from './src/ai.ts';
            let seed = 9187;
            const rows = [];
            for (let game = 0; game < 32; game++) {
                let s = createGame();
                while (s.winner === null) {
                    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                    const moves = legalMoves(s), move = moves[seed % moves.length];
                    const t = play(s, Math.floor(move / 9), move % 9);
                    rows.push([encodeState(s).join(''), move, encodeState(t).join(''),
                               t.winner, s.turn]);
                    s = t;
                }
            }
            console.log(JSON.stringify(rows));
        """
        root = Path(__file__).resolve().parents[1]
        run = subprocess.run(["node", "--input-type=module", "-e", script], cwd=root,
                             check=True, text=True, capture_output=True)
        records = json.loads(run.stdout)
        def states(column):
            values = "".join(row[column] for row in records).encode()
            return (np.frombuffer(values, dtype=np.uint8) - ord("0")).reshape(-1, 280)
        x, expected = states(0), states(2)
        outcome = [r[3] for r in records]
        self.assertEqual(set(outcome), {None, "X", "O", "draw"})
        self.assertTrue(x[:, 261:270].any())  # already-drawn mini-boards
        self.assertTrue((x[:, 279] & x[:, 243:270].any(1)).any())  # free after closing a board
        self.assertTrue(x[:, 270:279].any())  # forced board
        p = x[:, 162:243].astype(np.float32)
        p /= p.sum(1, keepdims=True)
        validate(x, p, np.zeros(len(x)))
        child, terminal, win = children(x, np.arange(len(x)), np.array([row[1] for row in records]))
        np.testing.assert_array_equal(child, expected)
        np.testing.assert_array_equal(terminal, [r[3] is not None for r in records])
        np.testing.assert_array_equal(win, [r[3] == r[4] for r in records])
        # An immediately provable win overrides even maximally contradictory source labels.
        i = np.flatnonzero(win)[0]
        new_p, v, proven = targets(x[i:i+1], p[i:i+1], np.array([-1.], np.float32),
                                    zero_teacher, 32, 1, .2, 0., 1.)
        self.assertEqual((v[0], proven), (1., 1))
        self.assertGreater(new_p[0, records[i][1]], 0.)
        legal_moves = np.flatnonzero(x[i, 162:243])
        _, _, winning = children(x[i:i+1], np.zeros(len(legal_moves), int), legal_moves)
        np.testing.assert_array_equal(new_p[0, legal_moves] > 0, winning)

    def test_soft_backup_child_perspective_and_control(self):
        x = np.zeros((1, 280), np.uint8)
        x[0, 162:243] = 1
        x[0, 279] = 1
        original = np.full((1, 81), 1 / 81, np.float16)
        def teacher(states, value_only=False):
            if value_only:
                # A negative child value means a better move for the root.
                return np.where(states[:, 81 + 13] > 0, -.5, .5).astype(np.float32)
            return np.zeros((len(states), 81), np.float32), np.full(len(states), .2, np.float32)
        uniform, root_v, _ = targets(x, original, np.array([0.]), teacher, 5, 0, .2, .5, 0.)
        np.testing.assert_array_equal(uniform, original)
        self.assertAlmostEqual(root_v.item(), .2, places=5)
        p, v, solved = targets(x, original, np.array([0.]), teacher, 5, 1, .2, .5, 0.)
        expected_p = np.exp(5.) / (80 + np.exp(5.))
        self.assertEqual(solved, 0)
        self.assertAlmostEqual(p[0, 13].item(), expected_p, places=3)
        self.assertAlmostEqual(v.item(), .5 * .2 + .5 * (expected_p - .5), places=5)
        x[0, 162] = 0
        with self.assertRaisesRegex(ValueError, "legal-move mask"):
            targets(x, original, np.array([0.]), teacher, 5, 1, .2, .5, 0.)


if __name__ == "__main__":
    unittest.main()
