import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.exact_targets import classify, eligible, proved_targets


class ExactTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if supplied := os.environ.get("UTTT_EXACT_SOLVER"):
            cls.solver = Path(supplied).resolve(strict=True)
            return
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.solver = Path(cls.temp.name) / "exact"
        subprocess.run(["c++", "-std=c++17", "-O2", "-pthread", str(Path(__file__).with_name("exact_targets.cpp")),
                        "-o", str(cls.solver)], check=True)

    def run_solver(self, x, nodes, threads):
        with subprocess.Popen([str(self.solver), str(nodes), str(threads)], text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
            selected, classes, used = classify(worker, x)
            worker.stdin.close()
            self.assertEqual(worker.wait(), 0)
        return selected, classes, used

    def test_wdl_matches_exhaustive_project_rules_and_budget_is_deterministic(self):
        script = """
        import {createGame, play} from './src/game.ts';
        import {legalMoves, encodeState} from './src/ai.ts';
        const rows = [];
        let random = 20261126;
        function evalMove(s, m) {
          const t = play(s, Math.floor(m/9), m%9);
          if (t.winner !== null) return t.winner === 'draw' ? 0 : 1;
          return -Math.max(...legalMoves(t).map(a => evalMove(t, a)));
        }
        for (let game = 0; game < 64; game++) {
          let s = createGame();
          while (!s.winner) {
            const moves = legalMoves(s);
            const left = s.boards.reduce((n,b,i) => n + (s.results[i] ? 0 : b.filter(c=>c === null).length),0);
            if (left <= 7) {
              const classes = Array(81).fill('.');
              for (const m of moves) classes[m] = 'LDW'[1 + evalMove(s,m)];
              rows.push([encodeState(s).join(''), classes.join(''), s.turn]);
            }
            random = (Math.imul(random, 1664525)+1013904223)>>>0;
            const m = moves[random % moves.length]; s=play(s,Math.floor(m/9),m%9);
          }
        }
        console.log(JSON.stringify(rows));
        """
        proc = subprocess.run(["node", "--input-type=module", "-e", script], text=True,
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True)
        records = json.loads(proc.stdout)
        self.assertGreater(len(records), 50)
        self.assertEqual({r[2] for r in records}, {"X", "O"})
        ref = np.frombuffer("".join(r[1] for r in records).encode(), np.uint8).reshape(-1, 81)
        self.assertEqual(set(np.unique(ref)), set(b".LDW"))
        states = (np.frombuffer("".join(r[0] for r in records).encode(), np.uint8)
                  - ord("0")).reshape(-1, 280)
        self.assertTrue(states[:, 279].any())
        self.assertTrue(states[:, 270:279].any())
        self.assertTrue(states[:, 261:270].any())
        np.testing.assert_array_equal(eligible(states, 7), np.arange(len(states)))
        for start in range(0, len(states), 200):
            batch = states[start:start+200]
            keep, result, nodes = self.run_solver(batch, 100000, 3)
            np.testing.assert_array_equal(keep, np.arange(len(batch)))
            np.testing.assert_array_equal(result, ref[start:start+200])
            one = self.run_solver(batch, 100000, 1)
            np.testing.assert_array_equal(one[1], result)
            self.assertEqual(one[2], nodes)
        x = states[states[:, 162:243].sum(1) > 1][:12]
        self.assertGreater(len(x), 0)
        keep, _, nodes = self.run_solver(x, 1, 2)
        self.assertEqual((len(keep), nodes), (0, len(x)))

    def test_target_preserves_prior_and_handles_zero_optimal_mass(self):
        c = np.array([list(b"WLDW" + b"." * 77), list(b"DLD." + b"." * 77)], np.uint8)
        p = np.zeros((2, 81), np.float16)
        p[0, :4] = [.6, .1, .1, .2]
        p[1, 1] = 1
        targets, v = proved_targets(c, p, 0.1)
        np.testing.assert_array_equal(v, [1, 0])
        np.testing.assert_allclose(targets[0, :4], [.725, 0, 0, .275], atol=.0003)
        np.testing.assert_allclose(targets[1, :4], [.5, 0, .5, 0], atol=.0003)
