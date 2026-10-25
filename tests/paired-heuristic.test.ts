import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { baselineEvaluation, legalMoves, search } from '../src/ai.ts';
import { createGame, play } from '../src/game.ts';

const root = resolve(import.meta.dirname, '..');
const parent = 'public/models/policy-value.onnx';
const challenger = parent;
function run(script: string, model: string, opponent: string, output: string, extra: string[] = []) {
  const child = spawnSync(process.execPath, ['--experimental-strip-types', script, 'arena', '--challenger', model, '--champion', opponent, '--pairs', '4', '--simulations', '8', '--seed', '928841', '--parallel-games', '4', '--output', output, ...extra], { cwd: root, encoding: 'utf8', timeout: 120_000 });
  assert.ifError(child.error); assert.equal(child.status, 0, child.stderr);
}

test('explicit heuristic evaluator exactly matches default search across reachable positions', async () => {
  let state = createGame();
  for (let ply = 0; ply < 65; ply++) {
    if (state.winner) break;
    if (ply % 3 === 0) {
      const before = JSON.stringify(state);
      const options = { simulations: 32, timeMs: 30_000, seed: 1000 + ply };
      const implicit = await search(state, options);
      const explicit = await search(state, { ...options, evaluate: async position => baselineEvaluation(position) });
      assert.equal(explicit.move, implicit.move); assert.deepEqual(explicit.policy, implicit.policy);
      assert.equal(explicit.simulations, implicit.simulations); assert.equal(JSON.stringify(state), before);
    }
    const legal = legalMoves(state); const action = legal[(ply * 17 + 7) % legal.length];
    state = play(state, Math.floor(action / 9), action % 9);
  }
});

test('paired heuristic adapter preserves ordinary model-versus-model trajectories', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paired-heuristic-'));
  try {
    run('scripts/ai.ts', challenger, parent, join(directory, 'original.json'));
    run('scripts/ai-paired-heuristic.ts', challenger, parent, join(directory, 'adapted.json'));
    const a = JSON.parse(await readFile(join(directory, 'original.json'), 'utf8'));
    const b = JSON.parse(await readFile(join(directory, 'adapted.json'), 'utf8'));
    assert.deepEqual(a.perPair, b.perPair); assert.deepEqual(a.pairScores, b.pairScores);
    assert.deepEqual(a.challenger, b.challenger); assert.deepEqual(a.pairedConfidence95, b.pairedConfidence95);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('heuristic guards share openings across different challengers and use no opponent network', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'paired-heuristic-cases-'));
  try {
    run('scripts/ai-paired-heuristic.ts', parent, 'heuristic', join(directory, 'parent.json'));
    run('scripts/ai-paired-heuristic.ts', challenger, 'heuristic', join(directory, 'candidate.json'), ['--challenger-value', 'zero']);
    const docs = await Promise.all(['parent', 'candidate'].map(async name => JSON.parse(await readFile(join(directory, `${name}.json`), 'utf8'))));
    const openings = (doc: any) => doc.perPair.map((p: any) => [p.pair, p.openingSeed, p.openingMoves]);
    assert.deepEqual(openings(docs[0]), openings(docs[1]));
    assert.equal(docs[0].challengerValueMode, 'model');
    assert.equal(docs[1].challengerValueMode, 'zero');
    for (const doc of docs) {
      assert.equal(doc.championKind, 'heuristic'); assert.equal(doc.championModel.sha256, null);
      assert.equal(doc.championValueMode, 'heuristic-policy-and-value');
      assert.equal(doc.inference.champion.inferenceCalls, 0); assert.equal(doc.pairs, 4); assert.equal(doc.games, 8);
      assert.equal(doc.tacticalProof, false);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
