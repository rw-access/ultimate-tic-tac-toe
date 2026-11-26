import assert from 'node:assert/strict';
import { test } from 'node:test';
import { pairSeedGenerator, parseOpeningSchedule } from '../scripts/arena-openings.ts';

test('independent pair seeds are domain-separated SHA-256 vectors indexed without a cursor', () => {
  const fail = () => { throw new Error('Should not read the legacy stream'); };
  const seed = pairSeedGenerator('independent', 2026112602, fail);
  assert.equal(seed(7), 2232889396);
  assert.equal(seed(0), 3771934597);
  assert.equal(seed(1), 785342151);
  assert.equal(seed(7), 2232889396);
  assert.equal(pairSeedGenerator('independent', 0, fail)(0), 3942041168);
  assert.equal(pairSeedGenerator('independent', 12, fail)(345), 3555655948);
  assert.equal(pairSeedGenerator('independent', 123, fail)(45), 712886187);
});

test('omitted and explicit legacy settings forward exactly one draw per pair', () => {
  for (const value of [undefined, 'legacy']) {
    let next = 41;
    const seed = pairSeedGenerator(parseOpeningSchedule(value), 0, () => ++next);
    assert.deepEqual([seed(0), seed(1), seed(2)], [42, 43, 44]);
  }
  assert.equal(parseOpeningSchedule('independent'), 'independent');
  assert.throws(() => parseOpeningSchedule('random'), /opening-schedule/);
});

// This rejects a genuine but incorrect ONNX before loading a session or opening a match.
test('official arena rejects a substituted opponent before writing evidence', async () => {
  const { spawnSync } = await import('node:child_process');
  const { mkdtemp, readdir, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { resolve, join } = await import('node:path');
  const directory = await mkdtemp(join(tmpdir(), 'official-opponent-digest-'));
  try {
    const model = resolve('public/models/policy-value.onnx');
    const child = spawnSync(process.execPath, [
      resolve('scripts/official-uttt-arena.ts'), '--model', model,
      '--official-model', model, '--upstream', directory, '--output', join(directory, 'result.json'),
    ], { encoding: 'utf8', timeout: 10_000 });
    assert.ifError(child.error);
    assert.equal(child.status, 1);
    assert.match(child.stderr, /Unrecognized official uttt.ai Stage 2 model SHA-256/);
    assert.deepEqual(await readdir(directory), []);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
