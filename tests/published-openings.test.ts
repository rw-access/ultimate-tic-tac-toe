import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { publishedSuite, recoverPublishedOpening } from '../scripts/official-uttt-arena.ts';

const cases: [string, number[]][] = [
  ['100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200', [0]],
  ['000000000000000000000000000000000000000010002000000000000000000000000000000010000000000000240', [40, 44, 76]],
  ['000000000000000000000000000000000000000010002000000000000000100000000000000000201000000000260', [40, 44, 80, 78, 60]],
  ['002000000000000000100000100000000000000000001000000000000002000000000000002000000000000000150', [44, 74, 18, 2, 24, 59]],
];

test('reconstructs published states at every tournament depth in deterministic move order', () => {
  for (const [target, moves] of cases) {
    assert.deepEqual(recoverPublishedOpening(target), moves);
    assert.deepEqual(recoverPublishedOpening(target), moves);
  }
});

test('checks turn, constraint, subboard outcomes and global outcome in the full state', () => {
  const [target] = cases[0];
  for (const [index, value] of [[90, '1'], [91, '9'], [81, '1'], [92, '2']] as const) {
    const invalid = target.slice(0, index) + value + target.slice(index + 1);
    assert.throws(() => recoverPublishedOpening(invalid), /no legal project history/);
  }
  assert.throws(() => recoverPublishedOpening(target + '0'), /Invalid published state encoding/);
});

test('rejects an edited upstream file before extracting its states', async () => {
  const root = await mkdtemp(join(tmpdir(), 'uttt-published-test-'));
  try {
    const dir = join(root, 'utttpy/selfplay');
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'evaluation_uttt_states.py'), '    "' + cases[0][0] + '",\n');
    await assert.rejects(publishedSuite(root), /hash mismatch/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
