import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { encodeState, legalMoves } from '../src/ai.ts';
import { play, type GameState } from '../src/game.ts';
import { InferenceSession, Tensor } from 'onnxruntime-node';

interface ParityCase {
  state: GameState;
  encoded: number[];
  legalMoves: number[];
  next?: { move: number; state: GameState };
  policy: number[];
  value: number;
}

test('bundled checkpoint and numeric training/inference fixtures share one rules contract', () => {
  const fixture = JSON.parse(readFileSync(new URL('../training/parity.json', import.meta.url), 'utf8'));
  const model = readFileSync(new URL('../public/models/policy-value.onnx', import.meta.url));
  assert.equal(createHash('sha256').update(model).digest('hex'), fixture.modelSha256);
  assert.ok(fixture.cases.length >= 12);
  for (const item of fixture.cases as ParityCase[]) {
    assert.deepEqual(Array.from(encodeState(item.state)), item.encoded);
    assert.deepEqual(legalMoves(item.state), item.legalMoves);
    if (item.next) {
      assert.deepEqual(play(item.state, Math.floor(item.next.move / 9), item.next.move % 9), item.next.state);
    }
  }
});

test('native ONNX inference matches the bundled model fixtures', async () => {
  const fixture = JSON.parse(readFileSync(new URL('../training/parity.json', import.meta.url), 'utf8'));
  const session = await InferenceSession.create(
    readFileSync(new URL('../public/models/policy-value.onnx', import.meta.url)),
    { executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1 },
  );
  try {
    for (const item of fixture.cases as ParityCase[]) {
      const input = new Tensor('float32', encodeState(item.state), [1, 280]);
      const output = await session.run({ state: input });
      try {
        assert.deepEqual(output.policy.dims, [1, 81]);
        assert.deepEqual(output.value.dims, [1, 1]);
        for (let action = 0; action < 81; action++) {
          assert.ok(Math.abs(Number(output.policy.data[action]) - item.policy[action]) < 0.0001);
        }
        assert.ok(Math.abs(Number(output.value.data[0]) - item.value) < 0.0001);
      } finally {
        input.dispose();
        for (const tensor of Object.values(output)) tensor.dispose();
      }
    }
  } finally {
    await session.release();
  }
});
