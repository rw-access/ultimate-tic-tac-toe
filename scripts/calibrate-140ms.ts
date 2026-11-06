#!/usr/bin/env node
/** Calibrate exact 140ms single-thread CPU simulation throughput for any ONNX models. */
import { createGame, play } from '../src/game.ts';
import { encodeState, legalMoves, search } from '../src/ai.ts';

const [modelPath, utttPath] = process.argv.slice(2);
if (!modelPath) throw new Error('Usage: calibrate-140ms.ts <model.onnx> [utttai.onnx]');

const { InferenceSession, Tensor } = await import('onnxruntime-node');

async function createEvaluator(path: string, isUtttai: boolean) {
  const session = await InferenceSession.create(path, {
    executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1, graphOptimizationLevel: 'all',
  });
  const evaluate = async (state: any) => {
    if (isUtttai) {
      const features = new Float32Array(324), legal = legalMoves(state);
      for (let move = 0; move < 81; move++) {
        const board = Math.floor(move / 9), cell = move % 9;
        const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
        const col = board % 3 * 3 + cell % 3;
        const flat = row * 9 + col, mark = state.boards[board][cell];
        if (mark === state.turn) features[flat] = 1;
        else if (mark !== null) features[81 + flat] = 1;
        features[162 + flat] = state.turn === 'X' ? 1 : -1;
      }
      for (const move of legal) {
        const board = Math.floor(move / 9), cell = move % 9;
        const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
        const col = board % 3 * 3 + cell % 3;
        features[243 + row * 9 + col] = 1;
      }
      const input = new Tensor('float32', features, [1, 4, 9, 9]);
      const out = await session.run({ input });
      try {
        const logits = out.policy_logits.data as Float32Array;
        const value = Number((out.state_value.data as Float32Array)[0]);
        const policy = new Float32Array(81);
        for (let move = 0; move < 81; move++) {
          const board = Math.floor(move / 9), cell = move % 9;
          const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
          const col = board % 3 * 3 + cell % 3;
          policy[move] = logits[row * 9 + col];
        }
        return { policy, value };
      } finally { input.dispose(); for (const t of Object.values(out)) t.dispose(); }
    } else {
      const input = new Tensor('float32', encodeState(state), [1, 280]);
      const out = await session.run({ state: input });
      try {
        return { policy: Float32Array.from(out.policy.data as Float32Array), value: Number((out.value.data as Float32Array)[0]) };
      } finally { input.dispose(); for (const t of Object.values(out)) t.dispose(); }
    }
  };
  return { evaluate, close: () => session.release() };
}

async function calibrate(name: string, path: string, isUtttai: boolean) {
  const { evaluate, close } = await createEvaluator(path, isUtttai);
  try {
    let state = createGame();
    // Warmup
    await search(state, { simulations: 50, timeMs: 50, evaluate });
    const samples: number[] = [];
    for (let ply = 0; ply < 10; ply++) {
      const res = await search(state, { simulations: 10000, timeMs: 140, evaluate, seed: 100 + ply });
      samples.push(res.simulations);
      const moves = legalMoves(state);
      state = play(state, Math.floor(moves[0] / 9), moves[0] % 9);
    }
    const mean = Math.round(samples.reduce((a, b) => a + b, 0) / samples.length);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    return { name, mean, min, max, samples };
  } finally {
    await close();
  }
}

const cand = await calibrate('Candidate', modelPath, false);
console.log(`[Calibration] ${cand.name} (${modelPath}): Mean = ${cand.mean} sims in 140ms (range: ${cand.min}-${cand.max})`);

if (utttPath) {
  const opp = await calibrate('uttt.ai', utttPath, true);
  console.log(`[Calibration] ${opp.name} (${utttPath}): Mean = ${opp.mean} sims in 140ms (range: ${opp.min}-${opp.max})`);
  console.log(`[Calibration] Ratio (Candidate / uttt.ai): ${(cand.mean / opp.mean).toFixed(2)}x`);
  console.log(`Recommended Tournament Command:`);
  console.log(`scripts/external-arena.ts --model ${modelPath} --external-model ${utttPath} --opponent utttai --model-simulations ${cand.mean} --opponent-simulations ${opp.mean} --parallel 16`);
}
