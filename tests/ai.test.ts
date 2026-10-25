import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { baselineEvaluation, encodeState, INPUT_SIZE, legalMoves, search } from '../src/ai.ts';
import { createGame, play } from '../src/game.ts';
import type { GameState } from '../src/game.ts';

test('state encoder marks perspective, legal squares, mini-board results, and forced board', () => {
  const state = play(createGame(), 4, 2);
  const input = encodeState(state);
  assert.equal(input.length, INPUT_SIZE);
  assert.equal(input[81 + 4 * 9 + 2], 1); // X is now the opponent of O.
  assert.equal(input[4 * 9 + 2], 0);
  assert.equal(input[162 + 2 * 9], 1);
  assert.equal(input[162 + 2 * 9 + 8], 1);
  assert.equal(input[162 + 3 * 9], 0);
  assert.equal(input[272], 1); // forced board 2
  assert.equal(input[279], 0);

  const opening = encodeState(createGame());
  assert.equal(opening[279], 1); // null forced-board bucket
  assert.equal(opening.slice(162, 243).reduce((sum, x) => sum + x, 0), 81);

  const statuses = createGame();
  statuses.turn = 'O';
  statuses.results = ['O', 'X', 'draw', null, null, null, null, null, null];
  const statusInput = encodeState(statuses);
  assert.equal(statusInput[243], 1);
  assert.equal(statusInput[252 + 1], 1);
  assert.equal(statusInput[261 + 2], 1);
});

test('legal moves are flattened board*9+cell actions and obey forced play', () => {
  const state = play(createGame(), 4, 2);
  const moves = legalMoves(state);
  assert.equal(moves.length, 9);
  assert.deepEqual(moves, Array.from({ length: 9 }, (_, cell) => 18 + cell));
  assert.deepEqual(legalMoves(play(state, 2, 5)), Array.from({ length: 9 }, (_, cell) => 45 + cell));
});

test('search takes an immediate global win even when the evaluator ranks it last', async () => {
  const state = createGame();
  state.turn = 'X';
  state.results[0] = state.results[1] = 'X';
  state.boards[2] = ['X', 'X', null, null, null, null, null, null, null];
  state.forced = 2;
  const result = await search(state, {
    simulations: 1,
    tacticalProof: true,
    rootNoise: { epsilon: 1, concentrationMass: 2.7 },
    evaluate: async () => ({ policy: new Float32Array(81).fill(-100), value: -1 }),
  });
  assert.equal(result.move, 20);
  assert.equal(result.policy[20], 1);
  assert.equal(result.simulations, 0);
});

test('negamax backup treats evaluator values as current-player perspective', async () => {
  const state = createGame();
  state.forced = 0;
  state.boards[0] = ['X', 'O', 'X', 'X', 'O', 'O', 'O', null, null];
  const result = await search(state, {
    simulations: 80,
    timeMs: 10_000,
    seed: 17,
    evaluate: async (position: GameState) => {
      const policy = new Float32Array(81);
      // For the child (O to move), +1 favors the root's move 7 for O, while
      // -1 says the root's move 8 is favorable to X.
      const value = position.turn === 'O' && position.boards[0][7] === 'X' ? 1 :
        position.turn === 'O' && position.boards[0][8] === 'X' ? -1 : 0;
      return { policy, value };
    },
  });
  assert.equal(result.move, 8);
  assert.ok(result.policy[8] > result.policy[7]);
});

test('fixed seed and simulation budget produce a reproducible visit policy', async () => {
  const state = play(createGame(), 4, 4);
  const options = { simulations: 60, timeMs: 10_000, seed: 2026 };
  const first = await search(state, options);
  const second = await search(state, { ...options, tacticalProof: false });
  assert.equal(first.move, second.move);
  assert.deepEqual(Array.from(first.policy), Array.from(second.policy));
  assert.equal(first.simulations, 60);
  assert.equal(second.simulations, 60);
  assert.ok(Math.abs(first.policy.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
});

test('root noise defaults off and epsilon zero preserves the legacy search stream', async () => {
  const state = play(createGame(), 4, 4);
  const options = {
    simulations: 60,
    timeMs: 10_000,
    seed: 2026,
    evaluate: async (position: GameState) => {
      const policy = new Float32Array(81);
      for (const move of legalMoves(position)) policy[move] = 0;
      return { policy, value: 0 };
    },
  };
  const legacy = await search(state, options);
  const disabled = await search(state, {
    ...options,
    rootNoise: { epsilon: 0, concentrationMass: 2.7 },
  });
  assert.equal(legacy.move, 37);
  assert.equal(legacy.simulations, 60);
  assert.equal(createHash('sha256').update(Buffer.from(legacy.policy.buffer)).digest('hex'),
    'b8111392ce8afe95ca6a841e7d57f2bf3426af11002610937070f54f946c43d6');
  assert.deepEqual(Array.from(disabled.policy), Array.from(legacy.policy));
  assert.equal(disabled.move, legacy.move);
});

test('root Dirichlet noise is seeded, legal-only, normalized, and changes root priors', async () => {
  const state = play(createGame(), 4, 4);
  const evaluator = async () => ({ policy: new Float32Array(81), value: 0 });
  const clean = await search(state, { simulations: 1, seed: 89, evaluate: evaluator });
  const options = {
    simulations: 1,
    seed: 89,
    evaluate: evaluator,
    rootNoise: { epsilon: 0.25, concentrationMass: 2.7 },
  };
  const first = await search(state, options);
  const repeated = await search(state, options);
  assert.deepEqual(Array.from(first.policy), Array.from(repeated.policy));
  assert.notDeepEqual(Array.from(first.policy), Array.from(clean.policy));
  assert.ok(Math.abs(first.policy.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
  for (let move = 0; move < 81; move++) {
    if (!legalMoves(state).includes(move)) assert.equal(first.policy[move], 0);
  }
});

test('baseline evaluator returns finite current-player values and legal-only priors', () => {
  const state = play(createGame(), 4, 4);
  const result = baselineEvaluation(state);
  const legal = new Set(legalMoves(state));
  assert.ok(Number.isFinite(result.value));
  assert.ok(result.value >= -1 && result.value <= 1);
  for (let move = 0; move < 81; move++) {
    if (!legal.has(move)) assert.equal(result.policy[move], 0);
    else assert.ok(Number.isFinite(result.policy[move]));
  }
});

test('terminal games never request inference or return a move', async () => {
  for (const winner of ['X', 'O', 'draw'] as const) {
    const state = { ...createGame(), winner };
    const result = await search(state, { evaluate: async () => { throw new Error('Unexpected inference'); } });
    assert.equal(result.move, null);
    assert.equal(result.simulations, 0);
    assert.equal(result.policy.reduce((sum, value) => sum + value, 0), 0);
    assert.equal(encodeState(state).slice(162, 243).some(Boolean), false);
  }
});

test('search masks illegal logits and tolerates invalid network scores', async () => {
  const state = play(createGame(), 4, 2);
  const policy = new Float32Array(81).fill(Infinity);
  policy[0] = 1e20; // This square is outside the forced board.
  const result = await search(state, {
    simulations: 12, timeMs: 10_000,
    evaluate: async () => ({ policy, value: NaN }),
  });
  assert.ok(legalMoves(state).includes(result.move!));
  assert.ok(result.policy.every(Number.isFinite));
  for (let action = 0; action < 81; action++) {
    if (!legalMoves(state).includes(action)) assert.equal(result.policy[action], 0);
  }
});

test('search looks ahead to avoid sending the opponent to a global winning board', async () => {
  const state = createGame();
  state.results[0] = state.results[1] = 'O';
  state.boards[2] = ['O', 'O', null, null, null, null, null, null, null];
  state.boards[4] = ['X', 'O', null, 'O', 'X', null, 'O', 'X', 'O'];
  state.forced = 4;
  const policy = new Float32Array(81);
  policy[38] = 8; // Misleading preference sends O to its winning board 2.
  const result = await search(state, {
    simulations: 100, timeMs: 10_000,
    evaluate: async () => ({ policy, value: 0 }),
  });
  assert.equal(result.move, 41); // Sending O to board 5 avoids the immediate loss.
});

const reachableThreatPath = [
  63, 8, 75, 27, 1, 15, 62, 79, 69, 54, 0, 2, 18, 5, 49, 44,
  78, 61, 64, 11, 20, 24, 60, 56, 26, 77, 53, 72, 67, 40, 39,
  34, 71, 73, 10, 13, 37, 22, 41, 50, 48, 32, 47, 23, 46, 57,
];

function createReachableGlobalThreat(): GameState {
  return reachableThreatPath.reduce(
    (state, action) => play(state, Math.floor(action / 9), action % 9),
    createGame(),
  );
}

function adversarialEvaluator(root: GameState, unsafe: number, safe: number) {
  return async (position: GameState) => {
    const moves = legalMoves(position);
    const policy = new Float32Array(81).fill(-100);
    for (const move of moves) policy[move] = 0;
    if (position === root) {
      policy[unsafe] = 10;
      policy[safe] = -20;
    } else {
      // Deliberately hide any immediate macro win from the network's prior.
      for (const move of moves) {
        if (play(position, Math.floor(move / 9), move % 9).winner === position.turn) {
          policy[move] = -1000;
        }
      }
    }
    return { policy, value: 0 };
  };
}

test('optional tactical proof avoids a low-prior immediate global loss', async () => {
  const state = createReachableGlobalThreat();
  assert.equal(state.moves, 46);
  assert.equal(state.turn, 'X');
  assert.equal(state.forced, 3);
  const unsafe = 29; // X sends O to board 2, where O can win globally at action 21.
  const safe = 30;
  const unsafeState = play(state, Math.floor(unsafe / 9), unsafe % 9);
  const safeState = play(state, Math.floor(safe / 9), safe % 9);
  assert.ok(legalMoves(unsafeState).some(move =>
    play(unsafeState, Math.floor(move / 9), move % 9).winner === 'O'));
  assert.equal(legalMoves(safeState).some(move =>
    play(safeState, Math.floor(move / 9), move % 9).winner === 'O'), false);

  const evaluate = adversarialEvaluator(state, unsafe, safe);
  const legacy = await search(state, { simulations: 384, timeMs: 10_000, seed: 11, evaluate });
  assert.equal(legacy.move, unsafe);

  const proven = await search(state, {
    simulations: 384, timeMs: 10_000, seed: 11, evaluate, tacticalProof: true,
  });
  assert.notEqual(proven.move, unsafe);
  assert.ok([30, 31, 33, 35].includes(proven.move!));
  assert.equal(proven.policy[unsafe], 0);
  assert.ok(Math.abs(proven.policy.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
});

test('tactical proof keeps legal moves when every move permits an immediate loss', async () => {
  const state = createReachableGlobalThreat();
  // Leave only destinations 2 (the winning board) and 7 (closed, so play is
  // unrestricted) available in the forced board. Both allow O to win next.
  state.boards[3] = ['O', 'X', null, 'X', 'O', 'O', 'O', null, 'X'];
  const moves = legalMoves(state);
  assert.deepEqual(moves, [29, 34]);

  let inferenceCalls = 0;
  const result = await search(state, {
    simulations: 2,
    timeMs: 10_000,
    tacticalProof: true,
    evaluate: async position => {
      inferenceCalls++;
      const policy = new Float32Array(81).fill(-100);
      for (const move of legalMoves(position)) policy[move] = 0;
      if (position === state) policy[29] = 10;
      return { policy, value: 0 };
    },
  });
  assert.ok(moves.includes(result.move!));
  assert.equal(result.simulations, 2);
  // The root is evaluated once; its proven winning-reply leaf bypasses inference.
  assert.equal(inferenceCalls, 1);
});
