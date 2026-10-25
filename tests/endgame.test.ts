import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGame } from '../src/game.ts';
import { immediateWinningMoves, solveEndgame } from '../src/endgame.ts';

test('bounded endgame search proves an immediate forced win', () => {
  const state = createGame();
  state.turn = 'O';
  state.results = ['O', 'O', 'X', 'X', null, 'O', 'X', 'O', null];
  state.boards[0] = ['O', 'O', 'O', 'X', 'X', null, null, null, null];
  state.boards[1] = ['O', 'O', 'O', 'X', 'X', null, null, null, null];
  state.boards[2] = ['X', 'X', 'X', 'O', 'O', null, null, null, null];
  state.boards[3] = ['X', 'X', 'X', 'O', 'O', null, null, null, null];
  state.boards[4] = [null, null, null, 'O', 'X', 'X', 'O', 'X', 'O'];
  state.boards[5] = ['O', 'O', 'O', 'X', null, null, null, null, null];
  state.boards[6] = ['X', 'X', 'X', 'O', null, null, null, null, null];
  state.boards[7] = ['O', 'O', 'O', 'X', null, null, null, null, null];
  state.boards[8] = ['X', null, 'O', 'X', null, 'X', 'O', 'X', 'X'];
  state.forced = 4;
  assert.deepEqual(immediateWinningMoves(state), [36]);
  const result = solveEndgame(state, { timeMs: 500, maxNodes: 100_000 });
  assert.ok(result?.solved);
  assert.equal(result.value, 1);
  assert.equal(result.move, 36); // Board 5, square 1 completes O's macro line.
});

test('positions outside the small-tree threshold are left to ordinary search', () => {
  assert.equal(solveEndgame(createGame()), null);
});
