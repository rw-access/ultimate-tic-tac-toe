import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGame } from '../src/game.ts';
import { hasImmediateWin, immediateWinningMoves, missedImmediateWin, sacrificeMotif } from '../src/review.ts';

test('review accepts any of multiple immediate winning moves', () => {
  const state = createGame();
  state.results[0] = state.results[1] = state.results[3] = state.results[4] = 'X';
  state.boards[2] = ['X', 'X', null, null, null, null, null, null, null];
  state.boards[5] = ['X', 'X', null, null, null, null, null, null, null];
  const winningMoves = immediateWinningMoves(state);

  assert.deepEqual(winningMoves, [20, 47]);
  assert.equal(hasImmediateWin(state), true);
  assert.equal(missedImmediateWin(state, winningMoves[0]), false);
  assert.equal(missedImmediateWin(state, winningMoves[1]), false);
  assert.equal(missedImmediateWin(state, 22), true);
});

test('review recognizes immediate local-board concessions and free-choice routing gifts', () => {
  const localBoardGift = createGame();
  localBoardGift.turn = 'O';
  localBoardGift.forced = 4;
  localBoardGift.boards[4] = ['O', 'O', null, null, null, null, null, null, null];
  assert.equal(sacrificeMotif(localBoardGift, 4), 'local-board');

  const freeChoiceGift = createGame();
  freeChoiceGift.turn = 'O';
  freeChoiceGift.results[4] = 'draw';
  assert.equal(sacrificeMotif(freeChoiceGift, 4), 'free-choice');

  const onlyOneChoice = createGame();
  onlyOneChoice.turn = 'O';
  onlyOneChoice.results = Array(9).fill('draw');
  onlyOneChoice.results[8] = null;
  assert.equal(sacrificeMotif(onlyOneChoice, 4), null);
});
