import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGame, play, type GameState } from '../src/game.ts';
import { immediateWinningMoves, solveEndgame } from '../src/endgame.ts';

function bruteValue(state: GameState, memo = new Map<string, number>()): number {
  if (state.winner === 'draw') return 0;
  if (state.winner !== null) return state.winner === state.turn ? 1 : -1;
  const key = `${state.boards.map(board => board.map(mark => mark ?? '.').join('')).join('')}|${state.turn}|${state.forced}`;
  const cached = memo.get(key);
  if (cached !== undefined) return cached;
  const moves = state.boards.flatMap((board, index) => (state.forced === null || state.results[state.forced] !== null || state.forced === index)
    && state.results[index] === null
    ? board.flatMap((mark, cell) => mark === null ? [index * 9 + cell] : [])
    : []);
  if (!moves.length) return 0;
  let best = -1;
  for (const move of moves) {
    best = Math.max(best, -bruteValue(play(state, Math.floor(move / 9), move % 9), memo));
    if (best === 1) break;
  }
  memo.set(key, best);
  return best;
}

function drawBeforeWinPosition(): GameState {
  const state = createGame();
  state.boards = [
    ['O', 'X', null, 'O', 'X', 'O', 'X', 'X', 'O'],
    ['X', null, 'O', 'X', 'X', null, null, 'X', 'X'],
    ['X', 'O', 'X', null, 'O', 'X', null, 'O', 'O'],
    ['X', 'O', 'O', 'X', 'O', 'O', null, null, null],
    ['O', 'X', 'O', 'O', 'X', null, 'O', 'O', 'X'],
    ['X', null, 'X', null, null, 'O', 'X', 'X', 'X'],
    ['O', 'O', null, 'X', 'X', null, 'O', 'X', 'X'],
    ['O', 'O', 'X', 'X', 'O', 'O', 'O', null, 'X'],
    [null, 'O', 'X', null, 'X', 'O', 'X', 'O', 'O'],
  ];
  state.results = ['X', 'X', 'O', null, 'O', 'X', null, null, 'X'];
  state.turn = 'O';
  state.forced = 3;
  state.winner = null;
  state.moves = 63;
  return state;
}

function replay(moves: number[]): GameState {
  return moves.reduce((state, move) => play(state, Math.floor(move / 9), move % 9), createGame());
}

const retainedWinMoves = [
  80, 78, 60, 58, 40, 36, 0, 2, 20, 22, 37, 13, 43, 67,
  19, 10, 16, 70, 64, 11, 18, 6, 61, 66, 34, 68, 48, 30,
  28, 9, 8, 72, 4, 75, 31, 62, 54, 57, 50, 49, 59, 45,
];

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

  assert.equal(result.guaranteedValue, 1);
  assert.equal(result.matePlies, 1);
  assert.equal(result.shortestWinProven, true);
});

test('positions outside the small-tree threshold are left to ordinary search', () => {
  assert.equal(solveEndgame(createGame()), null);
});

test('an aborted search keeps a proved draw without claiming the root is solved', () => {
  const state = drawBeforeWinPosition();
  const memo = new Map<string, number>();
  const drawnMove = play(state, 3, 6); // Global move 33; first move searched.
  const laterWinningMove = play(state, 3, 7); // Global move 34; available but searched later.
  assert.equal(bruteValue(drawnMove, memo), 0);
  assert.equal(-bruteValue(laterWinningMove, memo), 1);

  const result = solveEndgame(state, { maxNodes: 6, timeMs: 2_000 });
  assert.equal(result?.solved, false);
  assert.equal(result?.budgetExhausted, true);
  assert.equal(result?.move, 33);
  assert.equal(result?.value, 0);
  assert.equal(result?.guaranteedValue, 0);
});

test('a partial root bound is not exposed as a proved draw', () => {
  const result = solveEndgame(drawBeforeWinPosition(), { maxNodes: 1, timeMs: 2_000 });
  assert.equal(result?.solved, false);
  assert.equal(result?.move, null);
  assert.equal(result?.guaranteedValue, null);
});

test('completed endgame search prefers the forced win over an earlier draw', () => {
  const state = drawBeforeWinPosition();
  const result = solveEndgame(state, { maxNodes: 100_000, timeMs: 2_000 });
  assert.equal(result?.solved, true);
  assert.equal(result?.move, 34);
  assert.equal(result?.value, 1);
  assert.equal(result?.guaranteedValue, 1);
});

test('a proved win is retained while the search runs out before proving it shortest', () => {
  const result = solveEndgame(replay(retainedWinMoves), { maxNodes: 64, timeMs: 10_000 });
  assert.equal(result?.solved, true);
  assert.equal(result?.value, 1);
  assert.equal(result?.move, 51);
  assert.equal(result?.matePlies, 5);
  assert.equal(result?.shortestWinProven, false);
  assert.equal(result?.budgetExhausted, true);
});

test('a partial search without a completed winning proof stays unknown', () => {
  const result = solveEndgame(replay(retainedWinMoves), { maxNodes: 1, timeMs: 10_000 });
  assert.equal(result?.solved, false);
  assert.equal(result?.move, null);
  assert.equal(result?.guaranteedValue, null);
  assert.equal(result?.matePlies, null);
  assert.equal(result?.budgetExhausted, true);
});

test('exact mate search chooses a three-ply win over a five-ply win', () => {
  const state = replay([
    65, 23, 49, 40, 41, 50, 52, 64, 9, 0, 1, 16, 67, 44, 79, 63,
    4, 36, 7, 68, 46, 17, 80, 73, 15, 62, 78, 60, 61, 66, 28, 12,
    34, 69, 57, 33, 58, 10, 13, 30, 31, 20, 24, 56, 26, 19,
  ]);
  const result = solveEndgame(state, { maxNodes: 100_000, timeMs: 10_000 });
  assert.equal(result?.solved, true);
  assert.equal(result?.value, 1);
  assert.equal(result?.move, 11);
  assert.equal(result?.matePlies, 3);
  assert.equal(result?.shortestWinProven, true);
});
