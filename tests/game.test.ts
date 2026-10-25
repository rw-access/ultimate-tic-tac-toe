import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGame, play, legalBoards, isLegal, result, restoreGame } from '../src/game.ts';

test('opening is free; a cell sends the next player to its board', () => {
  const initial = createGame();
  assert.equal(legalBoards(initial).length, 9);
  const next = play(initial, 4, 2);
  assert.equal(next.turn, 'O');
  assert.deepEqual(legalBoards(next), [2]);
  assert.equal(initial.boards[4][2], null);
  assert.equal(play(next, 3, 0), next);
  assert.equal(isLegal(next, -1, 0), false);
  assert.equal(isLegal(next, 2, 9), false);
});
test('all eight winning lines and drawn mini boards', () => {
  for (const line of [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]) {
    const b = Array(9).fill(null);
    line.forEach(i => b[i] = 'O');
    assert.equal(result(b), 'O');
  }
  assert.equal(result(['X','O','X','X','O','O','O','X','X']), 'draw');
  assert.equal(result(['draw','draw','draw',null,null,null,null,null,null]), null);
});
test('winning a mini board claims it and prevents more moves', () => {
  const s = createGame();
  s.boards[0] = ['X','X',null,'O',null,null,null,null,null];
  s.forced = 0;
  const next = play(s, 0, 2);
  assert.equal(next.results[0], 'X');
  assert.equal(next.forced, 2);
  assert.equal(isLegal({ ...next, forced: null }, 0, 4), false);
});
test('sending to a finished board releases the next player', () => {
  for (const completed of ['X','O','draw'] as const) {
    const s = createGame();
    s.results[2] = completed;
    const next = play(s, 4, 2);
    assert.equal(next.forced, null);
    assert.equal(legalBoards(next).length, 8);
    assert.equal(legalBoards(next).includes(2), false);
  }
});
test('a draw closes a mini board without claiming it', () => {
  const s = createGame();
  s.boards[4] = ['X','O','X','X','O','O','O','X',null];
  const next = play(s, 4, 8);
  assert.equal(next.results[4], 'draw');
  assert.equal(next.winner, null);
});
test('macro win ends the game and macro draw is handled', () => {
  const s = createGame();
  s.results[0] = s.results[1] = 'X';
  s.boards[2] = ['X','X',null,null,null,null,null,null,null];
  const next = play(s, 2, 2);
  assert.equal(next.winner, 'X');
  assert.deepEqual(legalBoards(next), []);
  assert.equal(play(next, 3, 1), next);
  const d = createGame();
  d.results = ['X','O','X','X','O','O','O','X',null];
  d.boards[8] = ['X','O','X','X','O','O','O','X',null];
  assert.equal(play(d, 8, 8).winner, 'draw');
});
test('restore derives outcomes and rejects corrupted saves', () => {
  const s = play(createGame(), 4, 4);
  assert.deepEqual(restoreGame(JSON.parse(JSON.stringify(s))), s);
  assert.equal(restoreGame(null), null);
  assert.equal(restoreGame({ ...s, boards: [] }), null);
  assert.equal(restoreGame({ ...s, turn: 'X' }), null);
  assert.equal(restoreGame({ ...s, forced: -1 }), null);
  assert.equal(restoreGame({ ...s, forced: 1.2 }), null);
  assert.deepEqual(restoreGame({ ...s, results: [] })?.results, s.results);
});
test('many legal full games preserve invariants without getting stuck', () => {
  let seed = 42;
  for (let game = 0; game < 150; game++) {
    let s = createGame();
    while (!s.winner) {
      const choices = legalBoards(s).flatMap(b => s.boards[b].flatMap((c,i) => c === null ? [[b,i]] : []));
      assert.ok(choices.length);
      seed = (Math.imul(seed,1664525) + 1013904223) >>> 0;
      const [b,c] = choices[seed % choices.length];
      const next = play(s,b,c);
      assert.equal(next.moves, s.moves + 1);
      assert.ok(next.moves <= 81);
      assert.deepEqual(next.results, next.boards.map(result));
      s = next;
    }
  }
});
