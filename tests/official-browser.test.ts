import assert from 'node:assert/strict';
import test from 'node:test';
import { playGame } from '../scripts/official-uttt-arena.ts';
import { legalMoves } from '../src/ai.ts';
import { createGame, play, type GameState } from '../src/game.ts';

class TestGame {
  state = createGame();
  getLegalIndexes() { return legalMoves(this.state); }
  getLegalActions() { return this.getLegalIndexes().map(index => ({ index })); }
  get nextSymbol() { return this.state.turn === 'X' ? 1 : 2; }
  get result() { return [null, 'X', 'O', 'draw'].indexOf(this.state.winner); }
  execute(action: {index: number}) { this.state = play(this.state, Math.floor(action.index / 9), action.index % 9); }
  clone() { return Object.assign(new TestGame(), { state: this.state }); }
}

class TestEngine {
  tree: { root: { visitCount: number; childNodes: {action: {index: number}; visitCount: number}[] }; synchronize: (game: TestGame) => void };
  constructor(game: TestGame) {
    const root = { visitCount: 0, childNodes: game.getLegalActions().map(action => ({ action, visitCount: 0 })) };
    this.tree = { root, synchronize(game) {
      root.visitCount = 0;
      root.childNodes = game.getLegalActions().map(action => ({ action, visitCount: 0 }));
    } };
  }
  async simulate() {
    this.tree.root.visitCount++;
    this.tree.root.childNodes[0].visitCount++;
  }
}

test('timed official matchup resets per game, scores only candidate turns and preserves the opponent cap', async () => {
  let resets = 0;
  const browser = {
    async beginGame() { resets++; },
    async move(state: GameState) {
      return { move: legalMoves(state)[0], simulations: 31, elapsedMs: 219.5, source: 'model' as const };
    },
  };
  const load = () => ({ Game: TestGame, Engine: TestEngine });
  const opponent = { async predict() { throw new Error('This fixture simulates visits without inference'); } };
  const opening = [40];
  for (const seat of ['X', 'O'] as const) {
    const result = await playGame(opening, seat, 7, 384, 13, true, load, undefined, opponent, undefined, browser);
    const ourMoves = result.moves.slice(seat === 'O' ? 1 : 2).filter((_, i) => i % 2 === 0);
    const observed = result.browserTurns!;
    assert.deepEqual(observed.map(t => t.move), ourMoves);
    assert.ok(observed.every(t => t.simulations === 31 && t.elapsedMs === 219.5));
    assert.equal(result.ourVisits, 31 * observed.length);
    assert.equal(result.theirVisits, 13 * (result.moves.length - opening.length - observed.length));
    assert.equal(result.ourReuse, null);
    assert.equal(result.retainedEndgameMoves, null);
  }
  assert.equal(resets, 2);
});
