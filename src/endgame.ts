import { legalMoves } from './ai.ts';
import { play, type GameState } from './game.ts';

export type EndgameResult = {
  solved: boolean;
  move: number | null;
  /** Result from the perspective of the player to move: win=1, draw=0, loss=-1. */
  value: number;
  nodes: number;
};

type Bound = 'exact' | 'lower' | 'upper';
type Entry = { value: number; move: number | null; bound: Bound };

function positionKey(state: GameState): string {
  const marks = state.boards.map(board => board.map(mark => mark ?? '.').join('')).join('');
  const results = state.results.map(result => result ?? '.').join('');
  return `${marks}|${results}|${state.turn}|${state.forced ?? '-'}`;
}

function emptySquares(state: GameState): number {
  return state.boards.reduce((total, board, index) => total + (state.results[index] === null
    ? board.reduce((count, mark) => count + (mark === null ? 1 : 0), 0)
    : 0), 0);
}

function terminalValue(state: GameState): number {
  if (state.winner === 'draw' || state.winner === null) return 0;
  return state.winner === state.turn ? 1 : -1;
}

export function immediateWinningMoves(state: GameState): number[] {
  return legalMoves(state).filter(move =>
    play(state, Math.floor(move / 9), move % 9).winner === state.turn,
  );
}

export function movesSafeFromImmediateLoss(state: GameState): number[] {
  return legalMoves(state).filter(move => {
    const next = play(state, Math.floor(move / 9), move % 9);
    return next.winner !== null || immediateWinningMoves(next).length === 0;
  });
}

/**
 * Solve late positions exactly when their remaining game tree fits a strict
 * node/time budget. An incomplete search is never presented as an exact score.
 */
export function solveEndgame(
  state: GameState,
  options: { maxEmptySquares?: number; maxNodes?: number; timeMs?: number } = {},
): EndgameResult | null {
  const emptyLimit = Math.max(0, options.maxEmptySquares ?? 18);
  if (emptySquares(state) > emptyLimit) return null;
  const rootMoves = legalMoves(state);
  if (!rootMoves.length) return null;

  const maxNodes = Math.max(1, Math.floor(options.maxNodes ?? 32_000));
  const deadline = performance.now() + Math.max(1, options.timeMs ?? 36);
  const table = new Map<string, Entry>();
  let nodes = 0;
  let aborted = false;
  let bestRootMove: number | null = null;

  function orderedMoves(position: GameState, moves: number[], preferred: number | null): number[] {
    const ranked = moves.map(move => {
      const child = play(position, Math.floor(move / 9), move % 9);
      const tactical = child.winner === position.turn ? 100 : child.results[Math.floor(move / 9)] === position.turn ? 10 : 0;
      const center = move % 9 === 4 ? 2 : [0, 2, 6, 8].includes(move % 9) ? 1 : 0;
      return { move, score: (move === preferred ? 1000 : 0) + tactical + center };
    });
    ranked.sort((left, right) => right.score - left.score || left.move - right.move);
    return ranked.map(item => item.move);
  }

  function visit(position: GameState, alphaInput: number, betaInput: number, isRoot = false): number | null {
    if (position.winner !== null) return terminalValue(position);
    nodes++;
    if (nodes > maxNodes || ((nodes & 63) === 0 && performance.now() >= deadline)) {
      aborted = true;
      return null;
    }

    const key = positionKey(position);
    const alphaOriginal = alphaInput;
    const betaOriginal = betaInput;
    const entry = table.get(key);
    let alpha = alphaInput;
    let beta = betaInput;
    if (entry) {
      if (entry.bound === 'exact') return entry.value;
      if (entry.bound === 'lower') alpha = Math.max(alpha, entry.value);
      else beta = Math.min(beta, entry.value);
      if (alpha >= beta) return entry.value;
    }

    const moves = legalMoves(position);
    if (!moves.length) return 0;
    const ordered = orderedMoves(position, moves, entry?.move ?? null);
    let best = -2;
    let bestMove: number | null = null;
    for (const move of ordered) {
      const child = play(position, Math.floor(move / 9), move % 9);
      const reply = visit(child, -beta, -alpha);
      if (reply === null) return null;
      const value = -reply;
      if (value > best) {
        best = value;
        bestMove = move;
      }
      if (value > alpha) alpha = value;
      if (alpha >= beta) break;
    }

    if (isRoot) bestRootMove = bestMove;
    const bound: Bound = best <= alphaOriginal ? 'upper' : best >= betaOriginal ? 'lower' : 'exact';
    const previous = table.get(key);
    if (!previous || bound === 'exact' || previous.bound !== 'exact') {
      table.set(key, { value: best, move: bestMove, bound });
    }
    return best;
  }

  const value = visit(state, -2, 2, true);
  if (aborted || value === null || bestRootMove === null) {
    return { solved: false, move: null, value: 0, nodes };
  }
  return { solved: true, move: bestRootMove, value, nodes };
}
