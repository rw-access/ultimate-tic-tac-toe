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

type ForcedWinProof = 'win' | 'refuted' | 'unknown';

export type ForcedWinWithinPliesResult = {
  move: number;
  nodes: number;
};

/**
 * Return a move only when the player to move can prove a win within the ply
 * horizon. Budget exhaustion and horizon leaves remain unknown, never wins.
 */
export function findForcedWinWithinPlies(
  state: GameState,
  options: {
    /** Individual moves; seven plies includes four moves by the attacker. */
    maxPlies?: number;
    maxNodes?: number;
    timeMs?: number;
    rootMoveScores?: Float32Array;
  } = {},
): ForcedWinWithinPliesResult | null {
  const attacker = state.turn;
  const maxPlies = Math.max(1, Math.min(9, Math.floor(options.maxPlies ?? 7)));
  const maxNodes = Math.max(1, Math.floor(options.maxNodes ?? 12_000));
  const deadline = performance.now() + Math.max(1, options.timeMs ?? 24);
  const table = new Map<string, ForcedWinProof>();
  let nodes = 0;
  let exhausted = false;

  let rootMoves = legalMoves(state);
  if (!rootMoves.length) return null;
  const rootScores = options.rootMoveScores;
  if (rootScores) {
    const score = (move: number) => {
      const value = rootScores[move];
      return Number.isFinite(value) ? value : -Infinity;
    };
    rootMoves = rootMoves.sort((left, right) => {
      const leftScore = score(left);
      const rightScore = score(right);
      return rightScore > leftScore ? 1 : rightScore < leftScore ? -1 : left - right;
    });
  }

  function withinBudget(): boolean {
    if (nodes >= maxNodes || performance.now() >= deadline) {
      exhausted = true;
      return false;
    }
    return true;
  }

  function orderedMoves(position: GameState): number[] {
    return legalMoves(position).map(move => {
      const board = Math.floor(move / 9);
      const child = play(position, board, move % 9);
      const score = (child.winner === position.turn ? 10_000 : 0)
        + (child.results[board] === position.turn ? 100 : 0)
        + (move % 9 === 4 ? 2 : 0);
      return { move, score };
    }).sort((left, right) => right.score - left.score || left.move - right.move)
      .map(item => item.move);
  }

  function prove(position: GameState, remainingPlies: number): ForcedWinProof {
    if (position.winner !== null) return position.winner === attacker ? 'win' : 'refuted';
    if (remainingPlies <= 0 || !withinBudget()) return 'unknown';

    nodes++;
    const key = `${positionKey(position)}|${remainingPlies}|${attacker}`;
    const cached = table.get(key);
    if (cached) return cached;

    const moves = orderedMoves(position);
    if (!moves.length) return 'refuted';

    let sawUnknown = false;
    if (position.turn === attacker) {
      for (const move of moves) {
        const child = play(position, Math.floor(move / 9), move % 9);
        const result = prove(child, remainingPlies - 1);
        if (result === 'win') {
          table.set(key, 'win');
          return 'win';
        }
        if (result === 'unknown') sawUnknown = true;
        if (exhausted) break;
      }
      const result = sawUnknown || exhausted ? 'unknown' : 'refuted';
      if (result !== 'unknown') table.set(key, result);
      return result;
    }

    for (const move of moves) {
      const child = play(position, Math.floor(move / 9), move % 9);
      const result = prove(child, remainingPlies - 1);
      if (result === 'refuted') {
        table.set(key, 'refuted');
        return 'refuted';
      }
      if (result === 'unknown') sawUnknown = true;
      if (exhausted) break;
    }
    const result = sawUnknown || exhausted ? 'unknown' : 'win';
    if (result !== 'unknown') table.set(key, result);
    return result;
  }

  for (const move of rootMoves) {
    if (!withinBudget()) break;
    const child = play(state, Math.floor(move / 9), move % 9);
    if (child.winner === attacker) return { move, nodes };
    if (prove(child, maxPlies - 1) === 'win') return { move, nodes };
    if (exhausted) break;
  }
  return null;
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
