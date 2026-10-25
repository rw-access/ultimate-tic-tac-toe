import { isLegal, legalBoards, play, type GameState } from './game.ts';

export function immediateWinningMoves(state: GameState): number[] {
  if (state.winner !== null) return [];
  return legalBoards(state).flatMap(board => state.boards[board].flatMap((mark, cell) => {
    if (mark !== null || !isLegal(state, board, cell)) return [];
    return play(state, board, cell).winner === state.turn ? [board * 9 + cell] : [];
  }));
}

export function hasImmediateWin(state: GameState): boolean {
  return immediateWinningMoves(state).length > 0;
}

export function missedImmediateWin(state: GameState, playedMove: number): boolean {
  const winningMoves = immediateWinningMoves(state);
  return winningMoves.length > 0 && !winningMoves.includes(playedMove);
}

export type SacrificeMotif = 'local-board' | 'free-choice';

/** Identify a visible local concession or routing gift in the position after a move. */
export function sacrificeMotif(after: GameState, targetBoard: number): SacrificeMotif | null {
  if (after.winner !== null || !Number.isInteger(targetBoard) || targetBoard < 0 || targetBoard > 8) return null;
  if (after.forced === null) {
    return after.results[targetBoard] !== null && legalBoards(after).length > 1 ? 'free-choice' : null;
  }
  if (after.forced !== targetBoard || after.results[targetBoard] !== null) return null;
  for (let cell = 0; cell < 9; cell += 1) {
    if (after.boards[targetBoard][cell] !== null) continue;
    if (play(after, targetBoard, cell).results[targetBoard] === after.turn) return 'local-board';
  }
  return null;
}
