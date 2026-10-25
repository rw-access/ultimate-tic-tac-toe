export type Player = 'X' | 'O';
export type Result = Player | 'draw' | null;
export interface GameState {
  boards: (Player | null)[][];
  results: Result[];
  turn: Player;
  forced: number | null;
  winner: Result;
  moves: number;
}
const lines = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];
export function result(cells: Result[]): Result {
  for (const [a, b, c] of lines) {
    if ((cells[a] === 'X' || cells[a] === 'O') && cells[a] === cells[b] && cells[a] === cells[c]) return cells[a];
  }
  return cells.every(Boolean) ? 'draw' : null;
}
export function createGame(): GameState {
  return { boards: Array.from({ length: 9 }, () => Array(9).fill(null)), results: Array(9).fill(null), turn: 'X', forced: null, winner: null, moves: 0 };
}
export function legalBoards(state: GameState): number[] {
  if (state.winner) return [];
  if (state.forced !== null && state.results[state.forced] === null) return [state.forced];
  return state.results.flatMap((r, i) => r === null ? [i] : []);
}
export function isLegal(state: GameState, board: number, cell: number): boolean {
  return Number.isInteger(board) && Number.isInteger(cell) && board >= 0 && board < 9 && cell >= 0 && cell < 9 && legalBoards(state).includes(board) && state.boards[board][cell] === null;
}
export function play(state: GameState, board: number, cell: number): GameState {
  if (!isLegal(state, board, cell)) return state;
  const boards = state.boards.map(b => [...b]);
  boards[board][cell] = state.turn;
  const results = [...state.results];
  results[board] = result(boards[board]);
  return { boards, results, turn: state.turn === 'X' ? 'O' : 'X', forced: results[cell] === null ? cell : null, winner: result(results), moves: state.moves + 1 };
}
/** Treat local browser data as untrusted; derive outcomes instead of trusting them. */
export function restoreGame(value: unknown): GameState | null {
  if (!value || typeof value !== 'object') return null;
  const s = value as GameState;
  if (!Array.isArray(s.boards) || s.boards.length !== 9 || !s.boards.every(b => Array.isArray(b) && b.length === 9 && b.every(c => c === null || c === 'X' || c === 'O'))) return null;
  const boards = s.boards.map(b => [...b]);
  const marks = boards.flat();
  const x = marks.filter(c => c === 'X').length;
  const o = marks.filter(c => c === 'O').length;
  if (x !== o && x !== o + 1) return null;
  const turn = x === o ? 'X' : 'O';
  if (s.turn !== turn || (s.forced !== null && (!Number.isInteger(s.forced) || s.forced < 0 || s.forced > 8))) return null;
  const results = boards.map(result);
  return { boards, results, turn, forced: s.forced !== null && results[s.forced] === null ? s.forced : null, winner: result(results), moves: x + o };
}
