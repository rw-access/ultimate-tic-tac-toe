import './style.css';
import {
  createGame,
  isLegal,
  legalBoards,
  play,
  restoreGame,
  type GameState,
  type Player,
  type Result,
} from './game';
import { ComputerPlayer, PositionAnalyzer, type Difficulty, type PositionAnalysis } from './ai-client';
import { hasImmediateWin, missedImmediateWin, sacrificeMotif } from './review';
import { LEGACY_STORAGE_KEYS, migrateLegacyStorage, readMigratedStorage, STORAGE_KEYS } from './storage';
import { OnlineRoomConnection, createOnlineRoom as requestOnlineRoom, joinOnlineRoom as requestJoinOnlineRoom, roomInviteUrl, type OnlineConnectionStatus, type OnlineSeat } from './online-client';
import { isRoomCode, normalizeRoomCode, type PublicRoom } from './multiplayer/core';

const STORAGE_KEY = STORAGE_KEYS.game;
const OPPONENT_KEY = STORAGE_KEYS.opponent;
const THEME_KEY = STORAGE_KEYS.theme;
const REVIEW_PLAYBACK_SPEEDS = [0.5, 1, 2] as const;
const app = document.querySelector<HTMLDivElement>('#app');

if (!app) throw new Error('The game root element is missing.');
migrateLegacyStorage();

type ThemePreference = 'auto' | 'light' | 'dark';
type OpponentMode = 'two-players' | 'computer' | 'online';
type SideChoice = Player | 'random';
type AnalysisView = 'position' | null;
type ReviewMove = { before: GameState; after: GameState; move: number; number: number; player: Player };
type ReviewItem = { move: ReviewMove; best: number | null; line: number[]; note: string; kind: 'tactical' | 'model' };
type ReviewEvaluation = { xValue: number; source: 'model' | 'search'; solved: boolean; lines: number[][] };
type ReviewTimeline = { snapshots: GameState[]; moves: ReviewMove[] };
type MoveGrade = { label: string; kind: string; delta: number; loss: number; tactical: boolean; brilliant: boolean };
type CriticalMove = MoveGrade & { moveIndex: number; move: ReviewMove; magnitude: number };
const DIFFICULTIES: Difficulty[] = ['easy', 'medium', 'hard'];

function difficultyLabel(value: Difficulty): string {
  return value[0].toUpperCase() + value.slice(1);
}

const computer = new ComputerPlayer();
const analyzer = new PositionAnalyzer();
const reviewDetailAnalyzer = new PositionAnalyzer();

let state = createGame();
let undoStack: GameState[] = [];
let computerMoves = new Set<number>();
let animatingBoards = new Set<number>();
let opponentMode: OpponentMode = 'two-players';
let difficulty: Difficulty = 'medium';
let sideChoice: SideChoice = 'X';
let draftOpponentMode: OpponentMode = 'two-players';
let draftDifficulty: Difficulty = 'medium';
let draftSideChoice: SideChoice = 'X';
let humanSide: Player = 'X';
let hintMove: number | null = null;
let hintPending = false;
let hintFailed = false;
let hintToken = 0;
let analysisView: AnalysisView = null;
let positionAnalysis: PositionAnalysis | null = null;
let positionAnalysisMoves = -1;
let analysisPending = false;
let analysisFailed = false;
let analysisToken = 0;
let reviewActive = false;
let reviewFinalState: GameState | null = null;
let reviewSnapshots: GameState[] = [];
let reviewMoves: ReviewMove[] = [];
let reviewItems: ReviewItem[] = [];
let reviewEvaluations: (ReviewEvaluation | null)[] = [];
let reviewCursor = 0;
let reviewPlaying = false;
let reviewPlaybackSpeedIndex = 1;
let reviewPlaybackTimer: number | null = null;
let reviewPending = false;
let reviewUnavailable = 0;
let reviewHistoryAvailable = true;
let reviewDetailAnalysis: PositionAnalysis | null = null;
let reviewDetailPosition = -1;
let reviewDetailPending = false;
let reviewDetailFailed = false;
let reviewDetailToken = 0;
let computerThinking = false;
let computerStatus = '';
let computerSource: 'model' | 'search' | '' = '';
let computerFailed = false;
let computerRequestToken = 0;
let draftRoomCode = '';
let onlineSeat: OnlineSeat | null = null;
let onlineConnection: OnlineRoomConnection | null = null;
let onlineConnectionStatus: OnlineConnectionStatus = 'closed';
let onlineSetupError = '';
let onlineError = '';
let onlineNotice = '';
let onlineBusy = false;
let onlineMovePending = false;
let onlineVersion = 0;

const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
let themePreference: ThemePreference = 'auto';
try {
  const saved = readMigratedStorage(THEME_KEY, LEGACY_STORAGE_KEYS.theme);
  if (saved === 'light' || saved === 'dark') themePreference = saved;
} catch {
  // Follow the system when browser preferences are unavailable.
}

function applyTheme(): void {
  document.documentElement.dataset.theme = themePreference === 'auto'
    ? systemTheme.matches ? 'dark' : 'light'
    : themePreference;
}

function themeButtonMarkup(): string {
  const label = themePreference[0].toUpperCase() + themePreference.slice(1);
  const next = themePreference === 'auto' ? 'Light' : themePreference === 'light' ? 'Dark' : 'Auto';
  const icon = themePreference === 'dark'
    ? '<path d="M12.5 10.5A5.5 5.5 0 0 1 5.5 3.5a5.5 5.5 0 1 0 7 7Z" />'
    : themePreference === 'light'
      ? '<circle cx="8" cy="8" r="3" /><path d="M8 1v1m0 12v1M1 8h1m12 0h1M3 3l1 1m8 8 1 1M3 13l1-1m8-8 1-1" />'
      : '<rect x="1.5" y="2.5" width="13" height="9" rx="1.5" /><path d="M5 14h6M8 11.5V14" />';
  return `<button class="theme-button" type="button" data-action="theme" aria-label="Theme: ${label}. Switch to ${next}" title="Theme: ${label} · click for ${next}"><svg viewBox="0 0 16 16" aria-hidden="true">${icon}</svg><span>${label}</span></button>`;
}

function cycleTheme(): void {
  themePreference = themePreference === 'auto' ? 'light' : themePreference === 'light' ? 'dark' : 'auto';
  try {
    if (themePreference === 'auto') localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, themePreference);
  } catch {
    // Theme changes still work for this session when storage is unavailable.
  }
  applyTheme();
  render();
}

applyTheme();
systemTheme.addEventListener('change', () => {
  if (themePreference === 'auto') applyTheme();
});

function readOpponentPreference(): void {
  try {
    const raw = readMigratedStorage(OPPONENT_KEY, LEGACY_STORAGE_KEYS.opponent);
    if (!raw) return;
    const saved: unknown = JSON.parse(raw);
    if (!saved || typeof saved !== 'object') return;
    const preference = saved as { mode?: unknown; difficulty?: unknown; sideChoice?: unknown };
    if (preference.mode === 'two-players' || preference.mode === 'computer') opponentMode = preference.mode;
    if (preference.sideChoice === 'X' || preference.sideChoice === 'O' || preference.sideChoice === 'random') {
      sideChoice = preference.sideChoice;
    }
    if (preference.difficulty === 'easy' || preference.difficulty === 'medium' || preference.difficulty === 'hard') {
      difficulty = preference.difficulty;
    }
  } catch {
    // Invalid or unavailable preferences should leave the default opponent selected.
  }
}

function persistOpponentPreference(): void {
  try {
    localStorage.setItem(OPPONENT_KEY, JSON.stringify({ mode: opponentMode === 'online' ? 'two-players' : opponentMode, difficulty, sideChoice }));
  } catch {
    // The selected opponent still works for this session when storage is unavailable.
  }
}

function snapshot(game: GameState): GameState {
  return JSON.parse(JSON.stringify(game)) as GameState;
}

function loadSavedGame(): void {
  try {
    const raw = readMigratedStorage(STORAGE_KEY, LEGACY_STORAGE_KEYS.game);
    if (!raw) return;
    const saved: unknown = JSON.parse(raw);
    if (!saved || typeof saved !== 'object' || !('state' in saved)) return;

    const record = saved as { version?: unknown; state?: unknown; undo?: unknown; computerMoves?: unknown; humanSide?: unknown };
    if (record.version !== 1) return;
    const restored = restoreGame(record.state);
    if (!restored) return;

    state = restored;
    if (record.humanSide === 'X' || record.humanSide === 'O') humanSide = record.humanSide;
    if (Array.isArray(record.undo)) {
      undoStack = record.undo
        .slice(-100)
        .map((item) => restoreGame(item))
        .filter((item): item is GameState => item !== null && item.moves < state.moves);
    }
    if (Array.isArray(record.computerMoves)) {
      computerMoves = new Set(
        record.computerMoves.filter((move): move is number =>
          Number.isInteger(move) && (move as number) >= 1 && (move as number) <= state.moves,
        ),
      );
    }
  } catch {
    // A blocked, malformed, or outdated local save should never prevent a new game.
    state = createGame();
    undoStack = [];
    computerMoves.clear();
  }
}

function persist(): void {
  if (reviewActive) return;
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 1, state, humanSide, undo: undoStack.slice(-100), computerMoves: [...computerMoves].sort((a, b) => a - b) }),
    );
  } catch {
    // Keep the current game playable when browser storage is unavailable.
  }
}

function countBoards(player: Player): number {
  return state.results.filter((result) => result === player).length;
}

function markerSvg(player: Player): string {
  if (player === 'X') {
    return '<svg viewBox="0 0 100 100" aria-hidden="true"><path d="M24 24 76 76M76 24 24 76" /></svg>';
  }
  return '<svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="34" /></svg>';
}

function resultOverlay(result: Result): string {
  if (!result) return '';
  if (result === 'draw') {
    return '<div class="result-overlay result-draw" aria-label="Board drawn"><span>—</span></div>';
  }
  return `<div class="result-overlay result-${result.toLowerCase()}" aria-label="Board won by ${result}">${markerSvg(result)}</div>`;
}

function sideChoiceLabel(choice: SideChoice): string {
  return choice === 'random' ? 'Random' : choice;
}

function advantageLabel(value: number, solved = false): string {
  if (solved) {
    if (Math.abs(value) < 0.01) return 'Draw with perfect play';
    return `${value > 0 ? 'X' : 'O'} can force a win`;
  }
  if (Math.abs(value) < 0.08) return 'Even';
  const side = value > 0 ? 'X' : 'O';
  const magnitude = Math.abs(value);
  return `${side} ${magnitude > 0.62 ? 'strong ' : magnitude > 0.3 ? '' : 'slight '}edge`;
}

function lineMarkup(line: number[]): string {
  return line.map((move) => `B${Math.floor(move / 9) + 1}·${move % 9 + 1}`).join(' <span aria-hidden="true">›</span> ');
}

function positionAnalysisMarkup(): string {
  if (analysisPending) {
    return '<section class="analysis-panel" aria-label="Position analysis"><span class="analysis-note" role="status">Reading position…</span></section>';
  }
  if (analysisFailed || !positionAnalysis) {
    return '<section class="analysis-panel" aria-label="Position analysis"><span class="analysis-note" role="status">Analysis unavailable</span></section>';
  }

  const value = positionAnalysis.value * (state.turn === 'X' ? 1 : -1);
  const label = advantageLabel(value, positionAnalysis.solved);
  const numericValue = Math.round(-value * 100);
  const explanation = positionAnalysis.solved
    ? `${label}. Proven by exact search.`
    : `${label}. Model estimate; not win odds.`;
  const lines = positionAnalysis.lines.slice(0, 2);
  const source = positionAnalysis.solved ? 'Exact search' : positionAnalysis.source === 'model' ? 'Model estimate' : 'Search estimate';
  return `<section class="analysis-panel" aria-label="Position analysis" data-position-moves="${positionAnalysisMoves}">
    <div class="analysis-heading"><strong>${label}</strong><span class="analysis-note">${source}${positionAnalysis.solved ? '' : ' · not win odds'}</span></div>
    <div class="advantage-row">
      <span class="advantage-side advantage-x">X</span>
      <div class="advantage-track" role="meter" aria-label="Position advantage" aria-valuemin="-100" aria-valuemax="100" aria-valuenow="${numericValue}" aria-valuetext="${explanation}">
        <span class="advantage-center"></span><span class="advantage-marker" style="left:${50 - value * 44}%"></span>
      </div>
      <span class="advantage-side advantage-o">O</span>
    </div>
    ${lines.length ? `<div class="candidate-lines"><div class="candidate-line"><span>Best</span><code>${lineMarkup(lines[0])}</code></div></div>${lines.length > 1 ? `<details class="alternative-lines"><summary>Alternative line</summary><div class="candidate-line"><span>Alt</span><code>${lineMarkup(lines[1])}</code></div></details>` : ''}` : '<span class="analysis-note">No line found</span>'}
  </section>`;
}

function boardMoveName(move: number): string {
  return `B${Math.floor(move / 9) + 1}·${move % 9 + 1}`;
}

function buildReviewTimeline(): ReviewTimeline {
  const positions = [...undoStack, snapshot(state)];
  if (positions.length < 2) return { snapshots: [], moves: [] };
  const snapshots = [snapshot(positions[0])];
  const moves: ReviewMove[] = [];
  for (let index = 0; index + 1 < positions.length; index += 1) {
    const before = positions[index];
    const after = positions[index + 1];
    if (after.moves !== before.moves + 1) return { snapshots: [], moves: [] };
    const changes: number[] = [];
    for (let board = 0; board < 9; board += 1) {
      for (let cell = 0; cell < 9; cell += 1) {
        if (before.boards[board][cell] === null && after.boards[board][cell] === before.turn) {
          changes.push(board * 9 + cell);
        }
      }
    }
    if (changes.length !== 1) return { snapshots: [], moves: [] };
    moves.push({ before, after, move: changes[0], number: after.moves, player: before.turn });
    snapshots.push(snapshot(after));
  }
  return { snapshots, moves };
}

function reviewPositionLabel(index = reviewCursor): string {
  if (index === 0) return 'Starting position';
  const move = reviewMoves[index - 1];
  return move ? `Move ${move.number} · ${move.player} ${boardMoveName(move.move)}` : `Position ${index}`;
}

function reviewPositionValue(index: number): number | null {
  const evaluation = reviewEvaluations[index];
  return evaluation?.xValue ?? null;
}

function reviewPositionScore(evaluation: ReviewEvaluation | null | undefined): string {
  if (!evaluation || !Number.isFinite(evaluation.xValue)) return '—';
  const score = Math.round(evaluation.xValue * 100);
  return score > 0 ? `+${score}` : String(score);
}

function reviewPositionButtonLabel(index: number): string {
  const label = reviewPositionLabel(index);
  const value = reviewPositionValue(index);
  return `${label}${value === null ? ', evaluation pending' : `, ${advantageLabel(value, reviewEvaluations[index]?.solved)}`}`;
}

function reviewBrilliantMotif(move: ReviewMove): string | null {
  const motif = sacrificeMotif(move.after, move.move % 9);
  return motif === 'free-choice' ? 'granting free board choice'
    : motif === 'local-board' ? 'allowing an immediate local-board win' : null;
}

function reviewGradeIcon(kind: string): string {
  const shapes: Record<string, string> = {
    brilliant: '<path d="m8 1.5 1.8 4 4.4.6-3.2 3.1.8 4.3L8 11.4l-3.8 2.1.8-4.3-3.2-3.1 4.4-.6z"/><path d="M13 1v2m1-1h-2"/>',
    best: '<path d="m8 1.5 1.8 4 4.4.6-3.2 3.1.8 4.3L8 11.4l-3.8 2.1.8-4.3-3.2-3.1 4.4-.6z"/>',
    'top-pick': '<path d="m8 1.5 1.8 4 4.4.6-3.2 3.1.8 4.3L8 11.4l-3.8 2.1.8-4.3-3.2-3.1 4.4-.6z"/><path d="M5 14.5h6"/>',
    good: '<circle cx="8" cy="8" r="6"/><path d="m5 8 2 2 4-4"/>',
    steady: '<circle cx="8" cy="8" r="5.5"/><circle cx="8" cy="8" r="1"/>',
    inaccuracy: '<circle cx="8" cy="8" r="6"/><path d="M8 4.5v4m0 2.5h.01"/>',
    mistake: '<path d="M8 1.7 15 14H1z"/><path d="M8 5v4m0 2h.01"/>',
    blunder: '<circle cx="8" cy="8" r="6"/><path d="m5.5 5.5 5 5m0-5-5 5"/>',
    tactical: '<path d="m9 1.5-6 8h4l-1 5 6-8H8z"/>',
  };
  return `<svg viewBox="0 0 16 16" aria-hidden="true">${shapes[kind] ?? shapes.steady}</svg>`;
}

function reviewNavigationIcon(kind: 'start' | 'previous' | 'next' | 'end'): string {
  const shapes = {
    start: '<path d="M4 2.5v11M11 4.5 7.5 8l3.5 3.5"/>',
    previous: '<path d="m10.5 3.5-4.5 4.5 4.5 4.5"/>',
    next: '<path d="m5.5 3.5 4.5 4.5-4.5 4.5"/>',
    end: '<path d="M12 2.5v11M5 4.5 8.5 8 5 11.5"/>',
  };
  return `<svg viewBox="0 0 16 16" aria-hidden="true">${shapes[kind]}</svg>`;
}

function reviewPlaybackIcon(kind: 'play' | 'pause'): string {
  const shape = kind === 'play'
    ? '<path d="m5 3.5 7 4.5-7 4.5z" />'
    : '<path d="M5.5 3.5v9m5-9v9" />';
  return `<svg viewBox="0 0 16 16" aria-hidden="true">${shape}</svg>`;
}

function reviewPlaybackSpeedLabel(): string {
  return `${REVIEW_PLAYBACK_SPEEDS[reviewPlaybackSpeedIndex]}×`;
}

function reviewMoveGrade(moveIndex: number): MoveGrade | null {
  const move = reviewMoves[moveIndex];
  const before = reviewEvaluations[moveIndex];
  const after = reviewEvaluations[moveIndex + 1];
  if (!move || !before || !after) return null;
  const delta = after.xValue - before.xValue;
  const moverSwing = move.player === 'X' ? delta : -delta;
  const loss = Math.max(0, -moverSwing);
  const tactical = reviewItems.find((item) => item.move.number === move.number && item.kind === 'tactical');
  if (tactical) {
    const missedWin = tactical.note.startsWith('Missed');
    return { label: missedWin ? 'Missed win' : 'Allowed win', kind: 'tactical', delta, loss, tactical: true, brilliant: false };
  }
  const bestMove = reviewDetailPosition === moveIndex && reviewDetailAnalysis?.lines[0]?.[0] !== undefined
    ? reviewDetailAnalysis.lines[0][0]
    : before.lines[0]?.[0];
  const provenWin = before.solved && (move.player === 'X' ? before.xValue : -before.xValue) >= 0.999;
  if (provenWin && before.lines[0]?.[0] === move.move && reviewBrilliantMotif(move)) {
    return { label: 'Brilliant', kind: 'brilliant', delta, loss, tactical: false, brilliant: true };
  }
  if (bestMove === move.move) {
    const grade = loss >= 0.13 ? { label: 'Top pick', kind: 'top-pick' } : { label: 'Best', kind: 'best' };
    return { ...grade, delta, loss, tactical: false, brilliant: false };
  }
  if (loss >= 0.45) return { label: 'Blunder', kind: 'blunder', delta, loss, tactical: false, brilliant: false };
  if (loss >= 0.28) return { label: 'Mistake', kind: 'mistake', delta, loss, tactical: false, brilliant: false };
  if (loss >= 0.13) return { label: 'Inaccuracy', kind: 'inaccuracy', delta, loss, tactical: false, brilliant: false };
  if (moverSwing >= 0.08) return { label: 'Good', kind: 'good', delta, loss, tactical: false, brilliant: false };
  return { label: 'OK', kind: 'steady', delta, loss, tactical: false, brilliant: false };
}

function reviewCriticalMoves(): CriticalMove[] {
  return reviewMoves.flatMap((move, moveIndex) => {
    const grade = reviewMoveGrade(moveIndex);
    if (!grade || (opponentMode === 'computer' && move.player !== humanSide)) return [];
    if (!grade.tactical && !grade.brilliant && grade.loss < 0.13) return [];
    return [{ ...grade, moveIndex, move, magnitude: Math.abs(grade.delta) }];
  }).sort((left, right) => Number(right.tactical || right.brilliant) - Number(left.tactical || left.brilliant) || right.loss - left.loss || right.magnitude - left.magnitude).slice(0, 5);
}

function reviewCriticalMarkup(): string {
  const critical = reviewCriticalMoves();
  const heading = '<div class="review-section-heading"><strong>Critical moves</strong></div>';
  const disclaimer = '<p class="review-disclaimer">Eval after played move · + favors X · - favors O</p>';
  if (!critical.length) {
    const incomplete = reviewEvaluations.some((evaluation) => evaluation === null);
    const status = !reviewHistoryAvailable ? 'Saved move history is unavailable.'
      : reviewPending ? 'Finding critical moments…'
        : incomplete && reviewUnavailable ? `Incomplete · ${reviewUnavailable} positions could not be evaluated.`
          : incomplete ? 'Finding critical moments…'
            : 'No major evaluation drops found.';
    return `<section class="review-critical" aria-label="Critical moves">${heading}${disclaimer}<p>${status}</p></section>`;
  }
  const rows = critical.map((item) => {
    const best = reviewDetailPosition === item.moveIndex && reviewDetailAnalysis?.lines[0]?.[0] !== undefined
      ? reviewDetailAnalysis.lines[0][0]
      : reviewEvaluations[item.moveIndex]?.lines[0]?.[0];
    const playedName = boardMoveName(item.move.move);
    const bestName = best === undefined ? '—' : boardMoveName(best);
    const positionScore = reviewPositionScore(reviewEvaluations[item.moveIndex + 1]);
    const comparison = best === item.move.move
      ? `<span class="review-move-choice is-played is-best is-best-match"><span>Played · Best</span><strong>${playedName}</strong></span>`
      : `<span class="review-move-choice is-played"><span>Played</span><strong>${playedName}</strong></span><span class="review-move-choice is-best"><span>Best</span><strong>${bestName}</strong></span>`;
    return `<button class="review-critical-move" type="button" data-review-position="${item.moveIndex}" aria-label="Review ${item.label.toLowerCase()}, move ${item.move.number}, ${item.move.player} played ${playedName}, best ${bestName}, eval after played move ${positionScore}">
      <span class="review-critical-heading"><span class="review-grade is-${item.kind}">${item.label}</span><strong>Move ${item.move.number} · ${item.move.player}</strong></span>
      <span class="review-move-comparison">${comparison}</span>
      <span class="review-score">${positionScore}</span>
    </button>`;
  }).join('');
  return `<section class="review-critical" aria-label="Critical moves">${heading}${disclaimer}<div class="review-critical-list">${rows}</div></section>`;
}

function reviewDetailMarkup(): string {
  const position = reviewSnapshots[reviewCursor];
  const baseAnalysis = reviewEvaluations[reviewCursor];
  const detailIsCurrent = reviewDetailPosition === reviewCursor;
  const detailAnalysis = detailIsCurrent ? reviewDetailAnalysis : null;
  const analysis = detailAnalysis?.lines.length ? detailAnalysis : baseAnalysis;
  const lines = analysis?.lines ?? [];
  const source = analysis?.solved ? 'Exact search' : analysis?.source === 'model' ? 'Model estimate' : 'Search estimate';
  if (position?.winner) {
    return `<section class="review-lines" aria-label="Best move and lines"><div class="review-section-heading"><strong>Best move &amp; lines</strong><span>Game complete</span></div><p>There is no move to play from this position.</p></section>`;
  }
  const message = reviewDetailPending && detailIsCurrent
    ? 'Finding the best move and alternate lines…'
    : reviewDetailFailed && detailIsCurrent
      ? 'Deeper analysis unavailable; showing the quick suggestion when available.'
      : '';
  if (!lines.length) {
    return `<section class="review-lines" aria-label="Best move and lines"><div class="review-section-heading"><strong>Best move &amp; lines</strong><span>${message ? 'Searching' : source}</span></div><p role="status">${message || 'No line available for this position.'}</p></section>`;
  }
  return `<section class="review-lines" aria-label="Best move and lines">
    <div class="review-section-heading"><strong>Best move &amp; lines</strong><span>${source}</span></div>
    ${message ? `<p class="review-detail-status" role="status">${message}</p>` : ''}
    <div class="review-candidate-lines">${lines.slice(0, 3).map((line, index) => `<div class="review-line"><span>${index === 0 ? 'Best' : `Alt ${index}`}</span><code>${lineMarkup(line)}</code></div>`).join('')}</div>
  </section>`;
}

function reviewGraphMarkup(): string {
  const width = 600;
  const height = 156;
  const paddingX = 20;
  const paddingY = 14;
  const values = reviewEvaluations;
  const cursorX = values.length <= 1 ? width / 2 : paddingX + (reviewCursor / (values.length - 1)) * (width - paddingX * 2);
  const futureX = Math.min(width - paddingX, cursorX + 6);
  const futureWidth = Math.max(0, width - paddingX - futureX);
  const pointAt = (index: number, evaluation: ReviewEvaluation) => ({
    x: values.length <= 1 ? width / 2 : paddingX + (index / (values.length - 1)) * (width - paddingX * 2),
    y: paddingY + (1 - (evaluation.xValue + 1) / 2) * (height - paddingY * 2),
  });
  const segments: string[] = [];
  let segment: string[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const evaluation = values[index];
    if (!evaluation) {
      if (segment.length) segments.push(segment.join(' '));
      segment = [];
      continue;
    }
    const point = pointAt(index, evaluation);
    segment.push(`${segment.length ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`);
  }
  if (segment.length) segments.push(segment.join(' '));
  const points = values.map((evaluation, index) => {
    if (!evaluation) return '';
    const point = pointAt(index, evaluation);
    return `<circle class="review-graph-point${index === reviewCursor ? ' is-current' : ''}" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="5" data-review-position="${index}" tabindex="0" role="button" aria-label="${reviewPositionButtonLabel(index)}" />`;
  }).join('');
  const evenY = (height / 2).toFixed(1);
  return `<svg class="review-graph" viewBox="0 0 ${width} ${height}" role="group" aria-label="Evaluation graph, X advantage above even and O advantage below even">
    <line class="review-graph-grid" x1="${paddingX}" y1="${paddingY}" x2="${width - paddingX}" y2="${paddingY}" />
    <line class="review-graph-even" x1="${paddingX}" y1="${evenY}" x2="${width - paddingX}" y2="${evenY}" />
    <line class="review-graph-grid" x1="${paddingX}" y1="${height - paddingY}" x2="${width - paddingX}" y2="${height - paddingY}" />
    ${segments.map((path) => `<path class="review-graph-line" d="${path}" />`).join('')}
    ${futureWidth ? `<rect class="review-graph-future" x="${futureX.toFixed(1)}" y="${paddingY}" width="${futureWidth.toFixed(1)}" height="${height - paddingY * 2}" />` : ''}
    <line class="review-graph-cursor" x1="${cursorX.toFixed(1)}" y1="${paddingY}" x2="${cursorX.toFixed(1)}" y2="${height - paddingY}" />
    ${points}
  </svg>`;
}

function reviewMeterMarkup(): string {
  const position = reviewSnapshots[reviewCursor];
  const value = reviewPositionValue(reviewCursor);
  if (value === null) {
    return `<div class="review-evaluation is-pending"><strong>${reviewPending ? 'Analyzing position…' : 'Evaluation unavailable'}</strong><div class="advantage-row"><span class="advantage-side advantage-x">X</span><div class="advantage-track" aria-hidden="true"><span class="advantage-center"></span></div><span class="advantage-side advantage-o">O</span></div></div>`;
  }
  const solved = reviewEvaluations[reviewCursor]?.solved ?? false;
  const terminalText = position?.winner === 'draw' ? 'Game drawn.' : position?.winner ? `${position.winner} won the game.` : '';
  const label = terminalText || advantageLabel(value, solved);
  const explanation = terminalText || (solved ? `${label}. Exact search.` : `${label}. Model estimate; not win odds.`);
  const source = terminalText || (solved ? 'Exact search' : reviewEvaluations[reviewCursor]?.source === 'model' ? 'Model estimate · not win odds' : 'Search estimate · not win odds');
  return `<div class="review-evaluation"><div class="analysis-heading"><strong>${label}</strong><span class="analysis-note">${source}</span></div>
    <div class="advantage-row"><span class="advantage-side advantage-x">X</span>
      <div class="advantage-track" role="meter" aria-label="Position advantage" aria-valuemin="-100" aria-valuemax="100" aria-valuenow="${Math.round(-value * 100)}" aria-valuetext="${explanation}"><span class="advantage-center"></span><span class="advantage-marker" style="left:${50 - value * 44}%"></span></div>
      <span class="advantage-side advantage-o">O</span></div></div>`;
}

function reviewMoveChipMarkup(moveIndex: number): string {
  const move = reviewMoves[moveIndex];
  if (!move) return '';
  const position = moveIndex + 1;
  const grade = reviewMoveGrade(moveIndex);
  return `<button class="review-move-chip${reviewCursor === position ? ' is-current' : ''}${position > reviewCursor ? ' is-future' : ''}${grade?.tactical ? ' has-mistake' : ''}" type="button" data-player="${move.player}" data-review-position="${position}" aria-label="${reviewPositionButtonLabel(position)}${grade ? `, ${grade.label}` : ''}"${reviewCursor === position ? ' aria-current="step"' : ''}>
    <strong>${move.player} ${boardMoveName(move.move)}</strong>${grade ? `<span class="review-grade review-grade-icon is-${grade.kind}" title="${grade.label}">${reviewGradeIcon(grade.kind)}</span>` : ''}
  </button>`;
}

function reviewMoveStripMarkup(): string {
  const rows: string[] = [];
  for (let index = 0; index < reviewMoves.length; index += 2) {
    const turn = Math.floor(index / 2) + 1;
    rows.push(`<div class="review-move-pair" role="group" aria-label="Turn ${turn}"><span class="review-turn-number" aria-hidden="true">${turn}</span>${reviewMoveChipMarkup(index)}${reviewMoveChipMarkup(index + 1)}</div>`);
  }
  return `<div class="review-move-strip" role="group" aria-label="Select a move to view that position"><div class="review-move-list">${rows.join('')}</div></div>`;
}

function gameNotationText(): string {
  const result = reviewFinalState?.winner === 'draw' ? 'Draw'
    : reviewFinalState?.winner ? `${reviewFinalState.winner} won` : 'Unknown';
  const moves = reviewMoves.map((move, index) => index % 2 === 0
    ? `${Math.floor(index / 2) + 1}. ${move.player} ${boardMoveName(move.move)}`
    : `${move.player} ${boardMoveName(move.move)}`);
  const movePairs: string[] = [];
  for (let index = 0; index < moves.length; index += 2) {
    movePairs.push([moves[index], moves[index + 1]].filter(Boolean).join(' '));
  }
  return [
    'Ultimate Tic-Tac-Toe',
    `Result: ${result} · ${reviewFinalState?.moves ?? reviewMoves.length} moves`,
    'Coordinates: B<board>·<square>; board and square are numbered 1–9 row-major from the top-left.',
    '',
    ...movePairs,
    '',
  ].join('\n');
}

function downloadGameNotation(): void {
  if (!reviewActive || !reviewHistoryAvailable || !reviewMoves.length || !reviewFinalState) return;
  const file = new Blob([gameNotationText()], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'ultimate-tic-tac-toe-game.txt';
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function reviewPanelMarkup(): string {
  if (!reviewActive) return '';
  const total = Math.max(0, reviewSnapshots.length - 1);
  const evaluated = reviewEvaluations.filter(Boolean).length;
  const status = !reviewHistoryAvailable
    ? 'Move history is unavailable for this saved game.'
    : reviewPending
      ? `Evaluated ${evaluated} of ${reviewSnapshots.length} positions…`
      : reviewUnavailable ? `${reviewUnavailable} positions could not be evaluated; you can still review the game.`
        : `${reviewMoves.length} moves reviewed.`;
  const canExportNotation = reviewHistoryAvailable && reviewMoves.length > 0;
  return `<section class="review-panel" id="review-panel" aria-label="Game review">
    <div class="review-heading"><div><h2>Game review</h2><p class="review-current-position">${reviewPositionLabel()}</p><p class="review-outcome">${reviewFinalState?.winner === 'draw' ? 'Draw' : `${reviewFinalState?.winner ?? '?'} won`} · ${reviewFinalState?.moves ?? 0} moves</p></div>
      <button class="analysis-button review-export" type="button" data-action="review-export" aria-label="${canExportNotation ? 'Export game notation' : 'Full move history unavailable'}" title="${canExportNotation ? 'Export game notation' : 'Full move history unavailable'}"${canExportNotation ? '' : ' disabled'}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5v8m-3-3 3 3 3-3M2 10.5v4h12v-4"/></svg><span>Export</span></button>
    </div>
    ${reviewCriticalMarkup()}
    ${reviewDetailMarkup()}
    ${reviewMeterMarkup()}
    <div class="review-chart-heading"><strong>Evaluation over time</strong></div>
    <div class="review-chart"><div class="review-chart-scale" aria-hidden="true"><span>X</span><span>Even</span><span>O</span></div><div class="review-graph-wrap">${reviewGraphMarkup()}<div class="review-chart-endpoints"><span>Start</span><span>Move ${reviewMoves.length}</span></div></div></div>
    <div class="review-navigation">
      <div class="review-step-buttons" role="group" aria-label="Replay and move navigation">
        <button type="button" data-action="review-start" aria-label="Go to starting position" title="First position"${reviewCursor === 0 ? ' disabled' : ''}>${reviewNavigationIcon('start')}</button>
        <button type="button" data-action="review-previous" aria-label="Previous move" title="Previous move"${reviewCursor === 0 ? ' disabled' : ''}>${reviewNavigationIcon('previous')}</button>
        <button type="button" data-action="review-playback" aria-label="${reviewPlaying ? 'Pause replay' : 'Play replay'}" title="${reviewPlaying ? 'Pause replay' : 'Play replay'}" aria-pressed="${reviewPlaying}"${total === 0 ? ' disabled' : ''}>${reviewPlaybackIcon(reviewPlaying ? 'pause' : 'play')}</button>
        <span>${reviewCursor} / ${total}</span>
        <button type="button" data-action="review-next" aria-label="Next move" title="Next move"${reviewCursor >= total ? ' disabled' : ''}>${reviewNavigationIcon('next')}</button>
        <button type="button" data-action="review-end" aria-label="Go to final position" title="Final position"${reviewCursor >= total ? ' disabled' : ''}>${reviewNavigationIcon('end')}</button>
      </div>
      <div class="review-navigation-heading"><label for="review-position-slider">Scrub through game</label><button class="review-playback-speed" type="button" data-action="review-speed" aria-label="Playback speed ${reviewPlaybackSpeedLabel()}; click to change" title="Change replay speed">${reviewPlaybackSpeedLabel()}</button></div>
      <input id="review-position-slider" type="range" min="0" max="${total}" step="1" value="${reviewCursor}" data-action="review-position" aria-label="Review move" aria-valuetext="${reviewPositionLabel()}" />
    </div>
    ${reviewMoveStripMarkup()}
    <p class="review-summary" role="status">${status}</p>
    ${reviewItems.length ? `<details class="review-findings"><summary>${reviewItems.length} move notes</summary><ol class="review-list">${reviewItems.map((item) => `<li class="review-item ${item.kind === 'tactical' ? 'review-tactical' : ''}"><strong>Move ${item.move.number} · ${item.move.player} · ${item.kind === 'tactical' ? 'Tactical' : 'Model note'}</strong><span>Played ${boardMoveName(item.move.move)}${item.best === null ? '' : ` · suggested ${boardMoveName(item.best)}`}</span><span class="review-note">${item.note}</span></li>`).join('')}</ol></details>` : ''}
  </section>`;
}

function analysisDialogMarkup(): string {
  if (!analysisView) return '';
  return `<dialog class="modal analysis-modal" id="analysis-dialog" aria-labelledby="analysis-title">
    <form method="dialog" class="modal-card analysis-card">
      <div class="analysis-modal-heading"><h2 id="analysis-title">Position analysis</h2><button class="analysis-close" id="analysis-close" type="submit" aria-label="Close analysis">×</button></div>
      <div class="analysis-modal-content">${positionAnalysisMarkup()}</div>
      <button class="primary-button" id="analysis-done" type="submit">Done</button>
    </form>
  </dialog>`;
}

function boardMarkup(board: number, legal: Set<number>, reviewedMove: number | null = null): string {
  const result = state.results[board];
  const cells = state.boards[board]
    .map((mark, cell) => {
      const playable = !reviewActive && !result && isLegal(state, board, cell) && canPlayTurn();
      const suggested = hintMove === board * 9 + cell;
      const isReviewedMove = reviewedMove === board * 9 + cell;
      const classes = [
        'cell',
        mark ? `cell-${mark.toLowerCase()}` : '',
        suggested ? 'cell-suggested' : '',
        isReviewedMove ? 'cell-review-current' : '',
      ]
        .filter(Boolean)
        .join(' ');
      const label = mark
        ? `Board ${board + 1}, square ${cell + 1}, ${mark}${isReviewedMove ? ', reviewed move' : ''}`
        : `Board ${board + 1}, square ${cell + 1}${suggested ? ', hinted move' : playable ? ', available' : ', unavailable'}`;
      return `<button class="${classes}" type="button" data-board="${board}" data-cell="${cell}" aria-label="${label}"${suggested ? ' title="Suggested move"' : ''}${playable ? '' : ' disabled'}>${mark ? `<span class="cell-mark mark-${mark.toLowerCase()}" aria-hidden="true">${mark}</span>` : ''}</button>`;
    })
    .join('');

  const isForced = state.forced === board && legal.has(board);
  const classes = [
    'mini-board',
    legal.has(board) ? 'is-legal' : '',
    isForced ? 'is-forced' : '',
    result ? 'is-complete' : '',
    animatingBoards.has(board) ? 'just-completed' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return `<section class="${classes}" data-mini-board="${board}" aria-label="Board ${board + 1}${result ? `, ${result === 'draw' ? 'drawn' : `won by ${result}`}` : legal.has(board) ? ', available' : ', unavailable'}">
    <div class="mini-cells">${cells}</div>
    ${resultOverlay(result)}
  </section>`;
}

function formatRoomCode(code: string): string {
  return `${code.slice(0, 4)} ${code.slice(4)}`;
}

function onlineTurnTitle(): string {
  if (!onlineSeat) return onlineBusy ? 'Setting up room…' : 'Online game';
  if (state.winner === 'draw') return 'Draw';
  if (state.winner) return state.winner === onlineSeat.seat ? 'You win' : `${state.winner} wins`;
  if (onlineConnectionStatus === 'reconnecting') return 'Reconnecting…';
  if (onlineConnectionStatus !== 'connected') return 'Connecting…';
  if (onlineSeat.room.players < 2) return 'Waiting for player';
  if (onlineMovePending) return 'Sending move…';
  return state.turn === onlineSeat.seat ? 'Your turn' : `${state.turn} to play`;
}

function onlineStatusText(): string {
  if (!onlineSeat) return onlineSetupError;
  if (onlineError) return onlineError;
  if (onlineConnectionStatus === 'error') return 'Reconnect by reopening the invite link.';
  if (onlineConnectionStatus === 'reconnecting') return 'Connection interrupted · reconnecting…';
  if (onlineConnectionStatus !== 'connected') return 'Connecting to the room…';
  if (onlineSeat.room.players < 2) return 'Waiting for the other player to join.';
  if (state.winner === 'draw') return 'Game drawn.';
  if (state.winner) return state.winner === onlineSeat.seat ? 'You won the game.' : 'The other player won.';
  if (onlineMovePending) return 'Move sent · waiting for the board update.';
  return state.turn === onlineSeat.seat ? `You play ${onlineSeat.seat}.` : `Waiting for ${state.turn}.`;
}

function onlineRoomMarkup(): string {
  if (!onlineSeat) return '';
  const formattedCode = formatRoomCode(onlineSeat.code);
  return `<section class="online-room" aria-label="Online game room">
    <div class="online-room-info"><span>Room code</span><strong>${formattedCode}</strong><small>${onlineNotice || onlineStatusText()}</small></div>
    <div class="online-room-actions">
      <button class="analysis-button" type="button" data-action="copy-room-code">Copy code</button>
      <button class="analysis-button" type="button" data-action="copy-room-link">Copy link</button>
      <button class="analysis-button" type="button" data-action="leave-online">Leave</button>
    </div>
  </section>`;
}

function render(): void {
  const openDialogId = document.querySelector<HTMLDialogElement>('dialog[open]')?.id;
  const reviewPanelScrollTop = app!.querySelector<HTMLElement>('.review-panel')?.scrollTop ?? null;
  const reviewMoveStripScrollTop = app!.querySelector<HTMLElement>('.review-move-strip')?.scrollTop ?? null;
  const focusedElement = document.activeElement;
  const activeControl = (focusedElement instanceof HTMLElement || focusedElement instanceof SVGElement) && app!.contains(focusedElement)
    ? {
      id: focusedElement instanceof HTMLElement ? focusedElement.id : '',
      action: focusedElement instanceof HTMLElement ? focusedElement.dataset.action : undefined,
      reviewPosition: focusedElement.getAttribute('data-review-position') ?? undefined,
      reviewControl: focusedElement.matches('.review-critical-move') ? 'critical'
        : focusedElement.matches('.review-move-chip') ? 'move-chip'
          : focusedElement.matches('.review-graph-point') ? 'graph-point' : undefined,
      mode: focusedElement instanceof HTMLElement ? focusedElement.dataset.newGameMode : undefined,
      sideChoice: focusedElement instanceof HTMLElement ? focusedElement.dataset.sideChoice : undefined,
    }
    : null;
  const legal = new Set(legalBoards(state));
  const title = reviewActive ? 'Game review'
    : opponentMode === 'online' && onlineSeat && state.winner
      ? state.winner === 'draw' ? 'Draw' : state.winner === onlineSeat.seat ? 'You win' : `${state.winner} wins`
      : state.winner === 'draw' ? 'Draw'
      : state.winner ? state.winner + ' wins'
        : computerThinking ? 'Computer thinking…'
          : isComputerTurn() ? 'Computer to play'
            : opponentMode === 'online' ? onlineTurnTitle()
              : state.turn + ' to play';
  const kind = (state.winner === 'draw' ? 'draw' : state.winner ?? state.turn).toLowerCase();
  const status = reviewActive ? `Reviewing ${reviewPositionLabel()}`
    : computerStatus || (state.winner === 'draw' ? 'Game drawn.'
      : opponentMode === 'computer' && state.winner && state.winner !== humanSide ? 'Computer wins.'
        : opponentMode === 'computer' && state.winner === humanSide ? 'You win.'
          : opponentMode === 'online' ? onlineStatusText()
            : state.winner ? '' : computerSource === 'model' ? `Your turn · you play ${humanSide}.`
              : computerSource === 'search' ? `You play ${humanSide} · computer uses basic search.` : '');

  app!.innerHTML = `
    <main class="page-shell${reviewActive ? ' review-open' : ''}">
      <header class="heading">
        <h1>Ultimate<span>Tic-Tac-Toe</span></h1>
        <div class="scores" aria-label="Boards won">
          <div class="score score-x" aria-label="X has ${countBoards('X')} boards"><span>X</span><strong>${countBoards('X')}</strong></div>
          <div class="score score-o" aria-label="O has ${countBoards('O')} boards"><span>O</span><strong>${countBoards('O')}</strong></div>
        </div>
      </header>
      <div class="game-column">
      <div class="toolbar">
        <h2 class="turn-title turn-${kind}" aria-live="polite">${title}</h2>
        <div class="actions">
          <button type="button" data-action="undo" aria-label="Undo" title="Undo"${undoStack.length && !reviewActive && opponentMode !== 'online' ? '' : ' disabled'}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4 5 8l4 4M5 8h8a6 6 0 1 1 0 12h-2" /></svg></button>
          <button type="button" data-action="new-game" aria-label="New game" title="New game">+</button>
          <button type="button" data-action="howto" aria-label="How to play" title="How to play">?</button>
        </div>
      </div>
      <div class="computer-status" role="status" aria-live="polite" data-computer-source="${reviewActive ? '' : computerSource}">${status}${computerFailed && !reviewActive ? '<button class="retry-computer" type="button" data-action="retry-computer">Retry computer</button>' : ''}</div>
      ${onlineRoomMarkup()}
      <div class="board-frame turn-${kind}" aria-label="Ultimate Tic-Tac-Toe board">
        <div class="big-board">${Array.from({ length: 9 }, (_, board) => boardMarkup(board, legal, reviewActive && reviewCursor > 0 ? reviewMoves[reviewCursor - 1]?.move ?? null : null)).join('')}</div>
      </div>
      <div class="under-board">
        ${themeButtonMarkup()}
        <div class="route-map" aria-label="${state.winner ? 'Game complete' : state.forced === null ? 'Any open board' : 'Play board ' + (state.forced + 1)}">
          ${Array.from({ length: 9 }, (_, i) => '<i class="' + (legal.has(i) ? 'lit' : '') + '"></i>').join('')}
        </div>
        ${reviewActive || state.winner
          ? `<div class="board-tools"><div class="analysis-actions"><button class="analysis-button" type="button" data-action="review-toggle" aria-pressed="${reviewActive}">${reviewActive ? 'Exit review' : 'Review game'}</button></div></div>`
          : opponentMode === 'computer'
            ? `<div class="board-tools">
                <span class="hint-status" role="status" aria-live="polite">${hintPending ? 'Finding a move…' : hintMove !== null ? `Try ${boardMoveName(hintMove)}` : hintFailed ? 'Hint unavailable' : ''}</span>
                <div class="analysis-actions">
                  <button class="analysis-button" type="button" data-action="hint"${isComputerTurn() || computerThinking || hintPending ? ' disabled' : ''}>${hintPending ? 'Finding…' : hintMove !== null ? 'Hint shown' : 'Hint'}</button>
                </div>
              </div>`
            : ''}
      </div>
      </div>
      ${reviewPanelMarkup()}
    </main>
    <dialog class="modal" id="new-game-dialog" aria-labelledby="new-game-title">
      <form method="dialog" class="modal-card">
        <h2 id="new-game-title">New game</h2>
        <div class="new-game-modes" role="group" aria-label="Game mode">
          <button class="mode-button${draftOpponentMode === 'two-players' ? ' is-selected' : ''}" type="button" data-new-game-mode="two-players" aria-label="Two players" aria-pressed="${draftOpponentMode === 'two-players'}">2 players</button>
          <button class="mode-button${draftOpponentMode === 'computer' ? ' is-selected' : ''}" type="button" data-new-game-mode="computer" aria-label="Play vs computer" aria-pressed="${draftOpponentMode === 'computer'}">Computer</button>
          <button class="mode-button${draftOpponentMode === 'online' ? ' is-selected' : ''}" type="button" data-new-game-mode="online" aria-label="Play online" aria-pressed="${draftOpponentMode === 'online'}">Online</button>
        </div>
        ${draftOpponentMode === 'online' ? `<div class="online-setup">
          <p>Play someone on another device.</p>
          <button class="primary-button" type="button" data-action="create-online-room"${onlineBusy ? ' disabled' : ''}>${onlineBusy ? 'Connecting…' : 'Create room'}</button>
          <div class="online-join-row"><label for="online-room-code">Have a code?</label><div><input id="online-room-code" type="text" inputmode="text" autocomplete="one-time-code" autocapitalize="characters" maxlength="10" placeholder="ABCD EFGH" value="${draftRoomCode}" data-action="online-room-code" aria-label="Room code" /><button class="analysis-button" type="button" data-action="join-online-room"${onlineBusy ? ' disabled' : ''}>Join</button></div></div>
          ${onlineSetupError ? `<p class="online-setup-error" role="alert">${onlineSetupError}</p>` : ''}
        </div>` : draftOpponentMode === 'computer' ? `<div class="new-game-computer-options">
          <div class="difficulty-setting">
            <div class="difficulty-heading"><label for="new-game-difficulty">Difficulty</label><small id="computer-difficulty-help">More search before each move.</small><strong>${difficultyLabel(draftDifficulty)}</strong></div>
            <input id="new-game-difficulty" class="difficulty-range" type="range" min="0" max="2" step="1" value="${DIFFICULTIES.indexOf(draftDifficulty)}" data-action="new-game-difficulty" aria-label="Computer difficulty" aria-valuetext="${difficultyLabel(draftDifficulty)}" aria-describedby="computer-difficulty-help" />
            <div class="difficulty-labels" aria-hidden="true"><span>Easy</span><span>Medium</span><span>Hard</span></div>
          </div>
          <fieldset class="side-picker">
            <legend>You play</legend>
            <div class="side-options" role="group" aria-label="Choose your side">
              ${(['X', 'O', 'random'] as SideChoice[]).map((choice) => `<button type="button" class="side-option${draftSideChoice === choice ? ' is-selected' : ''}" data-side-choice="${choice}" aria-pressed="${draftSideChoice === choice}">${sideChoiceLabel(choice)}</button>`).join('')}
            </div>
          </fieldset>
        </div>` : ''}
        <div class="modal-actions">
          <button type="submit">Cancel</button>
          ${draftOpponentMode === 'online' ? '' : '<button class="primary-button" type="button" data-action="start-new-game">Start game</button>'}
        </div>
      </form>
    </dialog>
    <dialog class="modal rules-modal" id="how-to-dialog" aria-labelledby="howto-title">
      <form method="dialog" class="modal-card rules-card">
        <h2 id="howto-title">How to play</h2>
        <p>Ultimate Tic-Tac-Toe is nine small tic-tac-toe boards in one big board. X moves first.</p>
        <ol class="rules-list">
          <li>On your turn, place your mark in any open square of an open small board.</li>
          <li>The square you choose sends your opponent to the matching small board. For example, choosing the top-right square sends them to the top-right board.</li>
          <li>If that board is already won or drawn, your opponent may choose any open board.</li>
          <li>Win a small board with three marks in a row. Win the game by claiming three small boards in a row.</li>
          <li>If every small board closes before either player wins the big board, the game is a draw.</li>
        </ol>
        <p class="rules-link"><a href="https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe" target="_blank" rel="noopener noreferrer">More about Ultimate Tic-Tac-Toe on Wikipedia ↗</a></p>
        <button class="primary-button" type="submit">Got it</button>
      </form>
    </dialog>
    ${analysisDialogMarkup()}
  `;
  const renderedReviewPanel = app!.querySelector<HTMLElement>('.review-panel');
  if (reviewPanelScrollTop !== null && renderedReviewPanel) renderedReviewPanel.scrollTop = reviewPanelScrollTop;
  const renderedMoveStrip = app!.querySelector<HTMLElement>('.review-move-strip');
  if (reviewMoveStripScrollTop !== null && renderedMoveStrip) renderedMoveStrip.scrollTop = reviewMoveStripScrollTop;
  bindEvents();
  if (openDialogId) openDialog(openDialogId);
  if (activeControl) {
    const selector = activeControl.id
      ? `#${activeControl.id}`
      : activeControl.mode
        ? `[data-new-game-mode="${activeControl.mode}"]`
      : activeControl.sideChoice
          ? `[data-side-choice="${activeControl.sideChoice}"]`
        : activeControl.reviewPosition !== undefined
          ? activeControl.reviewControl === 'critical'
            ? `.review-critical-move[data-review-position="${activeControl.reviewPosition}"]`
            : activeControl.reviewControl === 'move-chip'
              ? `.review-move-chip[data-review-position="${activeControl.reviewPosition}"]`
              : activeControl.reviewControl === 'graph-point'
                ? `.review-graph-point[data-review-position="${activeControl.reviewPosition}"]`
                : `[data-review-position="${activeControl.reviewPosition}"]`
        : activeControl.action
          ? `[data-action="${activeControl.action}"]`
          : null;
    if (selector) app!.querySelector<HTMLElement | SVGElement>(selector)?.focus({ preventScroll: true });
  }
  animatingBoards.clear();
}

function bindEvents(): void {
  app!.querySelector<HTMLButtonElement>('[data-action="theme"]')?.addEventListener('click', cycleTheme);
  app!.querySelectorAll<HTMLButtonElement>('[data-board][data-cell]').forEach((button) => {
    button.addEventListener('click', () => {
      const board = Number(button.dataset.board);
      const cell = Number(button.dataset.cell);
      if (!canPlayTurn()) return;
      if (!isLegal(state, board, cell)) return;

      if (opponentMode === 'online') submitOnlineMove(board, cell);
      else makeMove(board, cell);
    });

    button.addEventListener('keydown', (event) => {
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      const board = Number(button.dataset.board);
      const cell = Number(button.dataset.cell);
      const globalRow = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
      const globalColumn = (board % 3) * 3 + (cell % 3);
      const rowStep = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
      const columnStep = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;

      for (let distance = 1; distance < 9; distance += 1) {
        const row = (globalRow + rowStep * distance + 9) % 9;
        const column = (globalColumn + columnStep * distance + 9) % 9;
        const targetBoard = Math.floor(row / 3) * 3 + Math.floor(column / 3);
        const targetCell = (row % 3) * 3 + (column % 3);
        if (!isLegal(state, targetBoard, targetCell)) continue;
        event.preventDefault();
        focusCell(targetBoard, targetCell);
        return;
      }
    });
  });

  app!.querySelectorAll<HTMLButtonElement>('[data-action="howto"]').forEach((button) => {
    button.addEventListener('click', () => openDialog('how-to-dialog'));
  });
  app!.querySelector<HTMLButtonElement>('[data-action="new-game"]')?.addEventListener('click', () => {
    openNewGameMenu();
  });
  app!.querySelector<HTMLButtonElement>('[data-action="start-new-game"]')?.addEventListener('click', resetGame);
  app!.querySelector<HTMLButtonElement>('[data-action="create-online-room"]')?.addEventListener('click', () => { void createOnlineGame(); });
  app!.querySelector<HTMLButtonElement>('[data-action="join-online-room"]')?.addEventListener('click', () => {
    const code = app!.querySelector<HTMLInputElement>('#online-room-code')?.value ?? draftRoomCode;
    void joinOnlineGame(code);
  });
  app!.querySelector<HTMLInputElement>('#online-room-code')?.addEventListener('input', (event) => {
    draftRoomCode = normalizeRoomCode((event.currentTarget as HTMLInputElement).value);
  });
  app!.querySelector<HTMLButtonElement>('[data-action="copy-room-code"]')?.addEventListener('click', () => {
    if (onlineSeat) void copyOnlineInvite(formatRoomCode(onlineSeat.code), 'Room code');
  });
  app!.querySelector<HTMLButtonElement>('[data-action="copy-room-link"]')?.addEventListener('click', () => {
    if (onlineSeat) void copyOnlineInvite(roomInviteUrl(onlineSeat.code), 'Invite link');
  });
  app!.querySelector<HTMLButtonElement>('[data-action="leave-online"]')?.addEventListener('click', leaveOnlineGame);
  app!.querySelector<HTMLButtonElement>('[data-action="undo"]')?.addEventListener('click', undoMove);
  app!.querySelector<HTMLButtonElement>('[data-action="retry-computer"]')?.addEventListener('click', retryComputerMove);
  app!.querySelector<HTMLButtonElement>('[data-action="hint"]')?.addEventListener('click', showHint);
  app!.querySelector<HTMLButtonElement>('[data-action="analysis"]')?.addEventListener('click', openAnalysis);
  app!.querySelector<HTMLButtonElement>('[data-action="review-toggle"]')?.addEventListener('click', () => {
    if (reviewActive) closeGameReview();
    else openGameReview();
  });
  app!.querySelector<HTMLButtonElement>('[data-action="review-start"]')?.addEventListener('click', () => setReviewPosition(0));
  app!.querySelector<HTMLButtonElement>('[data-action="review-playback"]')?.addEventListener('click', toggleReviewPlayback);
  app!.querySelector<HTMLButtonElement>('[data-action="review-speed"]')?.addEventListener('click', changeReviewPlaybackSpeed);
  app!.querySelector<HTMLButtonElement>('[data-action="review-export"]')?.addEventListener('click', downloadGameNotation);
  app!.querySelector<HTMLButtonElement>('[data-action="review-previous"]')?.addEventListener('click', () => setReviewPosition(reviewCursor - 1));
  app!.querySelector<HTMLButtonElement>('[data-action="review-next"]')?.addEventListener('click', () => setReviewPosition(reviewCursor + 1));
  app!.querySelector<HTMLButtonElement>('[data-action="review-end"]')?.addEventListener('click', () => setReviewPosition(reviewSnapshots.length - 1));
  app!.querySelector<HTMLInputElement>('[data-action="review-position"]')?.addEventListener('change', (event) => {
    setReviewPosition(Number((event.currentTarget as HTMLInputElement).value));
  });
  app!.querySelectorAll<SVGCircleElement>('.review-graph-point').forEach((point) => {
    point.addEventListener('click', () => setReviewPosition(Number(point.dataset.reviewPosition)));
    point.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      setReviewPosition(Number(point.dataset.reviewPosition));
    });
  });
  app!.querySelector<SVGSVGElement>('.review-graph')?.addEventListener('click', (event) => {
    const target = event.target;
    if (target instanceof SVGElement && target.closest('.review-graph-point')) return;
    const graph = event.currentTarget as SVGSVGElement;
    const bounds = graph.getBoundingClientRect();
    if (!bounds.width || reviewSnapshots.length < 2) return;
    const graphX = ((event.clientX - bounds.left) / bounds.width) * 600;
    const ratio = Math.min(1, Math.max(0, (graphX - 20) / 560));
    setReviewPosition(Math.round(ratio * (reviewSnapshots.length - 1)));
  });
  app!.querySelectorAll<HTMLButtonElement>('.review-move-chip[data-review-position]').forEach((button) => {
    button.addEventListener('click', () => setReviewPosition(Number(button.dataset.reviewPosition)));
  });
  app!.querySelectorAll<HTMLButtonElement>('.review-critical-move[data-review-position]').forEach((button) => {
    button.addEventListener('click', () => setReviewPosition(Number(button.dataset.reviewPosition)));
  });
  const analysisDialog = app!.querySelector<HTMLDialogElement>('#analysis-dialog');
  analysisDialog?.addEventListener('close', () => {
    if (document.getElementById('analysis-dialog') !== analysisDialog) return;
    cancelAnalysisWork();
    analysisView = null;
    render();
  });
  app!.querySelectorAll<HTMLButtonElement>('[data-side-choice]').forEach((button) => {
    button.addEventListener('click', () => {
      const choice = button.dataset.sideChoice;
      if (choice !== 'X' && choice !== 'O' && choice !== 'random') return;
      draftSideChoice = choice;
      render();
    });
  });
  app!.querySelectorAll<HTMLButtonElement>('[data-new-game-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      const selectedMode = button.dataset.newGameMode;
      if (selectedMode !== 'two-players' && selectedMode !== 'computer' && selectedMode !== 'online') return;
      if (selectedMode === draftOpponentMode) return;
      draftOpponentMode = selectedMode;
      onlineSetupError = '';
      render();
    });
  });
  app!.querySelector<HTMLInputElement>('[data-action="new-game-difficulty"]')?.addEventListener('change', (event) => {
    const index = Number((event.currentTarget as HTMLInputElement).value);
    const selectedDifficulty = DIFFICULTIES[index];
    if (!selectedDifficulty || selectedDifficulty === draftDifficulty) return;
    draftDifficulty = selectedDifficulty;
    render();
  });
}

function openDialog(id: string): void {
  const dialog = document.getElementById(id);
  if (!(dialog instanceof HTMLDialogElement)) return;
  if (!dialog.open) dialog.showModal();
}

function openNewGameMenu(): void {
  draftOpponentMode = opponentMode;
  draftDifficulty = difficulty;
  draftSideChoice = sideChoice;
  draftRoomCode = onlineSeat?.code ?? '';
  onlineSetupError = '';
  render();
  openDialog('new-game-dialog');
}

function focusCell(board: number, cell?: number): void {
  requestAnimationFrame(() => {
    const targetCell = cell ?? state.boards[board]?.findIndex((mark, index) => mark === null && isLegal(state, board, index));
    const target = targetCell === undefined || targetCell < 0
      ? null
      : app!.querySelector<HTMLButtonElement>(`[data-board="${board}"][data-cell="${targetCell}"]`);
    target?.focus({ preventScroll: true });
    target?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  });
}

function makeMove(board: number, cell: number, computerMove = false, source?: 'model' | 'search'): void {
  if (reviewActive) return;
  const updateOpenAnalysis = analysisView === 'position';
  cancelAnalysisWork();
  const previousResults = [...state.results];
  undoStack.push(snapshot(state));
  state = play(state, board, cell);
  if (computerMove) computerMoves.add(state.moves);
  animatingBoards = new Set(
    state.results.flatMap((result, index) => previousResults[index] === null && result !== null ? [index] : []),
  );
  computerStatus = '';
  computerSource = source ?? '';
  computerFailed = false;
  analysisPending = updateOpenAnalysis && state.winner === null;
  persist();
  render();
  if (analysisPending) requestPositionAnalysis(false);
  if (!state.winner && !isComputerTurn()) {
    const availableBoards = legalBoards(state);
    const destination = availableBoards.includes(cell) ? cell : availableBoards[0];
    if (destination !== undefined) focusCell(destination);
  }
  scheduleComputerMove();
}

function undoMove(): void {
  if (reviewActive || opponentMode === 'online') return;
  cancelComputerMove();
  cancelAnalysisWork();
  // In computer mode, undo the complete X/O exchange. While O is still thinking,
  // only X's pending move exists in the history, so one snapshot is enough.
  const movesToUndo = computerMoves.has(state.moves) ? 2 : 1;
  let previous: GameState | undefined;
  for (let i = 0; i < movesToUndo; i += 1) previous = undoStack.pop() ?? previous;
  if (!previous) return;
  state = previous;
  computerMoves = new Set([...computerMoves].filter((move) => move <= state.moves));
  computerStatus = '';
  computerSource = '';
  computerFailed = false;
  persist();
  render();
  const availableBoards = legalBoards(state);
  if (availableBoards.length) focusCell(availableBoards[0]);
  scheduleComputerMove();
}

function resetGame(): void {
  stopReviewPlayback(false);
  reviewActive = false;
  reviewFinalState = null;
  reviewSnapshots = [];
  reviewMoves = [];
  reviewItems = [];
  reviewEvaluations = [];
  reviewCursor = 0;
  reviewPending = false;
  reviewUnavailable = 0;
  reviewHistoryAvailable = true;
  onlineConnection?.close();
  onlineConnection = null;
  onlineSeat = null;
  onlineConnectionStatus = 'closed';
  onlineSetupError = '';
  onlineError = '';
  onlineNotice = '';
  onlineMovePending = false;
  onlineBusy = false;
  setRoomQuery(null);
  cancelComputerMove();
  cancelAnalysisWork();
  opponentMode = draftOpponentMode;
  difficulty = draftDifficulty;
  sideChoice = draftSideChoice;
  persistOpponentPreference();
  humanSide = opponentMode === 'computer'
    ? sideChoice === 'random' ? (Math.random() < 0.5 ? 'X' : 'O') : sideChoice
    : 'X';
  state = createGame();
  undoStack = [];
  computerMoves.clear();
  computerStatus = '';
  computerSource = '';
  computerFailed = false;
  persist();
  const dialog = document.getElementById('new-game-dialog');
  if (dialog instanceof HTMLDialogElement && dialog.open) dialog.close();
  render();
  const availableBoards = legalBoards(state);
  if (availableBoards.length && !isComputerTurn()) focusCell(availableBoards[0]);
  scheduleComputerMove();
}

function isComputerTurn(): boolean {
  return opponentMode === 'computer' && state.turn !== humanSide && state.winner === null;
}

function canPlayTurn(): boolean {
  if (reviewActive || state.winner) return false;
  if (opponentMode === 'computer') return !isComputerTurn();
  if (opponentMode === 'online') {
    return onlineSeat !== null && onlineSeat.room.players === 2 && onlineConnectionStatus === 'connected'
      && onlineSeat.seat === state.turn && !onlineMovePending;
  }
  return true;
}

function reconstructOnlineRoom(room: PublicRoom): { state: GameState; undo: GameState[] } | null {
  if (!Array.isArray(room.moves) || room.version !== room.moves.length) return null;
  let replay = createGame();
  const undo: GameState[] = [];
  for (const move of room.moves) {
    if (!Number.isInteger(move) || move < 0 || move >= 81) return null;
    const board = Math.floor(move / 9);
    const cell = move % 9;
    if (!isLegal(replay, board, cell)) return null;
    undo.push(snapshot(replay));
    replay = play(replay, board, cell);
  }
  const reported = restoreGame(room.state);
  if (!reported || JSON.stringify(reported) !== JSON.stringify(replay)) return null;
  return { state: reported, undo };
}

function applyOnlineSnapshot(room: PublicRoom): void {
  if (!onlineSeat || room.code !== onlineSeat.code || room.version < onlineVersion) return;
  const restored = reconstructOnlineRoom(room);
  if (!restored) {
    onlineError = 'The room sent an invalid game state.';
    onlineMovePending = false;
    render();
    return;
  }
  const previousResults = [...state.results];
  state = restored.state;
  undoStack = restored.undo;
  onlineSeat = { ...onlineSeat, room };
  onlineVersion = room.version;
  onlineMovePending = false;
  onlineError = '';
  animatingBoards = new Set(
    state.results.flatMap((result, index) => previousResults[index] === null && result !== null ? [index] : []),
  );
  cancelAnalysisWork();
  render();
}

function setRoomQuery(code: string | null): void {
  const url = new URL(window.location.href);
  if (code) url.searchParams.set('room', code);
  else url.searchParams.delete('room');
  window.history.replaceState(null, '', url);
}

function activateOnlineSeat(seat: OnlineSeat): void {
  const restored = reconstructOnlineRoom(seat.room);
  if (!restored) {
    onlineSetupError = 'The room sent an invalid game state.';
    render();
    return;
  }
  onlineConnection?.close();
  cancelComputerMove();
  cancelAnalysisWork();
  reviewActive = false;
  reviewFinalState = null;
  reviewSnapshots = [];
  reviewMoves = [];
  reviewItems = [];
  reviewEvaluations = [];
  reviewCursor = 0;
  opponentMode = 'online';
  onlineSeat = seat;
  onlineVersion = seat.room.version;
  onlineConnectionStatus = 'connecting';
  onlineSetupError = '';
  onlineError = '';
  onlineNotice = '';
  onlineMovePending = false;
  onlineBusy = false;
  state = restored.state;
  undoStack = restored.undo;
  computerMoves.clear();
  computerStatus = '';
  computerSource = '';
  computerFailed = false;
  setRoomQuery(seat.code);
  persistOpponentPreference();
  const dialog = document.getElementById('new-game-dialog');
  if (dialog instanceof HTMLDialogElement && dialog.open) dialog.close();
  render();

  const connection = new OnlineRoomConnection(seat, {
    onStatus(status) {
      if (onlineSeat?.code !== seat.code) return;
      onlineConnectionStatus = status;
      render();
    },
    onMessage(message) {
      if (onlineSeat?.code !== seat.code) return;
      if (message.type === 'snapshot') {
        if (message.seat && message.seat !== seat.seat) {
          onlineError = 'This room connection belongs to a different seat.';
          render();
          return;
        }
        applyOnlineSnapshot(message.room);
      } else {
        if (message.room) applyOnlineSnapshot(message.room);
        onlineError = message.message;
        onlineMovePending = false;
        render();
      }
    },
    onError(message) {
      if (onlineSeat?.code !== seat.code) return;
      onlineError = message;
      onlineMovePending = false;
      render();
    },
  });
  onlineConnection = connection;
  connection.connect();
}

async function createOnlineGame(): Promise<void> {
  if (onlineBusy) return;
  onlineBusy = true;
  onlineSetupError = '';
  render();
  try {
    const seat = await requestOnlineRoom();
    onlineBusy = false;
    activateOnlineSeat(seat);
  } catch (error: unknown) {
    onlineBusy = false;
    onlineSetupError = error instanceof Error ? error.message : 'Could not create a room.';
    render();
  }
}

async function joinOnlineGame(input: string, fromInvite = false): Promise<void> {
  if (onlineBusy) return;
  draftRoomCode = normalizeRoomCode(input);
  if (fromInvite) {
    opponentMode = 'online';
    state = createGame();
    undoStack = [];
    computerMoves.clear();
  }
  onlineBusy = true;
  onlineSetupError = '';
  render();
  try {
    const seat = await requestJoinOnlineRoom(draftRoomCode);
    onlineBusy = false;
    activateOnlineSeat(seat);
  } catch (error: unknown) {
    onlineBusy = false;
    onlineSetupError = error instanceof Error ? error.message : 'Could not join the room.';
    draftOpponentMode = 'online';
    render();
    if (fromInvite) openDialog('new-game-dialog');
  }
}

async function copyOnlineInvite(value: string, label: string): Promise<void> {
  try {
    if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable.');
    await navigator.clipboard.writeText(value);
    onlineNotice = `${label} copied.`;
  } catch {
    onlineNotice = `Copy this ${label.toLowerCase()}: ${value}`;
  }
  render();
}

function leaveOnlineGame(): void {
  onlineConnection?.close();
  onlineConnection = null;
  onlineSeat = null;
  onlineConnectionStatus = 'closed';
  onlineError = '';
  onlineNotice = '';
  onlineMovePending = false;
  onlineSetupError = '';
  onlineBusy = false;
  opponentMode = 'two-players';
  humanSide = 'X';
  state = createGame();
  undoStack = [];
  computerMoves.clear();
  setRoomQuery(null);
  persistOpponentPreference();
  persist();
  render();
}

function submitOnlineMove(board: number, cell: number): void {
  if (!onlineConnection || !onlineSeat || !canPlayTurn()) return;
  onlineMovePending = true;
  onlineError = '';
  onlineNotice = '';
  const sent = onlineConnection.sendMove(board, cell, onlineVersion);
  if (!sent) {
    onlineMovePending = false;
    onlineError = 'Reconnecting to the room…';
  }
  render();
}

function cancelComputerMove(): void {
  computerRequestToken += 1;
  if (computerThinking) computer.cancel();
  computerThinking = false;
}

function cancelAnalysisWork(): void {
  hintToken += 1;
  analysisToken += 1;
  analyzer.cancel();
  reviewDetailToken += 1;
  reviewDetailAnalyzer.cancel();
  reviewDetailAnalysis = null;
  reviewDetailPosition = -1;
  reviewDetailPending = false;
  reviewDetailFailed = false;
  hintPending = false;
  hintFailed = false;
  hintMove = null;
  analysisPending = false;
  analysisFailed = false;
  positionAnalysis = null;
  positionAnalysisMoves = -1;
}

function showHint(): void {
  if (opponentMode !== 'computer' || reviewActive || state.winner || isComputerTurn() || computerThinking || hintPending) return;
  analysisToken += 1;
  analyzer.cancel();
  const token = ++hintToken;
  hintMove = null;
  hintFailed = false;
  hintPending = true;
  render();
  void analyzer.analyze(snapshot(state), 'hint').then(analysis => {
    if (token !== hintToken || opponentMode !== 'computer' || state.winner || isComputerTurn()) return;
    hintPending = false;
    hintMove = analysis.lines[0]?.[0] ?? null;
    hintFailed = hintMove === null;
    render();
  }).catch((error: unknown) => {
    if (token !== hintToken || error instanceof DOMException && error.name === 'AbortError') return;
    hintPending = false;
    hintFailed = true;
    render();
  });
}

function openAnalysis(): void {
  const dialog = document.getElementById('analysis-dialog');
  if (analysisView === 'position' && dialog instanceof HTMLDialogElement && dialog.open) {
    dialog.close();
    return;
  }
  if (state.winner) return;
  cancelAnalysisWork();
  analysisView = 'position';
  analysisPending = true;
  render();
  openDialog('analysis-dialog');
  requestPositionAnalysis(false);
}

function requestPositionAnalysis(renderBeforeRequest = true): void {
  if (analysisView !== 'position' || state.winner) return;
  const token = ++analysisToken;
  analysisPending = true;
  analysisFailed = false;
  positionAnalysis = null;
  positionAnalysisMoves = -1;
  if (renderBeforeRequest) render();
  void analyzer.analyze(snapshot(state), 'analysis').then(analysis => {
    if (token !== analysisToken || analysisView !== 'position') return;
    analysisPending = false;
    positionAnalysis = analysis;
    positionAnalysisMoves = state.moves;
    render();
  }).catch((error: unknown) => {
    if (token !== analysisToken || analysisView !== 'position' || error instanceof DOMException && error.name === 'AbortError') return;
    analysisPending = false;
    analysisFailed = true;
    render();
  });
}

function openGameReview(): void {
  if (reviewActive || !state.winner) return;
  cancelAnalysisWork();
  analysisView = null;
  reviewFinalState = snapshot(state);
  const timeline = buildReviewTimeline();
  reviewSnapshots = timeline.snapshots.length ? timeline.snapshots : [snapshot(state)];
  reviewMoves = timeline.moves;
  reviewItems = [];
  reviewEvaluations = Array(reviewSnapshots.length).fill(null);
  reviewCursor = Math.max(0, reviewSnapshots.length - 2);
  stopReviewPlayback(false);
  reviewPlaybackSpeedIndex = 1;
  reviewUnavailable = 0;
  reviewHistoryAvailable = timeline.snapshots.length > 1;
  reviewEvaluations[reviewSnapshots.length - 1] = {
    xValue: state.winner === 'X' ? 1 : state.winner === 'O' ? -1 : 0,
    source: 'search',
    solved: true,
    lines: [],
  };
  reviewPending = reviewSnapshots.length > 1;
  reviewActive = true;
  state = snapshot(reviewSnapshots[reviewCursor]);
  render();
  scrollReviewMoveIntoView(reviewCursor);
  requestReviewDetail(reviewCursor);
  if (reviewPending) void runGameReview(++analysisToken);
}

function stopReviewPlayback(updateUi = true): void {
  if (reviewPlaybackTimer !== null) window.clearTimeout(reviewPlaybackTimer);
  reviewPlaybackTimer = null;
  const wasPlaying = reviewPlaying;
  reviewPlaying = false;
  if (wasPlaying && updateUi && reviewActive) render();
}

function toggleReviewPlayback(): void {
  if (!reviewActive || reviewSnapshots.length < 2) return;
  if (reviewPlaying) {
    stopReviewPlayback();
    return;
  }
  if (reviewCursor >= reviewSnapshots.length - 1) setReviewPosition(0);
  reviewPlaying = true;
  render();
  scheduleReviewPlayback();
}

function scheduleReviewPlayback(): void {
  if (!reviewPlaying || !reviewActive) return;
  const finalPosition = reviewSnapshots.length - 1;
  if (reviewCursor >= finalPosition) {
    stopReviewPlayback();
    return;
  }
  const delay = 1000 / REVIEW_PLAYBACK_SPEEDS[reviewPlaybackSpeedIndex];
  reviewPlaybackTimer = window.setTimeout(() => {
    reviewPlaybackTimer = null;
    if (!reviewPlaying || !reviewActive) return;
    const nextPosition = reviewCursor + 1;
    const isFinalPosition = nextPosition >= finalPosition;
    if (isFinalPosition) reviewPlaying = false;
    setReviewPosition(nextPosition, true);
    if (!isFinalPosition && reviewPlaying) scheduleReviewPlayback();
  }, delay);
}

function changeReviewPlaybackSpeed(): void {
  const wasPlaying = reviewPlaying;
  if (reviewPlaybackTimer !== null) window.clearTimeout(reviewPlaybackTimer);
  reviewPlaybackTimer = null;
  reviewPlaybackSpeedIndex = (reviewPlaybackSpeedIndex + 1) % REVIEW_PLAYBACK_SPEEDS.length;
  render();
  if (wasPlaying) scheduleReviewPlayback();
}

function setReviewPosition(position: number, autoplayStep = false): void {
  const playbackStopped = reviewPlaying && !autoplayStep;
  if (playbackStopped) stopReviewPlayback(false);
  if (!reviewActive || !reviewSnapshots.length || !Number.isFinite(position)) return;
  const nextPosition = Math.min(reviewSnapshots.length - 1, Math.max(0, Math.floor(position)));
  if (nextPosition === reviewCursor) {
    if (playbackStopped) render();
    return;
  }
  reviewCursor = nextPosition;
  state = snapshot(reviewSnapshots[reviewCursor]);
  render();
  scrollReviewMoveIntoView(reviewCursor);
  requestReviewDetail(reviewCursor);
}

function scrollReviewMoveIntoView(position: number): void {
  const strip = app!.querySelector<HTMLElement>('.review-move-strip');
  if (strip && position === 0) {
    strip.scrollTop = 0;
    return;
  }
  const move = strip?.querySelector<HTMLElement>(`.review-move-chip[data-review-position="${position}"]`);
  if (!strip || !move) return;
  const stripBounds = strip.getBoundingClientRect();
  const moveBounds = move.getBoundingClientRect();
  const moveTop = moveBounds.top - stripBounds.top + strip.scrollTop;
  const moveBottom = moveTop + moveBounds.height;
  if (moveTop < strip.scrollTop) strip.scrollTop = moveTop;
  else if (moveBottom > strip.scrollTop + strip.clientHeight) strip.scrollTop = moveBottom - strip.clientHeight;
}

function requestReviewDetail(positionIndex: number): void {
  reviewDetailToken += 1;
  const token = reviewDetailToken;
  reviewDetailAnalyzer.cancel();
  reviewDetailAnalysis = null;
  reviewDetailPosition = positionIndex;
  reviewDetailPending = false;
  reviewDetailFailed = false;
  const position = reviewSnapshots[positionIndex];
  if (!reviewActive || !position || position.winner !== null) {
    render();
    return;
  }
  reviewDetailPending = true;
  render();
  void reviewDetailAnalyzer.analyze(snapshot(position), 'analysis').then((analysis) => {
    if (token !== reviewDetailToken || !reviewActive || reviewCursor !== positionIndex) return;
    reviewDetailAnalysis = analysis;
    reviewDetailPending = false;
    render();
  }).catch((error: unknown) => {
    if (token !== reviewDetailToken || !reviewActive || error instanceof DOMException && error.name === 'AbortError') return;
    reviewDetailPending = false;
    reviewDetailFailed = true;
    render();
  });
}

function closeGameReview(): void {
  if (!reviewActive) return;
  stopReviewPlayback(false);
  analysisToken += 1;
  analyzer.cancel();
  reviewDetailToken += 1;
  reviewDetailAnalyzer.cancel();
  reviewPending = false;
  reviewActive = false;
  if (reviewFinalState) state = snapshot(reviewFinalState);
  reviewFinalState = null;
  reviewSnapshots = [];
  reviewMoves = [];
  reviewItems = [];
  reviewEvaluations = [];
  reviewCursor = 0;
  reviewUnavailable = 0;
  reviewHistoryAvailable = true;
  reviewDetailAnalysis = null;
  reviewDetailPosition = -1;
  reviewDetailPending = false;
  reviewDetailFailed = false;
  render();
}

async function runGameReview(token: number): Promise<void> {
  for (let index = 0; index < reviewSnapshots.length; index += 1) {
    if (token !== analysisToken || !reviewActive) return;
    if (reviewEvaluations[index]) continue;
    const position = reviewSnapshots[index];
    try {
      const analysis = await analyzer.analyze(snapshot(position), 'review');
      if (token !== analysisToken || !reviewActive) return;
      const xValue = analysis.value * (position.turn === 'X' ? 1 : -1);
      reviewEvaluations[index] = { xValue, source: analysis.source, solved: analysis.solved, lines: analysis.lines };
      const move = reviewMoves[index];
      const shouldExplainMove = move && (opponentMode !== 'computer' || move.player === humanSide);
      if (shouldExplainMove) {
        const best = analysis.lines[0]?.[0] ?? null;
        const playedAfter = play(move.before, Math.floor(move.move / 9), move.move % 9);
        const bestAfter = best === null ? null : play(move.before, Math.floor(best / 9), best % 9);
        if (playedAfter.winner !== move.player) {
          let kind: ReviewItem['kind'] = 'model';
          let note = best === move.move ? 'Matches the top model pick.' : 'Model preferred another move.';
          if (best !== move.move && missedImmediateWin(move.before, move.move)) {
            kind = 'tactical';
            note = 'Missed immediate win.';
          } else if (best !== move.move && hasImmediateWin(playedAfter) && bestAfter && !hasImmediateWin(bestAfter)) {
            kind = 'tactical';
            note = 'Allowed immediate win.';
          }
          reviewItems.push({ move, best, line: analysis.lines[0] ?? [], note, kind });
        }
      }
    } catch (error: unknown) {
      if (token !== analysisToken || error instanceof DOMException && error.name === 'AbortError') return;
      reviewUnavailable += 1;
    }
    render();
  }
  if (token === analysisToken && reviewActive) {
    reviewPending = false;
    render();
  }
}

function scheduleComputerMove(): void {
  if (reviewActive || !isComputerTurn() || computerThinking) return;
  computerThinking = true;
  computerStatus = '';
  computerSource = '';
  computerFailed = false;
  const token = ++computerRequestToken;
  render();
  void Promise.resolve().then(() => computer.chooseMove(snapshot(state), difficulty)).then(({ move, source }) => {
    if (token !== computerRequestToken || !isComputerTurn()) return;
    computerThinking = false;
    const selectedMove = Number.isInteger(move) && move !== null && move >= 0 && move < 81
      ? move
      : null;
    let board = selectedMove === null ? -1 : Math.floor(selectedMove / 9);
    let cell = selectedMove === null ? -1 : selectedMove % 9;
    if (selectedMove === null || !isLegal(state, board, cell)) {
      const legal = legalBoards(state).flatMap((candidateBoard) =>
        state.boards[candidateBoard].flatMap((mark, candidateCell) => mark === null ? [[candidateBoard, candidateCell] as const] : []),
      );
      if (!legal.length) {
        computerStatus = 'Computer could not find a move.';
        computerFailed = true;
        render();
        return;
      }
      [board, cell] = legal[0];
      source = 'search';
    }
    makeMove(board, cell, true, source);
  }).catch((error: unknown) => {
    if (token !== computerRequestToken) return;
    computerThinking = false;
    if (error instanceof DOMException && error.name === 'AbortError') return;
    computerSource = '';
    computerFailed = true;
    computerStatus = 'Computer could not move.';
    render();
  });
}

function retryComputerMove(): void {
  computerStatus = '';
  computerSource = '';
  computerFailed = false;
  scheduleComputerMove();
}

readOpponentPreference();
loadSavedGame();
render();
const inviteRoomCode = normalizeRoomCode(new URLSearchParams(window.location.search).get('room') ?? '');
if (isRoomCode(inviteRoomCode)) void joinOnlineGame(inviteRoomCode, true);
scheduleComputerMove();
