import { advanceSubtree, baselineEvaluation, legalMoves, search, type Evaluator, type Node } from './ai';
import { play, type GameState } from './game';
import { createEvaluator, type ModelEvaluator } from './onnx';
import { findForcedWinWithinPlies, immediateWinningMoves, movesSafeFromImmediateLoss, solveEndgame } from './endgame';
import type { AnalysisMode, Difficulty } from './ai-client';

type Request = {
  id: number;
  action: 'move' | 'analyze';
  state: GameState;
  difficulty?: Difficulty;
  analysisMode?: AnalysisMode;
  modelUrl: string;
};

let evaluator: Promise<ModelEvaluator> | null = null;
let failed = false;

let lastRoot: Node | undefined;
let lastAiStateAfterOurMove: GameState | undefined;
let lastAiMove: number | null = null;

function findOpponentMove(before: GameState, after: GameState): number | null {
  for (let b = 0; b < 9; b++) {
    for (let c = 0; c < 9; c++) {
      if (before.boards[b][c] === null && after.boards[b][c] !== null) {
        return b * 9 + c;
      }
    }
  }
  return null;
}

function sampleFromVisitDistribution(policy: Float32Array, temperature = 0.8): number | null {
  const candidates: number[] = [];
  const weights: number[] = [];
  const invTemp = 1 / Math.max(0.01, temperature);
  for (let move = 0; move < policy.length; move++) {
    const p = policy[move];
    if (p > 0) {
      candidates.push(move);
      weights.push(Math.pow(p, invTemp));
    }
  }
  if (!candidates.length) return null;
  if (candidates.length === 1) return candidates[0];

  const sum = weights.reduce((acc, w) => acc + w, 0);
  if (!(sum > 0) || !Number.isFinite(sum)) return candidates[0];

  let remaining = Math.random() * sum;
  for (let i = 0; i < candidates.length; i++) {
    remaining -= weights[i];
    if (remaining <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}

async function loadEvaluator(modelUrl: string): Promise<Evaluator | undefined> {
  if (failed) return undefined;
  try {
    evaluator ??= createEvaluator(modelUrl);
    return await evaluator;
  } catch {
    failed = true;
    evaluator = null;
    return undefined;
  }
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const { id, action, state, modelUrl } = event.data;
  try {
    if (action === 'analyze') {
      const mode = event.data.analysisMode ?? 'analysis';
      const moves = legalMoves(state);
      const immediateWin = immediateWinningMoves(state)[0];
      if (immediateWin !== undefined) {
        self.postMessage({ id, value: 1, lines: [[immediateWin]], source: 'search', solved: true });
        return;
      }
      if (moves.length > 0 && movesSafeFromImmediateLoss(state).length === 0) {
        self.postMessage({ id, value: -1, lines: [], source: 'search', solved: true });
        return;
      }

      const endgame = solveEndgame(state, mode === 'analysis'
        ? { maxEmptySquares: 18, maxNodes: 32_000, timeMs: 500 }
        : mode === 'hint'
          ? { maxEmptySquares: 18, maxNodes: 32_000, timeMs: 250 }
          : { maxEmptySquares: 18, maxNodes: 8_000, timeMs: 12 });
      // A retained draw is only a safe suggested move; unexplored root moves
      // could still win, so keep solved=false and display value 0 as an estimate.
      if (endgame && endgame.move !== null && (endgame.solved || endgame.guaranteedValue !== null)) {
        self.postMessage({ id, value: endgame.value, lines: [[endgame.move]], source: 'search', solved: endgame.solved });
        return;
      }

      let evaluate = await loadEvaluator(modelUrl);
      const simulations = mode === 'analysis' ? 512 : mode === 'review' ? 32 : 48;
      const timeMs = mode === 'analysis' ? 450 : mode === 'review' ? 50 : 100;
      let estimate;
      let result;
      try {
        estimate = await (evaluate ?? baselineEvaluation)(state);
        result = await search(state, { simulations, timeMs, evaluate, tacticalProof: true });
      } catch {
        failed = true;
        await evaluator?.then(model => model.dispose()).catch(() => {});
        evaluator = null;
        evaluate = undefined;
        estimate = await baselineEvaluation(state);
        result = await search(state, { simulations, timeMs, tacticalProof: true });
      }
      if (result.move !== null && result.simulations === 0) {
        self.postMessage({ id, value: 1, lines: [[result.move]], source: 'search', solved: true });
        return;
      }
      const rankedMoves = legalMoves(state).sort((left, right) => result.policy[right] - result.policy[left]);
      const roots = mode === 'analysis'
        ? [result.move, rankedMoves.find(move => move !== result.move)].filter((move): move is number => move !== null && move !== undefined)
        : [result.move].filter((move): move is number => move !== null && move !== undefined);
      const lines: number[][] = [];
      for (const firstMove of roots) {
        const line = [firstMove];
        let position = play(state, Math.floor(firstMove / 9), firstMove % 9);
        const lineDepth = mode === 'analysis' ? 7 : 1;
        for (let depth = 1; depth < lineDepth && position.winner === null; depth++) {
          const continuation = legalMoves(position);
          if (!continuation.length) break;
          let policy: Float32Array;
          try {
            policy = (await (evaluate ?? baselineEvaluation)(position)).policy;
          } catch {
            policy = (await baselineEvaluation(position)).policy;
          }
          let selected = continuation[0];
          for (const move of continuation) if (policy[move] > policy[selected]) selected = move;
          line.push(selected);
          position = play(position, Math.floor(selected / 9), selected % 9);
        }
        lines.push(line);
      }
      self.postMessage({
        id,
        value: estimate.value,
        lines,
        source: evaluate ? 'model' : 'search',
        solved: false,
        ownership: estimate.ownership ? Array.from(estimate.ownership) : undefined,
      });
      return;
    }

    const difficulty = event.data.difficulty ?? 'medium';
    const settings = difficulty === 'hard'
      ? { simulations: 384, timeMs: 220, tacticalProof: true }
      : difficulty === 'medium'
        ? { simulations: 64, timeMs: 120, tacticalProof: false }
        : { simulations: 16, timeMs: 60, tacticalProof: false };

    // Endgame minimax solver:
    // Only run on Hard (up to 14 squares) and Medium (shallow 8 squares).
    // Easy mode avoids the exact minimax endgame solver so casual players have fun.
    if (difficulty !== 'easy') {
      const maxEmpty = difficulty === 'hard' ? 14 : 8;
      const maxNodes = difficulty === 'hard' ? 16_000 : 1_000;
      const timeMs = difficulty === 'hard' ? 250 : 50;
      const endgame = solveEndgame(state, {
        maxEmptySquares: maxEmpty,
        maxNodes: maxNodes,
        timeMs: timeMs,
      });
      if (endgame && endgame.move !== null && (endgame.solved || endgame.guaranteedValue !== null)) {
        lastRoot = undefined;
        lastAiMove = null;
        lastAiStateAfterOurMove = undefined;
        self.postMessage({ id, move: endgame.move, source: 'search', simulations: 0, settings });
        return;
      }
    }

    // Subtree carryover:
    // Exclusive to Hard difficulty so Hard reaches maximum search depth,
    // while Medium stays accessible with independent per-turn search.
    let reusedRoot: Node | undefined;
    if (difficulty === 'hard' && lastRoot && lastAiStateAfterOurMove && lastAiMove !== null && state.moves > 1) {
      const oppMove = findOpponentMove(lastAiStateAfterOurMove, state);
      if (oppMove !== null) {
        reusedRoot = advanceSubtree(lastRoot, lastAiMove, oppMove);
      }
    }

    let evaluate = await loadEvaluator(modelUrl);
    let move: number | null;
    let simulations = 0;
    try {
      const result = await search(state, { ...settings, evaluate, reusedRoot });
      simulations = result.simulations;
      if (difficulty === 'easy' && result.move !== null && result.simulations > 0) {
        move = sampleFromVisitDistribution(result.policy, 0.8) ?? result.move;
      } else {
        move = result.move;
      }

      if (move !== null && difficulty === 'hard') {
        lastRoot = result.root;
        lastAiMove = move;
        lastAiStateAfterOurMove = play(state, Math.floor(move / 9), move % 9);
      } else {
        lastRoot = undefined;
        lastAiMove = null;
        lastAiStateAfterOurMove = undefined;
      }
    } catch {
      failed = true;
      await evaluator?.then(model => model.dispose()).catch(() => {});
      evaluator = null;
      evaluate = undefined;
      const result = await search(state, { ...settings, reusedRoot });
      simulations = result.simulations;
      if (difficulty === 'easy' && result.move !== null && result.simulations > 0) {
        move = sampleFromVisitDistribution(result.policy, 0.8) ?? result.move;
      } else {
        move = result.move;
      }

      if (move !== null && difficulty === 'hard') {
        lastRoot = result.root;
        lastAiMove = move;
        lastAiStateAfterOurMove = play(state, Math.floor(move / 9), move % 9);
      } else {
        lastRoot = undefined;
        lastAiMove = null;
        lastAiStateAfterOurMove = undefined;
      }
    }
    self.postMessage({ id, move, source: evaluate ? 'model' : 'search', simulations, settings });
  } catch {
    self.postMessage({ id, error: action === 'analyze' ? 'The computer could not analyze this position. Please try again.' : 'The computer could not choose a move. Please try again.' });
  }
};
