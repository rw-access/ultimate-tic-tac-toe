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

function sampleMove(state: GameState, logits: Float32Array, candidates?: number[], temperature = 1.25): number | null {
  const moves = candidates?.length ? candidates : legalMoves(state);
  if (!moves.length) return null;
  const winning = moves.find(move => play(state, Math.floor(move / 9), move % 9).winner === state.turn);
  if (winning !== undefined) return winning;
  const maximum = Math.max(...moves.map(move => logits[move]));
  const weights = moves.map(move => Math.exp((logits[move] - maximum) / temperature));
  let remaining = Math.random() * weights.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < moves.length; i++) {
    remaining -= weights[i];
    if (remaining <= 0) return moves[i];
  }
  return moves[moves.length - 1];
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
      self.postMessage({ id, value: estimate.value, lines, source: evaluate ? 'model' : 'search', solved: false });
      return;
    }

    const difficulty = event.data.difficulty ?? 'medium';
    const settings = difficulty === 'hard'
      ? { simulations: 4096, timeMs: 140, tacticalProof: true }
      : difficulty === 'medium'
        ? { simulations: 48, timeMs: 100, tacticalProof: true }
        : { simulations: 16, timeMs: 50, tacticalProof: false };

    // Endgame minimax solver:
    // Only run on Hard (full 18 squares) and Medium (shallow 8 squares).
    // Easy mode avoids the exact minimax endgame solver so casual players have fun.
    if (difficulty !== 'easy') {
      const maxEmpty = difficulty === 'hard' ? 18 : 8;
      const maxNodes = difficulty === 'hard' ? 32_000 : 2_000;
      const timeMs = difficulty === 'hard' ? 2_000 : 150;
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
      if (difficulty === 'easy') {
        lastRoot = undefined;
        lastAiMove = null;
        lastAiStateAfterOurMove = undefined;

        // Take an immediate 1-move win if available right now.
        const immediateWins = immediateWinningMoves(state);
        if (immediateWins.length > 0) {
          move = immediateWins[0];
        } else if (evaluate) {
          const evaluation = await evaluate(state);
          // 75% of the time play safe moves with relaxed temperature; 25% allow loose play
          const safeMoves = movesSafeFromImmediateLoss(state);
          const candidates = (safeMoves.length > 0 && Math.random() < 0.75) ? safeMoves : legalMoves(state);
          move = sampleMove(state, evaluation.policy, candidates, 2.0);
        } else {
          const safeMoves = movesSafeFromImmediateLoss(state);
          const candidates = safeMoves.length > 0 && Math.random() < 0.75 ? safeMoves : legalMoves(state);
          move = candidates[Math.floor(Math.random() * candidates.length)] ?? null;
        }
      } else {
        const result = await search(state, { ...settings, evaluate, reusedRoot });
        move = result.move;
        simulations = result.simulations;
        if (result.move !== null && difficulty === 'hard') {
          lastRoot = result.root;
          lastAiMove = result.move;
          lastAiStateAfterOurMove = play(state, Math.floor(result.move / 9), result.move % 9);
        } else {
          lastRoot = undefined;
          lastAiMove = null;
          lastAiStateAfterOurMove = undefined;
        }
      }
    } catch {
      failed = true;
      await evaluator?.then(model => model.dispose()).catch(() => {});
      evaluator = null;
      evaluate = undefined;
      const result = await search(state, { ...settings, reusedRoot });
      move = result.move;
      simulations = result.simulations;
      if (result.move !== null && difficulty === 'hard') {
        lastRoot = result.root;
        lastAiMove = result.move;
        lastAiStateAfterOurMove = play(state, Math.floor(result.move / 9), result.move % 9);
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
