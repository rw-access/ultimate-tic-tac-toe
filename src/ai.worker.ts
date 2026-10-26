import { baselineEvaluation, legalMoves, search, type Evaluator } from './ai';
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

function sampleMove(state: GameState, logits: Float32Array, candidates?: number[]): number | null {
  const moves = candidates?.length ? candidates : legalMoves(state);
  if (!moves.length) return null;
  const winning = moves.find(move => play(state, Math.floor(move / 9), move % 9).winner === state.turn);
  if (winning !== undefined) return winning;
  const maximum = Math.max(...moves.map(move => logits[move]));
  const weights = moves.map(move => Math.exp((logits[move] - maximum) / 1.25));
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
        ? { maxEmptySquares: 18, maxNodes: 32_000, timeMs: 36 }
        : { maxEmptySquares: 18, maxNodes: 8_000, timeMs: 12 });
      if (endgame?.solved && endgame.move !== null) {
        self.postMessage({ id, value: endgame.value, lines: [[endgame.move]], source: 'search', solved: true });
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

    let evaluate = await loadEvaluator(modelUrl);
    const difficulty = event.data.difficulty ?? 'medium';
    const settings = difficulty === 'hard' ? { simulations: 4096, timeMs: 140, tacticalProof: true }
      : difficulty === 'medium' ? { simulations: 96, timeMs: 200, tacticalProof: true }
        : { simulations: 24, timeMs: 80, tacticalProof: false };
    let move: number | null;
    let simulations = 0;
    try {
      if (difficulty === 'easy' && evaluate) {
        const endgame = solveEndgame(state, { maxEmptySquares: 18, maxNodes: 8_000, timeMs: 12 });
        if (endgame?.solved && endgame.move !== null) {
          move = endgame.move;
        } else {
          const evaluation = await evaluate(state);
          const forcedWin = findForcedWinWithinPlies(state, {
            maxPlies: 7,
            maxNodes: 12_000,
            timeMs: 24,
            rootMoveScores: evaluation.policy,
          });
          if (forcedWin) move = forcedWin.move;
          else {
            const safeMoves = movesSafeFromImmediateLoss(state);
            move = sampleMove(state, evaluation.policy, safeMoves);
          }
        }
      } else {
        const result = await search(state, { ...settings, evaluate });
        move = result.move;
        simulations = result.simulations;
      }
    } catch {
      failed = true;
      await evaluator?.then(model => model.dispose()).catch(() => {});
      evaluator = null;
      evaluate = undefined;
      const result = await search(state, settings);
      move = result.move;
      simulations = result.simulations;
    }
    self.postMessage({ id, move, source: evaluate ? 'model' : 'search', simulations, settings });
  } catch {
    self.postMessage({ id, error: action === 'analyze' ? 'The computer could not analyze this position. Please try again.' : 'The computer could not choose a move. Please try again.' });
  }
};
