import { play } from './game.ts';
import type { GameState, Player } from './game.ts';

export const INPUT_SIZE = 280;
const ACTIONS = 81;
const MAX_SIMULATIONS = 10_000;
const MAX_TIME_MS = 30_000;

export interface Evaluation {
  /** Unnormalized move scores. Invalid and illegal actions are ignored. */
  policy: Float32Array;
  /** Position value from the perspective of state.turn, in [-1, 1]. */
  value: number;
  /** Predicted final mini-board ownership in [-1, 1] for each of the 9 mini-boards. */
  ownership?: Float32Array;
}

export type Evaluator = (state: GameState) => Promise<Evaluation>;

export interface RootNoiseOptions {
  /** Fraction of the root prior replaced by a fresh Dirichlet sample. */
  epsilon: number;
  /** Total Dirichlet concentration, divided evenly across legal root moves. */
  concentrationMass: number;
}

export function legalMoves(state: GameState): number[] {
  if (state.winner) return [];
  const moves: number[] = [];
  const forcedBoard = state.forced !== null && state.results[state.forced] === null
    ? state.forced
    : null;
  for (let board = 0; board < 9; board++) {
    if (forcedBoard !== null ? board !== forcedBoard : state.results[board] !== null) continue;
    const cells = state.boards[board];
    const offset = board * 9;
    for (let cell = 0; cell < 9; cell++) {
      if (cells[cell] === null) moves.push(offset + cell);
    }
  }
  return moves;
}

/**
 * Encode a position from the side-to-move perspective.
 * Feature groups: own marks, opponent marks, legal cells, own mini-board wins,
 * opponent mini-board wins, drawn mini-boards, and forced-board one-hot (null=9).
 */
export function encodeState(state: GameState): Float32Array {
  const input = new Float32Array(INPUT_SIZE);
  const opponent: Player = state.turn === 'X' ? 'O' : 'X';
  for (let board = 0; board < 9; board++) {
    for (let cell = 0; cell < 9; cell++) {
      const mark = state.boards[board][cell];
      const index = board * 9 + cell;
      if (mark === state.turn) input[index] = 1;
      else if (mark === opponent) input[81 + index] = 1;
    }
  }
  for (const move of legalMoves(state)) input[162 + move] = 1;
  for (let board = 0; board < 9; board++) {
    const status = state.results[board];
    if (status === state.turn) input[243 + board] = 1;
    else if (status === opponent) input[252 + board] = 1;
    else if (status === 'draw') input[261 + board] = 1;
  }
  input[270 + (state.forced ?? 9)] = 1;
  return input;
}

const lines = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

function terminalValue(state: GameState): number {
  if (state.winner === 'draw') return 0;
  if (state.winner === 'X' || state.winner === 'O') return state.winner === state.turn ? 1 : -1;
  return 0;
}

/** Find canonical one-move global wins, skipping states without a macro threat. */
function immediateWinningMoves(state: GameState, moves = legalMoves(state)): number[] {
  if (state.winner !== null) return [];
  let hasMacroThreat = false;
  for (const [a, b, c] of lines) {
    let own = 0;
    let open = 0;
    for (const board of [a, b, c]) {
      const result = state.results[board];
      if (result === state.turn) own++;
      else if (result === null) open++;
      else break;
    }
    if (own === 2 && open === 1) {
      hasMacroThreat = true;
      break;
    }
  }
  if (!hasMacroThreat) return [];

  const wins: number[] = [];
  for (const move of moves) {
    const next = play(state, Math.floor(move / 9), move % 9);
    if (next.winner === state.turn) wins.push(move);
  }
  return wins;
}

/** Fast necessary condition for whether the opponent could win on the next ply. */
function hasOpponentImmediateThreat(state: GameState): boolean {
  if (state.winner !== null) return false;
  const opponent: Player = state.turn === 'X' ? 'O' : 'X';
  for (const [a, b, c] of lines) {
    let opp = 0;
    let openBoard = -1;
    for (const board of [a, b, c]) {
      const result = state.results[board];
      if (result === opponent) opp++;
      else if (result === null) openBoard = board;
      else break;
    }
    if (opp === 2 && openBoard !== -1) {
      const cells = state.boards[openBoard];
      for (const [x, y, z] of lines) {
        const c0 = cells[x];
        const c1 = cells[y];
        const c2 = cells[z];
        const oppMarks = Number(c0 === opponent) + Number(c1 === opponent) + Number(c2 === opponent);
        const emptyMarks = Number(c0 === null) + Number(c1 === null) + Number(c2 === null);
        if (oppMarks === 2 && emptyMarks === 1) return true;
      }
    }
  }
  return false;
}

function filterSafeMoves(state: GameState, moves: number[]): number[] {
  if (!hasOpponentImmediateThreat(state)) return moves;
  const safeMoves: number[] = [];
  for (const move of moves) {
    const next = play(state, Math.floor(move / 9), move % 9);
    if (next.winner !== null || immediateWinningMoves(next).length === 0) safeMoves.push(move);
  }
  return safeMoves;
}

type CellResult = Player | 'draw' | null;

function linePotential(
  cells: readonly CellResult[],
  player: Player,
  changedIndex = -1,
  changedValue: CellResult = null,
): number {
  let score = 0;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const a = line[0];
    const b = line[1];
    const c = line[2];
    const first = a === changedIndex ? changedValue : cells[a];
    const second = b === changedIndex ? changedValue : cells[b];
    const third = c === changedIndex ? changedValue : cells[c];
    if ((first !== null && first !== player) ||
        (second !== null && second !== player) ||
        (third !== null && third !== player)) continue;
    const count = Number(first === player) + Number(second === player) + Number(third === player);
    if (count === 1) score += 0.08;
    else if (count === 2) score += 0.38;
    else if (count === 3) score += 1;
  }
  return score;
}

function resultAfterReplacement(cells: readonly CellResult[], index: number, value: CellResult): CellResult {
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex];
    const a = line[0];
    const b = line[1];
    const c = line[2];
    const first = a === index ? value : cells[a];
    const second = b === index ? value : cells[b];
    const third = c === index ? value : cells[c];
    if ((first === 'X' || first === 'O') && first === second && first === third) return first;
  }
  for (let cell = 0; cell < cells.length; cell++) {
    if ((cell === index ? value : cells[cell]) === null) return null;
  }
  return 'draw';
}

function baselineValueAfterMove(
  state: GameState,
  board: number,
  cell: number,
  mark: Player,
  localResult: CellResult,
  nextForced: number | null,
  localValueForNextPlayer: Float64Array,
): number {
  const nextPlayer: Player = mark === 'X' ? 'O' : 'X';
  let score = linePotential(state.results, nextPlayer, board, localResult) -
    linePotential(state.results, mark, board, localResult);
  for (let index = 0; index < 9; index++) {
    const result = index === board ? localResult : state.results[index];
    if (result !== null) continue;
    const weight = nextForced === index ? 1.35 : 0.7;
    const localValue = index === board
      ? linePotential(state.boards[index], nextPlayer, cell, mark) -
        linePotential(state.boards[index], mark, cell, mark)
      : localValueForNextPlayer[index];
    score += weight * localValue;
  }
  return Math.tanh(score / 3.5);
}

/** Small deterministic evaluator used until a learned ONNX model is supplied. */
export function baselineEvaluation(state: GameState): Evaluation {
  const moves = legalMoves(state);
  const policy = new Float32Array(ACTIONS);
  if (state.winner !== null) return { policy, value: terminalValue(state) };

  const opponent: Player = state.turn === 'X' ? 'O' : 'X';
  let positionScore = linePotential(state.results, state.turn) - linePotential(state.results, opponent);
  // Candidate moves only alter one mini-board. Cache the next player's value
  // for every untouched board so each candidate needs to score just that board.
  const localValueForNextPlayer = new Float64Array(9);
  for (let board = 0; board < 9; board++) {
    if (state.results[board] !== null) continue;
    const weight = state.forced === board ? 1.35 : 0.7;
    const currentValue = linePotential(state.boards[board], state.turn);
    const opponentValue = linePotential(state.boards[board], opponent);
    localValueForNextPlayer[board] = opponentValue - currentValue;
    positionScore += weight * (currentValue - opponentValue);
  }
  const value = Math.tanh(positionScore / 3.5);

  for (const move of moves) {
    const board = (move / 9) | 0;
    const cell = move - board * 9;
    const localResult = resultAfterReplacement(state.boards[board], cell, state.turn);
    const nextWinner = resultAfterReplacement(state.results, board, localResult);
    if (nextWinner === state.turn) {
      policy[move] = 100;
      continue;
    }
    const destinationResult = cell === board ? localResult : state.results[cell];
    const nextForced = destinationResult === null ? cell : null;
    const afterValue = nextWinner === null
      ? baselineValueAfterMove(state, board, cell, state.turn, localResult, nextForced, localValueForNextPlayer)
      : nextWinner === 'draw' ? 0 : nextWinner === opponent ? 1 : -1;
    const wonMiniBoard = localResult === state.turn;
    // Convert the after-state value back to the moving player's perspective.
    policy[move] = -afterValue + (wonMiniBoard ? 1.8 : 0);
    // Prefer central squares slightly when the strategic scores are otherwise equal.
    if (cell === 4) policy[move] += 0.025;
  }
  return { policy, value };
}

export interface Child {
  move: number;
  prior: number;
  node: Node | null;
}

export interface Node {
  state: GameState;
  visits: number;
  valueSum: number;
  children: Child[] | null;
  priors: Float32Array | null;
  proven: 1 | -1 | 0;
}

export function statesMatch(a: GameState, b: GameState): boolean {
  if (a.turn !== b.turn || a.forced !== b.forced || a.winner !== b.winner) return false;
  for (let i = 0; i < 9; i++) {
    if (a.results[i] !== b.results[i]) return false;
    const bA = a.boards[i], bB = b.boards[i];
    for (let j = 0; j < 9; j++) {
      if (bA[j] !== bB[j]) return false;
    }
  }
  return true;
}

export function advanceSubtree(root: Node | undefined, ourMove: number, opponentMove: number): Node | undefined {
  if (!root || !root.children) return undefined;
  const ourChild = root.children.find(c => c.move === ourMove);
  if (!ourChild?.node || !ourChild.node.children) return undefined;
  const oppChild = ourChild.node.children.find(c => c.move === opponentMove);
  return oppChild?.node ?? undefined;
}

function softmaxLegal(policy: Float32Array | undefined, moves: number[]): Map<number, number> {
  if (!moves.length) return new Map();
  let max = -Infinity;
  for (const move of moves) {
    const score = policy?.[move];
    if (Number.isFinite(score)) max = Math.max(max, score!);
  }
  if (!Number.isFinite(max)) return new Map(moves.map(move => [move, 1 / moves.length]));
  const weights = moves.map(move => {
    const score = policy?.[move];
    return Number.isFinite(score) ? Math.exp(Math.max(-30, score! - max)) : 0;
  });
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!(total > 0) || !Number.isFinite(total)) return new Map(moves.map(move => [move, 1 / moves.length]));
  return new Map(moves.map((move, index) => [move, weights[index] / total]));
}

function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function chooseChild(node: Node, random: () => number, tacticalProof = false, fpuReduction?: number): Child {
  const allChildren = node.children!;
  if (tacticalProof) {
    for (const child of allChildren) {
      if (child.node?.proven === -1) return child;
    }
  }
  const nonLosing = tacticalProof
    ? allChildren.filter(child => child.node?.proven !== 1)
    : allChildren;
  const children = nonLosing.length > 0 ? nonLosing : allChildren;
  const exploration = Math.SQRT2;
  const parentQ = node.visits > 0 ? node.valueSum / node.visits : 0;
  const defaultUnvisitedQ = fpuReduction !== undefined
    ? Math.max(-1, Math.min(1, parentQ - fpuReduction))
    : 0;
  let bestScore = -Infinity;
  let best: Child[] = [];
  for (const child of children) {
    const childNode = child.node;
    const visits = childNode?.visits ?? 0;
    const q = visits === 0 ? defaultUnvisitedQ : -childNode!.valueSum / visits;
    const u = exploration * child.prior * Math.sqrt(Math.max(1, node.visits)) / (1 + visits);
    const score = q + u;
    if (score > bestScore + 1e-12) {
      bestScore = score;
      best = [child];
    } else if (Math.abs(score - bestScore) <= 1e-12) best.push(child);
  }
  return best[Math.floor(random() * best.length)];
}

function openUnit(random: () => number): number {
  return Math.max(Number.MIN_VALUE, Math.min(1 - Number.EPSILON, random()));
}

function normalSample(random: () => number): number {
  return Math.sqrt(-2 * Math.log(openUnit(random))) * Math.cos(2 * Math.PI * random());
}

/** Return log(Gamma(shape, 1)) to keep small-shape samples numerically stable. */
function logGammaSample(shape: number, random: () => number): number {
  if (shape < 1) {
    return logGammaSample(shape + 1, random) + Math.log(openUnit(random)) / shape;
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    const normal = normalSample(random);
    const base = 1 + c * normal;
    if (base <= 0) continue;
    const value = base ** 3;
    const uniform = openUnit(random);
    if (uniform < 1 - 0.0331 * normal ** 4 ||
        Math.log(uniform) < 0.5 * normal ** 2 + d * (1 - value + Math.log(value))) {
      return Math.log(d) + Math.log(value);
    }
  }
}

function sampleDirichlet(
  moves: number[],
  concentrationMass: number,
  random: () => number,
): Map<number, number> {
  const sample = new Map<number, number>();
  const alpha = concentrationMass / moves.length;
  // As alpha tends to zero, Dirichlet(alpha, ..., alpha) tends to a uniformly
  // chosen simplex vertex. Use that limit rather than underflowing Gamma draws.
  if (alpha < 1e-5) {
    const selected = Math.floor(random() * moves.length);
    for (let index = 0; index < moves.length; index++) sample.set(moves[index], index === selected ? 1 : 0);
    return sample;
  }

  const logWeights = moves.map(() => logGammaSample(alpha, random));
  const maximum = Math.max(...logWeights);
  const weights = logWeights.map(value => Math.exp(value - maximum));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!(total > 0) || !Number.isFinite(total)) {
    const selected = Math.floor(random() * moves.length);
    for (let index = 0; index < moves.length; index++) sample.set(moves[index], index === selected ? 1 : 0);
    return sample;
  }
  for (let index = 0; index < moves.length; index++) sample.set(moves[index], weights[index] / total);
  return sample;
}

function validateRootNoise(options: RootNoiseOptions | undefined): RootNoiseOptions | undefined {
  if (!options) return undefined;
  if (!Number.isFinite(options.epsilon) || options.epsilon < 0 || options.epsilon > 1) {
    throw new RangeError('root-noise epsilon must be finite and between 0 and 1');
  }
  if (!Number.isFinite(options.concentrationMass) || options.concentrationMass <= 0) {
    throw new RangeError('root-noise concentration mass must be finite and positive');
  }
  return options;
}

async function evaluate(evaluator: Evaluator | undefined, state: GameState): Promise<Evaluation> {
  const evaluation = evaluator ? await evaluator(state) : baselineEvaluation(state);
  const policy = evaluation?.policy instanceof Float32Array ? evaluation.policy : new Float32Array(ACTIONS);
  const value = Number.isFinite(evaluation?.value) ? Math.max(-1, Math.min(1, evaluation.value)) : 0;
  return { policy, value };
}

export async function search(
  state: GameState,
  options: {
    simulations?: number;
    timeMs?: number;
    evaluate?: Evaluator;
    seed?: number;
    tacticalProof?: boolean;
    rootNoise?: RootNoiseOptions;
    fpuReduction?: number;
    reusedRoot?: Node;
  } = {},
): Promise<{ move: number | null; policy: Float32Array; simulations: number; root?: Node }> {
  const rootNoise = validateRootNoise(options.rootNoise);
  let rootMoves = legalMoves(state);
  const policy = new Float32Array(ACTIONS);
  if (!rootMoves.length) return { move: null, policy, simulations: 0 };

  const tacticalProof = options.tacticalProof === true;
  if (tacticalProof) {
    // Preserve the immediate-win priority before considering defensive filtering.
    const winningMoves = immediateWinningMoves(state, rootMoves);
    if (winningMoves.length) {
      const move = winningMoves[0];
      policy[move] = 1;
      return { move, policy, simulations: 0 };
    }
    const safeMoves = filterSafeMoves(state, rootMoves);
    // If defeat cannot be prevented in one move, keep the full legal action set.
    if (safeMoves.length > 0) rootMoves = safeMoves;
  } else {
    // Preserve legacy behavior unless callers explicitly enable tactical proofs.
    for (const move of rootMoves) {
      const next = play(state, Math.floor(move / 9), move % 9);
      if (next.winner === state.turn) {
        policy[move] = 1;
        return { move, policy, simulations: 0 };
      }
    }
  }

  const simulationsLimit = Math.max(1, Math.min(MAX_SIMULATIONS, Math.floor(options.simulations ?? 128)));
  const timeLimit = Math.max(0, Math.min(MAX_TIME_MS, options.timeMs ?? 1_000));
  const deadline = performance.now() + timeLimit;
  const searchRandom = makeRng(options.seed ?? 1);
  // Keep noise sampling on its own stream so enabling it cannot perturb the
  // legacy PUCT tie-break sequence.
  const noiseRandom = rootNoise && rootNoise.epsilon > 0
    ? makeRng((options.seed ?? 1) ^ 0x6d2b79f5)
    : undefined;
  const canReuse = options.reusedRoot !== undefined && statesMatch(options.reusedRoot.state, state);
  const root: Node = canReuse
    ? options.reusedRoot!
    : { state, visits: 0, valueSum: 0, children: null, priors: null, proven: 0 };
  const path: Node[] = [];
  let simulations = root.visits;

  for (let simulation = simulations; simulation < simulationsLimit && performance.now() <= deadline; simulation++) {
    path.length = 0;
    let node = root;
    path.push(node);
    while (node.children !== null && node.children.length > 0 && (!tacticalProof || node.proven === 0)) {
      const selected = chooseChild(node, searchRandom, tacticalProof, options.fpuReduction);
      if (selected.node === null) {
        selected.node = {
          state: play(node.state, Math.floor(selected.move / 9), selected.move % 9),
          visits: 0,
          valueSum: 0,
          children: null,
          priors: null,
          proven: 0,
        };
      }
      node = selected.node;
      path.push(node);
    }

    let value: number;
    if (tacticalProof && node.proven !== 0) {
      value = node.proven;
    } else if (node.state.winner !== null) {
      value = terminalValue(node.state);
      if (tacticalProof) {
        if (value === 1) node.proven = 1;
        else if (value === -1) node.proven = -1;
      }
    } else {
      let moves = node === root ? rootMoves : legalMoves(node.state);
      if (!moves.length) value = 0;
      else {
        const winningMoves = tacticalProof && node !== root
          ? immediateWinningMoves(node.state, moves)
          : [];
        if (winningMoves.length) {
          const winning = new Set(winningMoves);
          const prior = 1 / winningMoves.length;
          node.priors = new Float32Array(ACTIONS);
          node.children = moves.map(move => {
            const movePrior = winning.has(move) ? prior : 0;
            node.priors![move] = movePrior;
            return { move, prior: movePrior, node: null };
          });
          // This node can force an immediate win, from its current-player view.
          node.proven = 1;
          value = 1;
        } else {
          const safeMoves = tacticalProof && node !== root
            ? filterSafeMoves(node.state, moves)
            : moves;
          if (tacticalProof && node !== root && safeMoves.length === 0) {
            node.proven = -1;
            value = -1;
          } else {
            if (safeMoves.length > 0) moves = safeMoves;
            const evaluation = await evaluate(options.evaluate, node.state);
            const priors = softmaxLegal(evaluation.policy, moves);
            if (node === root && rootNoise && noiseRandom) {
              const noise = sampleDirichlet(moves, rootNoise.concentrationMass, noiseRandom);
              for (const move of moves) {
                const prior = priors.get(move) ?? 0;
                const noisePrior = noise.get(move) ?? 0;
                priors.set(move, (1 - rootNoise.epsilon) * prior + rootNoise.epsilon * noisePrior);
              }
            }
            node.priors = new Float32Array(ACTIONS);
            node.children = moves.map(move => {
              const prior = priors.get(move) ?? 0;
              node.priors![move] = prior;
              return { move, prior, node: null };
            });
            value = evaluation.value;
          }
        }
      }
    }

    // value is expressed for the leaf's player to move. Flip perspective at
    // every parent edge so each node accumulates values for its own player.
    for (let index = path.length - 1; index >= 0; index--) {
      const visited = path[index];
      if (tacticalProof && visited.proven === 0 && visited.children !== null && visited.children.length > 0) {
        if (visited.children.some(child => child.node?.proven === -1)) {
          visited.proven = 1;
          value = 1;
        } else if (visited.children.every(child => child.node?.proven === 1)) {
          visited.proven = -1;
          value = -1;
        }
      }
      visited.visits++;
      visited.valueSum += value;
      value = -value;
    }
    simulations++;
  }

  const rootChildren = root.children ?? [];
  if (tacticalProof) {
    const provenWinChild = rootChildren.find(child => child.node?.proven === -1);
    if (provenWinChild) {
      policy[provenWinChild.move] = 1;
      return { move: provenWinChild.move, policy, simulations, root };
    }
  }

  const activeChildren = tacticalProof
    ? (() => {
        const surviving = rootChildren.filter(child => child.node?.proven !== 1);
        return surviving.length > 0 ? surviving : rootChildren;
      })()
    : rootChildren;
  const visitTotal = activeChildren.reduce((sum, child) => sum + (child.node?.visits ?? 0), 0);
  if (visitTotal > 0) {
    for (const child of activeChildren) policy[child.move] = (child.node?.visits ?? 0) / visitTotal;
  } else if (root.priors) {
    if (tacticalProof && activeChildren.length < rootChildren.length) {
      const priorSum = activeChildren.reduce((sum, child) => sum + (root.priors![child.move] ?? 0), 0);
      for (const child of activeChildren) {
        policy[child.move] = priorSum > 0 ? root.priors![child.move] / priorSum : 1 / activeChildren.length;
      }
    } else {
      policy.set(root.priors);
    }
  } else {
    for (const move of rootMoves) policy[move] = 1 / rootMoves.length;
  }

  if (tacticalProof && activeChildren.length > 0) {
    let bestChild = activeChildren[0];
    for (const candidate of activeChildren) {
      const candScore = policy[candidate.move];
      const bestScore = policy[bestChild.move];
      if (candScore > bestScore + 1e-12) {
        bestChild = candidate;
      } else if (Math.abs(candScore - bestScore) <= 1e-12) {
        const candVisits = candidate.node?.visits ?? 0;
        const bestVisits = bestChild.node?.visits ?? 0;
        if (candVisits > 0 && bestVisits > 0) {
          const candQ = -candidate.node!.valueSum / candVisits;
          const bestQ = -bestChild.node!.valueSum / bestVisits;
          if (candQ > bestQ + 1e-12 || (Math.abs(candQ - bestQ) <= 1e-12 && candidate.prior > bestChild.prior + 1e-12)) {
            bestChild = candidate;
          }
        } else if (candidate.prior > bestChild.prior + 1e-12) {
          bestChild = candidate;
        }
      }
    }
    return { move: bestChild.move, policy, simulations, root };
  }

  let move = rootMoves[0];
  for (const candidate of rootMoves) if (policy[candidate] > policy[move]) move = candidate;
  return { move, policy, simulations, root };
}
