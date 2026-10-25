#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { access, mkdir, open, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createGame, play } from '../src/game.ts';
import type { GameState, Player } from '../src/game.ts';
import { baselineEvaluation, encodeState, legalMoves, search } from '../src/ai.ts';
import type { Evaluation, Evaluator, RootNoiseOptions } from '../src/ai.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_SEED = 20261024;
const SEARCH_TIME_MS = 30_000;

type Options = Record<string, string>;
type Command = 'generate' | 'evaluate' | 'parity' | 'arena';
type WinnerCode = 1 | -1 | 2;
type ValueMode = 'model' | 'zero' | 'heuristic';
interface SearchCapExperiment {
  fastSimulations: number;
  fullSearchProbability: number;
}
const MAX_INFERENCE_BATCH = 32;

function parseCli(argv: string[]): { command: Command; options: Options } {
  const [command, ...args] = argv;
  if (command === '--help' || command === '-h') {
    printHelp();
    process.exit(0);
  }
  if (command !== 'generate' && command !== 'evaluate' && command !== 'parity' && command !== 'arena') {
    throw new Error('Usage: node --experimental-strip-types scripts/ai.ts generate|evaluate|parity|arena [options]');
  }
  const options: Options = {};
  for (let index = 0; index < args.length; index++) {
    const token = args[index];
    if (token === '--help' || token === '-h') {
      printHelp();
      process.exit(0);
    }
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    if (command === 'arena' && token === '--tactical-proof') {
      if (options['tactical-proof'] !== undefined) throw new Error('Option --tactical-proof was provided more than once');
      options['tactical-proof'] = 'true';
      continue;
    }
    const equal = token.indexOf('=');
    const key = token.slice(2, equal < 0 ? undefined : equal);
    const value = equal < 0 ? args[++index] : token.slice(equal + 1);
    if (!key || value === undefined || value.startsWith('--')) throw new Error(`Expected a value for ${token}`);
    if (options[key] !== undefined) throw new Error(`Option --${key} was provided more than once`);
    options[key] = value;
  }
  const allowed: Record<Command, string[]> = {
    generate: ['games', 'simulations', 'seed', 'output', 'model', 'parallel-games', 'teacher-mix', 'root-noise', 'root-noise-mass', 'fast-simulations', 'full-search-probability'],
    evaluate: ['model', 'games-per-seat', 'simulations', 'seed', 'opening-moves', 'metadata'],
    parity: ['model', 'output', 'seed'],
    arena: ['challenger', 'champion', 'pairs', 'pair-offset', 'pair-count', 'simulations', 'seed', 'parallel-games', 'output', 'challenger-value', 'champion-value', 'tactical-proof'],
  };
  for (const key of Object.keys(options)) {
    if (!allowed[command].includes(key)) throw new Error(`Unknown option for ${command}: --${key}`);
  }
  return { command, options };
}

function printHelp(): void {
  process.stdout.write(`Usage: node --experimental-strip-types scripts/ai.ts <command> [options]\n\n` +
    `Commands:\n` +
    `  generate  Create TypeScript MCTS self-play JSONL\n` +
    `  evaluate  Compare raw model and model-guided search against baselines\n` +
    `  parity    Write TypeScript state/encoder/ONNX parity cases\n` +
    `  arena     Compare challenger and champion in paired, seat-swapped games\n\n` +
    `Generate options: --games 1000 --simulations 24 --seed ${DEFAULT_SEED} ` +
    `--output training/data/baseline.jsonl [--model path] [--parallel-games n] [--teacher-mix 0..1] ` +
    `[--root-noise 0..1 --root-noise-mass positive] ` +
    `[--fast-simulations n --full-search-probability 0<p<=1]\n` +
    `Evaluate options: --model path --games-per-seat 10 --simulations 32 ` +
    `--seed 20262024 [--opening-moves 2] [--metadata path]\n` +
    `Parity options: --model path --output training/parity.json --seed 8901\n` +
    `Arena options: --challenger path --champion path --pairs 10 --simulations 64 ` +
    `--seed 20262024 --parallel-games 16 --output training/arena/result.json ` +
    `[--challenger-value model|zero|heuristic] [--champion-value model|zero|heuristic] [--tactical-proof]\n`);
}

function intOption(options: Options, name: string, fallback: number, minimum = 0): number {
  const raw = options[name];
  if (raw === undefined) return fallback;
  if (!/^-?\d+$/.test(raw)) throw new Error(`--${name} must be an integer`);
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${name} must be an integer >= ${minimum}`);
  return value;
}

function stringOption(options: Options, name: string, fallback: string): string {
  const value = options[name];
  return value === undefined ? fallback : value;
}

function choiceOption<T extends string>(options: Options, name: string, fallback: T, choices: readonly T[]): T {
  const value = options[name] ?? fallback;
  if (!choices.includes(value as T)) {
    throw new Error(`--${name} must be one of: ${choices.join(', ')}`);
  }
  return value as T;
}

function booleanOption(options: Options, name: string, fallback: boolean): boolean {
  const value = options[name];
  if (value === undefined) return fallback;
  if (value === 'true') return true;
  if (value === 'false') return false;
  throw new Error(`--${name} must be a boolean`);
}

function fractionOption(options: Options, name: string, fallback: number): number {
  const raw = options[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`--${name} must be a number between 0 and 1`);
  }
  return value;
}

function positiveNumberOption(options: Options, name: string, fallback: number): number {
  const raw = options[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`--${name} must be a finite number greater than 0`);
  }
  return value;
}

function requiredPath(options: Options, name: string): string {
  const value = options[name];
  if (!value) throw new Error(`--${name} is required`);
  return resolve(process.cwd(), value);
}

function optionalPath(options: Options, name: string): string | undefined {
  const value = options[name];
  return value ? resolve(process.cwd(), value) : undefined;
}

function repoRelative(filePath: string): string {
  return relative(ROOT, filePath).split(sep).join('/') || '.';
}

function modelRelative(filePath: string | undefined): string | null {
  return filePath ? repoRelative(filePath) : null;
}

function isInsideRoot(filePath: string): boolean {
  const rel = relative(ROOT, filePath);
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel));
}

async function requireFile(filePath: string, label: string): Promise<void> {
  try {
    await access(filePath);
  } catch {
    throw new Error(`${label} does not exist: ${repoRelative(filePath)}`);
  }
}

class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0 || 0x9e3779b9;
  }

  nextUint32(): number {
    this.state ^= this.state << 13;
    this.state ^= this.state >>> 17;
    this.state ^= this.state << 5;
    return this.state >>> 0;
  }

  next(): number {
    return this.nextUint32() / 0x1_0000_0000;
  }

  integer(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
}

function sampleMove(policy: Float32Array, moves: number[], random: SeededRandom): number {
  const weights = moves.map(move => {
    const value = policy[move];
    return Number.isFinite(value) ? Math.max(0, value) : 0;
  });
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!(total > 0) || !Number.isFinite(total)) return moves[random.integer(moves.length)];
  let draw = random.next() * total;
  for (let index = 0; index < moves.length; index++) {
    draw -= weights[index];
    if (draw < 0) return moves[index];
  }
  return moves[moves.length - 1];
}

function chooseHighestLogit(policy: Float32Array, moves: number[], random: SeededRandom): number {
  let maximum = -Infinity;
  let best: number[] = [];
  for (const move of moves) {
    const value = policy[move];
    const score = Number.isFinite(value) ? value : -Infinity;
    if (score > maximum) {
      maximum = score;
      best = [move];
    } else if (score === maximum) best.push(move);
  }
  return best[random.integer(best.length)];
}

function winnerCode(winner: GameState['winner']): WinnerCode {
  if (winner === 'draw') return 2;
  if (winner === 'X') return 1;
  if (winner === 'O') return -1;
  throw new Error('Cannot record a game without a winner');
}

function valueForPlayer(winner: WinnerCode, player: Player): number {
  if (winner === 2) return 0;
  return (winner === 1) === (player === 'X') ? 1 : -1;
}

interface ModelClient {
  evaluate: Evaluator;
  close: () => Promise<void>;
  stats: { batches: number; positions: number; meanBatchSize: number };
}

interface InferenceRequest {
  features: Float32Array;
  resolve: (evaluation: Evaluation) => void;
  reject: (error: Error) => void;
}

async function createModelClient(modelPath: string, signal: AbortSignal): Promise<ModelClient> {
  // Native ONNX inference stays inside the self-play process. Python is used
  // only between training rounds to fit/export the network, never per leaf.
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const session = await InferenceSession.create(modelPath, {
    executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1,
    graphOptimizationLevel: 'all',
  });
  if (session.inputNames.join() !== 'state' || session.outputNames.join() !== 'policy,value') {
    await session.release();
    throw new Error('Unsupported policy/value model interface');
  }
  let closed = false;
  let failure: Error | null = null;
  let scheduled: NodeJS.Immediate | null = null;
  let inFlight: Promise<void> | null = null;
  let closePromise: Promise<void> | null = null;
  const queue: InferenceRequest[] = [];
  const stats = { batches: 0, positions: 0, meanBatchSize: 0 };

  const rejectQueued = (error: Error) => {
    while (queue.length) queue.shift()!.reject(error);
  };

  const flush = () => {
    scheduled = null;
    if (closed || failure || inFlight || queue.length === 0) return;
    const batch = queue.splice(0, MAX_INFERENCE_BATCH);
    const runBatch = async () => {
      let input: InstanceType<typeof Tensor> | undefined;
      let output: Awaited<ReturnType<typeof session.run>> | undefined;
      try {
        const features = new Float32Array(batch.length * 280);
        for (let row = 0; row < batch.length; row++) features.set(batch[row].features, row * 280);
        input = new Tensor('float32', features, [batch.length, 280]);
        output = await session.run({ state: input });
        const policyTensor = output.policy;
        const valueTensor = output.value;
        const policyData = policyTensor.data as Float32Array;
        const valueData = valueTensor.data as Float32Array;
        let valuesValid = true;
        for (let index = 0; index < valueData.length; index++) {
          if (!Number.isFinite(valueData[index]) || Math.abs(valueData[index]) > 1.001) {
            valuesValid = false;
            break;
          }
        }
        if (policyTensor.dims.join() !== `${batch.length},81` || valueTensor.dims.join() !== `${batch.length},1` ||
            policyData.length !== batch.length * 81 || valueData.length !== batch.length ||
            !policyData.every(Number.isFinite) || !valuesValid) {
          throw new Error('Invalid batched model policy/value output');
        }
        stats.batches++;
        stats.positions += batch.length;
        stats.meanBatchSize = stats.positions / stats.batches;
        for (let row = 0; row < batch.length; row++) {
          batch[row].resolve({
            policy: policyData.slice(row * 81, (row + 1) * 81),
            value: valueData[row],
          });
        }
      } catch (error) {
        const reason = error instanceof Error ? error : new Error(String(error));
        failure = reason;
        for (const request of batch) request.reject(reason);
        rejectQueued(reason);
      } finally {
        try { input?.dispose(); } catch { /* native run errors still need cleanup */ }
        if (output) for (const tensor of Object.values(output)) {
          try { tensor.dispose(); } catch { /* release every tensor independently */ }
        }
      }
    };
    inFlight = runBatch();
    void inFlight.finally(() => {
      inFlight = null;
      if (queue.length) scheduleFlush();
    });
  };

  const scheduleFlush = () => {
    if (scheduled || inFlight || closed || failure || queue.length === 0) return;
    scheduled = setImmediate(flush);
  };

  const onAbort = () => {
    if (scheduled) clearImmediate(scheduled);
    scheduled = null;
    rejectQueued(new Error('Operation interrupted'));
  };
  signal.addEventListener('abort', onAbort);

  const evaluate: Evaluator = async state => {
    if (closed || signal.aborted) throw new Error('Operation interrupted');
    if (failure) throw failure;
    return await new Promise<Evaluation>((resolveEvaluation, rejectEvaluation) => {
      queue.push({ features: encodeState(state), resolve: resolveEvaluation, reject: rejectEvaluation });
      scheduleFlush();
    });
  };

  const close = () => {
    if (closePromise) return closePromise;
    closed = true;
    signal.removeEventListener('abort', onAbort);
    if (scheduled) clearImmediate(scheduled);
    scheduled = null;
    rejectQueued(new Error('Model client closed'));
    closePromise = (async () => {
      const active = inFlight;
      if (active) await active;
      await session.release();
    })();
    return closePromise;
  };

  return {
    evaluate,
    close,
    stats,
  };
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await rename(temporary, path);
}

interface GameRecord {
  game: number;
  winner: WinnerCode;
  moves: number;
  samples: [number[], number[], number][];
}

interface GameSearchStats {
  fullSearchMoves: number;
  fastSearchMoves: number;
  fullSearchSimulations: number;
  fastSearchSimulations: number;
}

interface GeneratedGame {
  record: GameRecord;
  stats: GameSearchStats;
}

const SEARCH_CAP_SEED_XOR = 0x51ed270b;

function makeSearchCapSeeds(generationSeed: number, games: number): number[] {
  const seedSource = new SeededRandom((generationSeed ^ SEARCH_CAP_SEED_XOR) >>> 0);
  return Array.from({ length: games }, () => seedSource.nextUint32());
}

function searchCapExperiment(options: Options, fullSimulations: number): SearchCapExperiment | undefined {
  const fastRaw = options['fast-simulations'];
  const probabilityRaw = options['full-search-probability'];
  if ((fastRaw === undefined) !== (probabilityRaw === undefined)) {
    throw new Error('--fast-simulations and --full-search-probability must be provided together');
  }
  if (fastRaw === undefined || probabilityRaw === undefined) return undefined;

  const fastSimulations = intOption(options, 'fast-simulations', fullSimulations, 1);
  if (fastSimulations > fullSimulations) {
    throw new Error(`--fast-simulations must be <= --simulations (${fullSimulations})`);
  }
  const fullSearchProbability = Number(probabilityRaw);
  if (!Number.isFinite(fullSearchProbability) || fullSearchProbability <= 0 || fullSearchProbability > 1) {
    throw new Error('--full-search-probability must be a finite number in (0, 1]');
  }
  return { fastSimulations, fullSearchProbability };
}

function makeTeacherEvaluator(model: Evaluator, baselineWeight: number): Evaluator {
  if (baselineWeight === 0) return model;
  return async state => {
    const [modelOutput, baselineOutput] = await Promise.all([
      model(state),
      Promise.resolve(baselineEvaluation(state)),
    ]);
    const modelWeight = 1 - baselineWeight;
    const policy = new Float32Array(81);
    for (let move = 0; move < 81; move++) {
      policy[move] = modelOutput.policy[move] * modelWeight + baselineOutput.policy[move] * baselineWeight;
    }
    return {
      policy,
      value: modelOutput.value * modelWeight + baselineOutput.value * baselineWeight,
    };
  };
}

async function generateGame(
  game: number,
  random: SeededRandom,
  simulations: number,
  modelClient: ModelClient | undefined,
  rootNoise: RootNoiseOptions | undefined,
  capExperiment: SearchCapExperiment | undefined,
  capRandom: SeededRandom | undefined,
  signal: AbortSignal,
): Promise<GeneratedGame> {
  let state = createGame();
  const positions: { features: number[]; policy: number[]; turn: Player }[] = [];
  const stats: GameSearchStats = {
    fullSearchMoves: 0,
    fastSearchMoves: 0,
    fullSearchSimulations: 0,
    fastSearchSimulations: 0,
  };
  while (state.winner === null) {
    if (signal.aborted) throw new Error('Operation interrupted');
    const fullSearch = !capExperiment || capRandom!.next() < capExperiment.fullSearchProbability;
    const moveSimulations = fullSearch ? simulations : capExperiment!.fastSimulations;
    const result = await search(state, {
      simulations: moveSimulations,
      timeMs: SEARCH_TIME_MS,
      evaluate: modelClient?.evaluate,
      seed: random.nextUint32(),
      rootNoise,
    });
    const moves = legalMoves(state);
    if (!moves.length || result.move === null) throw new Error(`Search returned no move during game ${game}`);
    if (!moves.includes(result.move)) throw new Error(`Search returned illegal move ${result.move} during game ${game}`);
    if (fullSearch) {
      stats.fullSearchMoves++;
      stats.fullSearchSimulations += result.simulations;
      positions.push({
        features: Array.from(encodeState(state)),
        policy: Array.from(result.policy),
        turn: state.turn,
      });
    } else {
      stats.fastSearchMoves++;
      stats.fastSearchSimulations += result.simulations;
    }
    const selected = sampleMove(result.policy, moves, random);
    state = play(state, Math.floor(selected / 9), selected % 9);
  }
  const winner = winnerCode(state.winner);
  return {
    record: {
      game,
      winner,
      moves: state.moves,
      samples: positions.map(position => [
        position.features,
        position.policy,
        valueForPlayer(winner, position.turn),
      ]),
    },
    stats,
  };
}

async function generate(options: Options, signal: AbortSignal): Promise<void> {
  const games = intOption(options, 'games', 1_000, 1);
  const simulations = intOption(options, 'simulations', 24, 1);
  const seed = intOption(options, 'seed', DEFAULT_SEED, 0);
  const output = resolve(process.cwd(), stringOption(options, 'output', resolve(ROOT, 'training/data/baseline.jsonl')));
  const model = optionalPath(options, 'model');
  const teacherMix = fractionOption(options, 'teacher-mix', 0);
  const capExperiment = searchCapExperiment(options, simulations);
  const rootNoiseEpsilon = fractionOption(options, 'root-noise', 0);
  const rootNoiseMass = positiveNumberOption(options, 'root-noise-mass', 2.7);
  const rootNoise: RootNoiseOptions | undefined = rootNoiseEpsilon > 0
    ? { epsilon: rootNoiseEpsilon, concentrationMass: rootNoiseMass }
    : undefined;
  if (teacherMix > 0 && !model) throw new Error('--teacher-mix requires --model');
  const parallelGames = Math.min(games, intOption(options, 'parallel-games', model ? 16 : 1, 1));
  if (model) await requireFile(model, 'Model');
  const modelSha256 = model ? createHash('sha256').update(await readFile(model)).digest('hex') : null;
  if (!isInsideRoot(output)) process.stderr.write('Note: output is outside the repository; provenance will use a relative path.\n');
  await mkdir(dirname(output), { recursive: true });
  const temporary = `${output}.${process.pid}.partial`;
  const modelClient = model ? await createModelClient(model, signal) : undefined;
  const teacherEvaluator = modelClient ? makeTeacherEvaluator(modelClient.evaluate, teacherMix) : undefined;
  const legacyBaselineRandom = !model && parallelGames === 1 ? new SeededRandom(seed) : undefined;
  let gameSeeds: number[] | undefined;
  if (!legacyBaselineRandom) {
    // Derive model/parallel game seeds from one master stream before scheduling.
    const seedSource = new SeededRandom(seed);
    gameSeeds = Array.from({ length: games }, () => seedSource.nextUint32());
  }
  const searchCapSeeds = capExperiment ? makeSearchCapSeeds(seed, games) : undefined;
  const start = Date.now();
  let sink;
  try {
    sink = await open(temporary, 'w');
    const logEvery = Math.max(1, Math.min(25, Math.ceil(games / 20)));
    let completed = 0;
    const aggregateSearchStats: GameSearchStats = {
      fullSearchMoves: 0,
      fastSearchMoves: 0,
      fullSearchSimulations: 0,
      fastSearchSimulations: 0,
    };
    let gamesWithZeroSamples = 0;
    let savedPositions = 0;
    const writeRecord = async (generated: GeneratedGame) => {
      const { record, stats } = generated;
      await sink!.write(`${JSON.stringify(record)}\n`);
      if (capExperiment) {
        aggregateSearchStats.fullSearchMoves += stats.fullSearchMoves;
        aggregateSearchStats.fastSearchMoves += stats.fastSearchMoves;
        aggregateSearchStats.fullSearchSimulations += stats.fullSearchSimulations;
        aggregateSearchStats.fastSearchSimulations += stats.fastSearchSimulations;
        savedPositions += record.samples.length;
        if (record.samples.length === 0) gamesWithZeroSamples++;
      }
      completed++;
      if (completed % logEvery === 0 || completed === games) {
        const elapsed = (Date.now() - start) / 1_000;
        process.stderr.write(`generated ${completed}/${games} games (${(completed / Math.max(elapsed, 0.001)).toFixed(2)} games/s)\n`);
      }
    };

    if (legacyBaselineRandom) {
      for (let game = 0; game < games; game++) {
        if (signal.aborted) throw new Error('Operation interrupted');
        await writeRecord(await generateGame(
          game,
          legacyBaselineRandom,
          simulations,
          undefined,
          rootNoise,
          capExperiment,
          searchCapSeeds ? new SeededRandom(searchCapSeeds[game]) : undefined,
          signal,
        ));
      }
    } else {
      for (let first = 0; first < games; first += parallelGames) {
        if (signal.aborted) throw new Error('Operation interrupted');
        const last = Math.min(games, first + parallelGames);
        const tasks = Array.from({ length: last - first }, (_, offset) => {
          const game = first + offset;
          return generateGame(
            game,
            new SeededRandom(gameSeeds![game]),
            simulations,
            teacherEvaluator ? { ...modelClient!, evaluate: teacherEvaluator } : undefined,
            rootNoise,
            capExperiment,
            searchCapSeeds ? new SeededRandom(searchCapSeeds[game]) : undefined,
            signal,
          );
        });
        let records: GeneratedGame[];
        try {
          records = await Promise.all(tasks);
        } catch (error) {
          await Promise.allSettled(tasks);
          throw error;
        }
        if (signal.aborted) throw new Error('Operation interrupted');
        for (const record of records) await writeRecord(record);
      }
    }
    await sink.close();
    sink = undefined;
    await rename(temporary, output);
    const elapsedMs = Date.now() - start;
    const elapsedSeconds = Number((elapsedMs / 1_000).toFixed(2));
    const inferenceStats = modelClient ? {
      ...modelClient.stats,
      positionsPerSecond: elapsedMs > 0
        ? Math.round(modelClient.stats.positions * 1_000 / elapsedMs)
        : modelClient.stats.positions,
    } : undefined;
    const metadata = {
      format: 'ultimate-tic-tac-toe-selfplay-v1',
      dataset: repoRelative(output),
      generatedAt: new Date().toISOString(),
      generator: 'scripts/ai.ts',
      rulesSource: 'src/game.ts',
      encoderSource: 'src/ai.ts',
      teacher: model
        ? teacherMix === 0 ? 'onnx-guided-typescript-mcts' : 'blended-onnx-typescript-mcts'
        : 'typescript-baseline-mcts',
      teacherMix,
      rootNoise: {
        epsilon: rootNoiseEpsilon,
        concentrationMass: rootNoiseMass,
        alphaFormula: 'concentrationMass / legalRootMoves',
      },
      ...(capExperiment ? {
        randomizedSearchCaps: {
          fullSimulations: simulations,
          fastSimulations: capExperiment.fastSimulations,
          fullSearchProbability: capExperiment.fullSearchProbability,
          scheduleSeedFormula: `capSeedSource=SeededRandom((generationSeed XOR 0x${SEARCH_CAP_SEED_XOR.toString(16)}) >>> 0); game i capRandom=SeededRandom(capSeedSource.nextUint32()) in game-index order; xorshift32 next()/ply; select full cap iff next() < probability; zero seed maps to 0x9e3779b9`,
          requestedCapMoves: {
            full: aggregateSearchStats.fullSearchMoves,
            fast: aggregateSearchStats.fastSearchMoves,
          },
          observedSearchCalls: {
            full: aggregateSearchStats.fullSearchMoves,
            fast: aggregateSearchStats.fastSearchMoves,
          },
          requestedSimulations: {
            fullCapTotal: aggregateSearchStats.fullSearchMoves * simulations,
            fastCapTotal: aggregateSearchStats.fastSearchMoves * capExperiment.fastSimulations,
            total: aggregateSearchStats.fullSearchMoves * simulations +
              aggregateSearchStats.fastSearchMoves * capExperiment.fastSimulations,
          },
          actualSimulations: {
            fullCapCalls: aggregateSearchStats.fullSearchMoves,
            fastCapCalls: aggregateSearchStats.fastSearchMoves,
            fullCapTotal: aggregateSearchStats.fullSearchSimulations,
            fastCapTotal: aggregateSearchStats.fastSearchSimulations,
            total: aggregateSearchStats.fullSearchSimulations + aggregateSearchStats.fastSearchSimulations,
          },
          savedPositions,
          gamesWithZeroSamples,
        },
      } : {}),
      model: modelRelative(model),
      modelSha256,
      games,
      simulationsPerPosition: simulations,
      parallelGames,
      simulationTimeLimitMs: SEARCH_TIME_MS,
      ...(inferenceStats ? { inference: inferenceStats } : {}),
      seed,
      policyTarget: 'normalized root visit counts',
      moveSampling: 'sample from root visit distribution (temperature 1)',
      valueTarget: 'final result relative to each sampled position turn: win=1, draw=0, loss=-1',
      outcomeCodes: { x: 1, o: -1, draw: 2 },
      elapsedSeconds,
    };
    const sidecar = output.endsWith('.jsonl') ? `${output.slice(0, -6)}.metadata.json` : `${output}.metadata.json`;
    await writeJsonAtomic(sidecar, metadata);
    process.stdout.write(`${JSON.stringify({ dataset: repoRelative(output), metadata: repoRelative(sidecar), games, seed, parallelGames, inference: metadata.inference ?? null, teacher: metadata.teacher, rootNoise: metadata.rootNoise })}\n`);
  } catch (error) {
    await sink?.close().catch(() => undefined);
    await rm(temporary, { force: true });
    throw error;
  } finally {
    await modelClient?.close();
  }
}

interface MatchSummary {
  games: number;
  wins: number;
  draws: number;
  losses: number;
  scoreRate: number;
  xSeat: { wins: number; draws: number; losses: number };
  oSeat: { wins: number; draws: number; losses: number };
}

function tallyMatch(summary: MatchSummary, state: GameState, modelPlayer: Player): void {
  const seat = modelPlayer === 'X' ? summary.xSeat : summary.oSeat;
  if (state.winner === 'draw') {
    summary.draws++;
    seat.draws++;
  } else if (state.winner === modelPlayer) {
    summary.wins++;
    seat.wins++;
  } else {
    summary.losses++;
    seat.losses++;
  }
  summary.games++;
}

function newSummary(): MatchSummary {
  return {
    games: 0, wins: 0, draws: 0, losses: 0, scoreRate: 0,
    xSeat: { wins: 0, draws: 0, losses: 0 },
    oSeat: { wins: 0, draws: 0, losses: 0 },
  };
}

async function playEvaluationGame(
  model: Evaluator,
  kind: 'raw-vs-random' | 'search-vs-baseline',
  modelPlayer: Player,
  simulations: number,
  openingMoves: number,
  random: SeededRandom,
  signal: AbortSignal,
): Promise<GameState> {
  let state = createGame();
  for (let index = 0; index < openingMoves && state.winner === null; index++) {
    const actions = legalMoves(state);
    const move = actions[random.integer(actions.length)];
    state = play(state, Math.floor(move / 9), move % 9);
  }
  while (state.winner === null) {
    if (signal.aborted) throw new Error('Operation interrupted');
    const actions = legalMoves(state);
    if (!actions.length) throw new Error('Generated a nonterminal state with no legal moves');
    let move: number;
    if (state.turn !== modelPlayer && kind === 'raw-vs-random') {
      move = actions[random.integer(actions.length)];
    } else if (state.turn === modelPlayer && kind === 'raw-vs-random') {
      const output = await model(state);
      move = chooseHighestLogit(output.policy, actions, random);
    } else if (state.turn === modelPlayer) {
      const result = await search(state, {
        simulations,
        timeMs: SEARCH_TIME_MS,
        evaluate: model,
        seed: random.nextUint32(),
      });
      if (result.move === null) throw new Error('Model search returned no legal move');
      move = result.move;
    } else {
      const result = await search(state, {
        simulations,
        timeMs: SEARCH_TIME_MS,
        seed: random.nextUint32(),
      });
      if (result.move === null) throw new Error('Baseline search returned no legal move');
      move = result.move;
    }
    if (!actions.includes(move)) throw new Error(`Evaluator selected illegal action ${move}`);
    state = play(state, Math.floor(move / 9), move % 9);
  }
  return state;
}

async function evaluateCommand(options: Options, signal: AbortSignal): Promise<void> {
  const modelPath = requiredPath(options, 'model');
  const gamesPerSeat = intOption(options, 'games-per-seat', 10, 1);
  const simulations = intOption(options, 'simulations', 32, 1);
  const seed = intOption(options, 'seed', 20262024, 0);
  const openingMoves = intOption(options, 'opening-moves', 2, 0);
  await requireFile(modelPath, 'Model');
  const modelHash = createHash('sha256').update(await readFile(modelPath)).digest('hex');
  const client = await createModelClient(modelPath, signal);
  try {
    const random = new SeededRandom(seed);
    const raw = newSummary();
    const guided = newSummary();
    const conditions: Array<{ name: 'raw-vs-random' | 'search-vs-baseline'; summary: MatchSummary }> = [
      { name: 'raw-vs-random', summary: raw },
      { name: 'search-vs-baseline', summary: guided },
    ];
    for (const condition of conditions) {
      for (const player of ['X', 'O'] as const) {
        for (let index = 0; index < gamesPerSeat; index++) {
          if (signal.aborted) throw new Error('Operation interrupted');
          const state = await playEvaluationGame(
            client.evaluate,
            condition.name,
            player,
            simulations,
            openingMoves,
            random,
            signal,
          );
          tallyMatch(condition.summary, state, player);
          process.stderr.write(`${condition.name}: ${condition.summary.games}/${gamesPerSeat * 2}\n`);
        }
      }
      condition.summary.scoreRate = (condition.summary.wins + condition.summary.draws * 0.5) / condition.summary.games;
    }
    const result = {
      method: 'typescript',
      model: repoRelative(modelPath),
      modelSha256: modelHash,
      gamesPerSeat,
      simulations,
      openingMoves,
      rawModelPolicy: 'greedy highest raw logit among legal moves',
      baselineSearch: 'TypeScript baseline evaluator with the same simulation budget',
      seed,
      seatsBalanced: true,
      results: {
        rawModelVsRandom: raw,
        modelSearchVsSameTsBaselineSearch: guided,
      },
    };
    const metadataPath = optionalPath(options, 'metadata');
    if (metadataPath) {
      const metadata = JSON.parse(await readFile(metadataPath, 'utf8')) as Record<string, unknown>;
      const evaluations = (metadata.evaluation && typeof metadata.evaluation === 'object'
        ? metadata.evaluation as Record<string, unknown>
        : {});
      evaluations.typescript = result;
      metadata.evaluation = evaluations;
      await writeJsonAtomic(metadataPath, metadata);
    }
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    await client.close();
  }
}

interface ArenaOpening {
  state: GameState;
  moves: [number, number];
}

interface ArenaPairPlan extends ArenaOpening {
  pair: number;
  seed: number;
  challengerXSeed: number;
  challengerOSeed: number;
}

function makeArenaOpening(seed: number): ArenaOpening {
  let state = createGame();
  const random = new SeededRandom(seed);
  const moves: number[] = [];
  for (let ply = 0; ply < 2; ply++) {
    const actions = legalMoves(state);
    const action = actions[random.integer(actions.length)];
    moves.push(action);
    state = play(state, Math.floor(action / 9), action % 9);
  }
  return { state, moves: [moves[0], moves[1]] };
}

function withValueMode(evaluator: Evaluator, mode: ValueMode): Evaluator {
  if (mode === 'model') return evaluator;
  return async state => {
    const evaluation = await evaluator(state);
    return {
      // Preserve the model policy exactly; this ablation changes only leaf value.
      policy: evaluation.policy,
      value: mode === 'zero' ? 0 : baselineEvaluation(state).value,
    };
  };
}

async function playArenaGame(
  opening: GameState,
  challengerSeat: Player,
  challenger: Evaluator,
  champion: Evaluator,
  simulations: number,
  tacticalProof: boolean,
  random: SeededRandom,
  signal: AbortSignal,
): Promise<GameState> {
  let state = opening;
  while (state.winner === null) {
    if (signal.aborted) throw new Error('Operation interrupted');
    const actions = legalMoves(state);
    if (!actions.length) throw new Error('Arena reached a nonterminal state with no legal moves');
    const evaluator = state.turn === challengerSeat ? challenger : champion;
    const result = await search(state, {
      simulations,
      timeMs: SEARCH_TIME_MS,
      evaluate: evaluator,
      seed: random.nextUint32(),
      tacticalProof,
    });
    if (result.move === null || !actions.includes(result.move)) {
      throw new Error('Arena search returned no legal move');
    }
    state = play(state, Math.floor(result.move / 9), result.move % 9);
  }
  return state;
}

function matchScore(winner: GameState['winner'], challengerSeat: Player): number {
  if (winner === 'draw') return 0.5;
  return winner === challengerSeat ? 1 : 0;
}

function wdl(score: number): 'win' | 'draw' | 'loss' {
  return score === 1 ? 'win' : score === 0.5 ? 'draw' : 'loss';
}

async function arenaCommand(options: Options, signal: AbortSignal): Promise<void> {
  const challengerValueMode = choiceOption(options, 'challenger-value', 'model', ['model', 'zero', 'heuristic'] as const);
  const championValueMode = choiceOption(options, 'champion-value', 'model', ['model', 'zero', 'heuristic'] as const);
  const tacticalProof = booleanOption(options, 'tactical-proof', false);
  const challengerPath = requiredPath(options, 'challenger');
  const championPath = requiredPath(options, 'champion');
  const plannedPairs = intOption(options, 'pairs', 10, 1);
  const pairOffset = intOption(options, 'pair-offset', 0, 0);
  if (pairOffset >= plannedPairs) throw new Error('--pair-offset must be less than --pairs');
  const pairs = intOption(options, 'pair-count', plannedPairs - pairOffset, 1);
  if (pairOffset + pairs > plannedPairs) {
    throw new Error('--pair-offset plus --pair-count must not exceed --pairs');
  }
  const simulations = intOption(options, 'simulations', 64, 1);
  const seed = intOption(options, 'seed', 20262024, 0);
  const requestedParallelGames = intOption(options, 'parallel-games', 16, 2);
  const output = resolve(process.cwd(), stringOption(options, 'output', resolve(ROOT, 'training/arena/latest.json')));
  await Promise.all([
    requireFile(challengerPath, 'Challenger model'),
    requireFile(championPath, 'Champion model'),
  ]);
  const [challengerSha256, championSha256] = await Promise.all([
    readFile(challengerPath).then(data => createHash('sha256').update(data).digest('hex')),
    readFile(championPath).then(data => createHash('sha256').update(data).digest('hex')),
  ]);
  const maxConcurrent = Math.min(pairs * 2, requestedParallelGames);
  const parallelGames = Math.max(2, maxConcurrent - (maxConcurrent % 2));
  const parallelPairs = Math.max(1, parallelGames / 2);
  const seedSource = new SeededRandom(seed);
  const plans: ArenaPairPlan[] = Array.from({ length: plannedPairs }, (_, pair) => {
    const pairSeed = seedSource.nextUint32();
    return {
      pair,
      seed: pairSeed,
      ...makeArenaOpening(pairSeed),
      challengerXSeed: pairSeed ^ 0x9e3779b9,
      challengerOSeed: pairSeed ^ 0x85ebca6b,
    };
  }).slice(pairOffset, pairOffset + pairs);

  const challengerClient = await createModelClient(challengerPath, signal);
  let championClient: ModelClient | undefined;
  try {
    championClient = await createModelClient(championPath, signal);
    const challengerEvaluator = withValueMode(challengerClient.evaluate, challengerValueMode);
    const championEvaluator = withValueMode(championClient.evaluate, championValueMode);
    const perPair: Array<Record<string, unknown>> = [];
    const gameScores: number[] = [];
    let completedPairs = 0;
    for (let first = 0; first < pairs; first += parallelPairs) {
      if (signal.aborted) throw new Error('Operation interrupted');
      const chunk = plans.slice(first, first + parallelPairs);
      const tasks = chunk.map(async plan => {
        const challengerX = playArenaGame(
          plan.state,
          'X',
          challengerEvaluator,
          championEvaluator,
          simulations,
          tacticalProof,
          new SeededRandom(plan.challengerXSeed),
          signal,
        );
        const challengerO = playArenaGame(
          plan.state,
          'O',
          challengerEvaluator,
          championEvaluator,
          simulations,
          tacticalProof,
          new SeededRandom(plan.challengerOSeed),
          signal,
        );
        let outcomes: [GameState, GameState];
        try {
          outcomes = await Promise.all([challengerX, challengerO]);
        } catch (error) {
          await Promise.allSettled([challengerX, challengerO]);
          throw error;
        }
        const [asX, asO] = outcomes;
        const scoreX = matchScore(asX.winner, 'X');
        const scoreO = matchScore(asO.winner, 'O');
        const points = scoreX + scoreO;
        const pairScore = points / 2;
        gameScores.push(scoreX, scoreO);
        return {
          pair: plan.pair,
          openingSeed: plan.seed,
          openingMoves: plan.moves,
          challengerAsX: { winner: asX.winner, score: scoreX, result: wdl(scoreX) },
          challengerAsO: { winner: asO.winner, score: scoreO, result: wdl(scoreO) },
          points,
          pairScore,
          pairResult: points > 1 ? 'win' : points < 1 ? 'loss' : 'draw',
        };
      });
      let outcomes: Record<string, unknown>[];
      try {
        outcomes = await Promise.all(tasks);
      } catch (error) {
        await Promise.allSettled(tasks);
        throw error;
      }
      perPair.push(...outcomes);
      completedPairs += outcomes.length;
      process.stderr.write(`arena: ${completedPairs}/${pairs} pairs complete\n`);
    }

    let wins = 0;
    let draws = 0;
    let losses = 0;
    for (const score of gameScores) {
      if (score === 1) wins++;
      else if (score === 0.5) draws++;
      else losses++;
    }
    let pairWins = 0;
    let pairDraws = 0;
    let pairLosses = 0;
    for (const item of perPair) {
      if (item.pairResult === 'win') pairWins++;
      else if (item.pairResult === 'draw') pairDraws++;
      else pairLosses++;
    }
    const totalPoints = gameScores.reduce((sum, score) => sum + score, 0);
    const scoreRate = totalPoints / gameScores.length;
    const pairScores = perPair.map(item => item.pairScore as number);
    const pairedMean = pairScores.reduce((sum, score) => sum + score, 0) / pairs;
    let pairedStandardError = 0;
    if (pairs > 1) {
      const sampleVariance = pairScores.reduce((sum, score) => sum + (score - pairedMean) ** 2, 0) / (pairs - 1);
      pairedStandardError = Math.sqrt(sampleVariance / pairs);
    }
    const confidenceMargin = pairs > 1 ? 1.96 * pairedStandardError : 1;
    const result = {
      format: 'ultimate-tic-tac-toe-paired-arena-v1',
      challengerModel: { model: repoRelative(challengerPath), sha256: challengerSha256 },
      championModel: { model: repoRelative(championPath), sha256: championSha256 },
      pairs,
      ...(options['pair-offset'] !== undefined || options['pair-count'] !== undefined
        ? { arenaShard: { plannedPairs, pairOffset, pairCount: pairs } } : {}),
      games: gameScores.length,
      simulations,
      challengerValueMode,
      championValueMode,
      tacticalProof,
      seed,
      openingPlyCount: 2,
      openingProtocol: 'same seeded two-ply random opening in both games of each pair; challenger and champion swap X/O',
      requestedParallelGames,
      parallelGames,
      challenger: { wins, draws, losses, games: gameScores.length, scoreRate },
      pairScores,
      pairedConfidence95: {
        lower: Math.max(0, pairedMean - confidenceMargin),
        upper: Math.min(1, pairedMean + confidenceMargin),
      },
      results: {
        pairs: { wins: pairWins, draws: pairDraws, losses: pairLosses },
        pairedScoreRate: pairedMean,
      },
      inference: {
        challenger: { ...challengerClient.stats },
        champion: { ...championClient.stats },
      },
      perPair,
      generatedAt: new Date().toISOString(),
    };
    await writeJsonAtomic(output, result);
    process.stdout.write(`${JSON.stringify({ output: repoRelative(output), pairs, games: result.games, challengerModel: challengerSha256, championModel: championSha256, scoreRate, inference: result.inference }, null, 2)}\n`);
  } finally {
    await Promise.allSettled([
      challengerClient.close(),
      championClient?.close() ?? Promise.resolve(),
    ]);
  }
}

function randomLegalGameCandidates(seed: number): Map<string, GameState> {
  const random = new SeededRandom(seed);
  const candidates = new Map<string, GameState>([['initial', createGame()]]);
  const milestones = [1, 2, 4, 8, 12, 18, 24, 30, 36];
  const first = play(createGame(), 0, 0);
  candidates.set('forced_after_first_move', first);
  candidates.set('opening_forced_move', play(first, 0, 4));
  for (let game = 0; game < 1_000; game++) {
    let state = createGame();
    while (state.winner === null) {
      const actions = legalMoves(state);
      if (!actions.length) break;
      const move = actions[random.integer(actions.length)];
      state = play(state, Math.floor(move / 9), move % 9);
      if (candidates.size < 40) candidates.set(`random_${game}_${state.moves}`, state);
      if (milestones.includes(state.moves)) candidates.set(`move_${state.moves}`, state);
      if (state.forced === null && state.results.some(result => result !== null)) {
        candidates.set('unrestricted_after_completed_destination', state);
      }
      if (state.results.includes('draw')) candidates.set('local_draw_present', state);
    }
    candidates.set('terminal', state);
    if (candidates.has('unrestricted_after_completed_destination') && candidates.has('local_draw_present')) break;
  }
  return candidates;
}

function stateJson(state: GameState): GameState {
  return {
    boards: state.boards.map(board => [...board]),
    results: [...state.results],
    turn: state.turn,
    forced: state.forced,
    winner: state.winner,
    moves: state.moves,
  };
}

async function parityCommand(options: Options, signal: AbortSignal): Promise<void> {
  const modelPath = requiredPath(options, 'model');
  const output = resolve(process.cwd(), stringOption(options, 'output', resolve(ROOT, 'training/parity.json')));
  const seed = intOption(options, 'seed', 8901, 0);
  await requireFile(modelPath, 'Model');
  const modelHash = createHash('sha256').update(await readFile(modelPath)).digest('hex');
  const client = await createModelClient(modelPath, signal);
  try {
    const candidates = randomLegalGameCandidates(seed);
    const names = [
      'initial', 'forced_after_first_move', 'opening_forced_move',
      'move_4', 'move_8', 'move_12', 'move_18', 'move_24',
      'unrestricted_after_completed_destination', 'local_draw_present', 'move_36', 'terminal',
    ];
    const selected = names.flatMap(name => {
      const state = candidates.get(name);
      return state ? [[name, state] as const] : [];
    });
    const selectedStates = new Set(selected.map(([, state]) => state));
    if (selected.length < 12) {
      for (const [name, state] of candidates) {
        if (selected.length >= 12) break;
        if (selectedStates.has(state)) continue;
        selected.push([name, state]);
        selectedStates.add(state);
      }
    }
    const cases = [];
    for (const [name, state] of selected) {
      if (signal.aborted) throw new Error('Operation interrupted');
      const encoded = encodeState(state);
      const evaluation = await client.evaluate(state);
      const actions = legalMoves(state);
      const next = actions.length ? (() => {
        const move = actions[0];
        return { move, state: stateJson(play(state, Math.floor(move / 9), move % 9)) };
      })() : undefined;
      cases.push({
        name,
        state: stateJson(state),
        encoded: Array.from(encoded),
        policy: Array.from(evaluation.policy),
        value: evaluation.value,
        legalMoves: actions,
        ...(next ? { next } : {}),
      });
    }
    const fixture = {
      model: repoRelative(modelPath),
      modelSha256: modelHash,
      seed,
      encoderSource: 'src/ai.ts',
      rulesSource: 'src/game.ts',
      cases,
    };
    await writeJsonAtomic(output, fixture);
    process.stdout.write(`${JSON.stringify({ fixture: repoRelative(output), cases: cases.length, modelSha256: modelHash })}\n`);
  } finally {
    await client.close();
  }
}

async function main(): Promise<void> {
  const controller = new AbortController();
  const interrupt = (signal: string) => {
    process.stderr.write(`\nReceived ${signal}; stopping after the current operation.\n`);
    controller.abort();
  };
  const onInterrupt = () => interrupt('SIGINT');
  const onTerminate = () => interrupt('SIGTERM');
  process.once('SIGINT', onInterrupt);
  process.once('SIGTERM', onTerminate);
  try {
    const { command, options } = parseCli(process.argv.slice(2));
    if (command === 'generate') await generate(options, controller.signal);
    else if (command === 'evaluate') await evaluateCommand(options, controller.signal);
    else if (command === 'parity') await parityCommand(options, controller.signal);
    else await arenaCommand(options, controller.signal);
  } finally {
    process.removeListener('SIGINT', onInterrupt);
    process.removeListener('SIGTERM', onTerminate);
  }
}

main().catch(error => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Error: ${message}\n`);
  process.exitCode = 1;
});
