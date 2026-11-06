#!/usr/bin/env node
/** Paired offline matches against independent Ultimate Tic-Tac-Toe engines (with Parallelism, Fixed Simulations, and Dynamic GPU Micro-Batching). */
import { createHash } from 'node:crypto';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface, type Interface } from 'node:readline';
import { appendFile, readFile, stat } from 'node:fs/promises';
import { createGame, play, type GameState, type Player } from '../src/game.ts';
import { encodeState, legalMoves, search } from '../src/ai.ts';
import { solveEndgame } from '../src/endgame.ts';
import type { Evaluator, Evaluation } from '../src/ai.ts';

type Opponent = 'alpha-beta' | 'learned-puct' | 'utttai';
type Options = {
  engine?: string;
  externalModel?: string;
  model: string;
  output: string;
  opponent: Opponent;
  pairs: number;
  offset: number;
  count: number;
  seed: number;
  timeMs: number;
  simulations?: number;
  modelSimulations: number;
  opponentSimulations: number;
  parityOnly: boolean;
  parallel: number;
};

function args(): Options {
  const values: Record<string, string> = {};
  for (let i = 0; i < process.argv.length - 2; i++) {
    const token = process.argv[i + 2];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument ${token}`);
    const key = token.slice(2);
    if (key === 'parity-only') { values[key] = 'true'; continue; }
    values[key] = process.argv[++i + 2];
  }
  const integer = (key: string, fallback: number) => {
    const raw = values[key] ?? String(fallback);
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw))) throw new Error(`Invalid --${key}`);
    return Number(raw);
  };
  const pairs = integer('pairs', 250), offset = integer('offset', 0), count = integer('count', pairs - offset);
  if (!count || offset + count > pairs) throw new Error('Invalid pair range');
  const opponent = values.opponent ?? 'alpha-beta';
  if (opponent !== 'alpha-beta' && opponent !== 'learned-puct' && opponent !== 'utttai') throw new Error('Invalid opponent');
  for (const key of ['model', 'output', opponent === 'utttai' ? 'external-model' : 'engine']) {
    if (!values[key]) throw new Error(`Missing --${key}`);
  }
  const parallel = integer('parallel', opponent === 'utttai' ? 16 : 1);

  const rawSims = values.simulations !== undefined ? integer('simulations', 512) : undefined;
  const modelSimulations = values['model-simulations'] !== undefined ? integer('model-simulations', 512) : (rawSims ?? 4096);
  const opponentSimulations = values['opponent-simulations'] !== undefined ? integer('opponent-simulations', 512) : (rawSims ?? 4096);
  const isFixedSims = rawSims !== undefined || values['model-simulations'] !== undefined || values['opponent-simulations'] !== undefined;
  const timeMs = isFixedSims ? 30_000 : integer('time-ms', 140);

  return {
    engine: values.engine,
    externalModel: values['external-model'],
    model: values.model,
    output: values.output,
    opponent,
    pairs,
    offset,
    count,
    seed: integer('seed', 2026102701),
    timeMs,
    simulations: rawSims,
    modelSimulations,
    opponentSimulations,
    parityOnly: values['parity-only'] === 'true',
    parallel,
  };
}

function random(seed: number): () => number {
  let state = seed >>> 0 || 0x9e3779b9;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return state >>> 0; };
}

function coord(move: number): string {
  const board = Math.floor(move / 9), cell = move % 9;
  const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
  const col = board % 3 * 3 + cell % 3;
  return String.fromCharCode(97 + col) + String(9 - row);
}

function action(text: string): number {
  if (!/^[a-i][1-9]$/.test(text)) throw new Error(`Invalid engine move ${text}`);
  const col = text.charCodeAt(0) - 97, row = 9 - Number(text[1]);
  return (Math.floor(row / 3) * 3 + Math.floor(col / 3)) * 9 + (row % 3) * 3 + col % 3;
}

class Engine {
  private child: ChildProcessWithoutNullStreams;
  private lines: Interface;
  private waiting: ((line: string) => void) | null = null;
  private error: Error | null = null;

  constructor(binary: string) {
    this.child = spawn(binary, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on('line', line => {
      const waiting = this.waiting;
      this.waiting = null;
      if (waiting) waiting(line);
      else this.error = new Error(`Unsolicited engine output: ${line}`);
    });
    this.child.on('error', error => { this.error = error; this.waiting?.(`error ${error.message}`); });
    this.child.on('exit', code => {
      this.error = new Error(`Engine exited with ${code}`);
      this.waiting?.(`error engine exited with ${code}`);
    });
  }

  private query(command: string): Promise<string> {
    if (this.error) return Promise.reject(this.error);
    return new Promise((resolve, reject) => {
      this.waiting = resolve;
      this.child.stdin.write(`${command}\n`, error => { if (error) reject(error); });
    });
  }

  async state(history: number[]): Promise<Record<string, unknown>> {
    const moves = history.map(coord).join(' ');
    const line = await this.query(`state${moves ? ` ${moves}` : ''}`);
    const match = /^state json (.+)$/.exec(line);
    if (!match) throw new Error(`Unexpected state response: ${line}`);
    return JSON.parse(match[1]);
  }

  async move(history: number[], opponent: Opponent, timeMs: number, seed: number): Promise<{ move: number; work: number; elapsedMs: number }> {
    const started = performance.now();
    const moves = history.map(coord).join(' ');
    const setLine = await this.query(`position startpos${moves ? ` moves ${moves}` : ''}`);
    if (setLine !== 'ok') throw new Error(`Unexpected position response: ${setLine}`);
    const line = await this.query(opponent === 'alpha-beta'
      ? `play depth 20 softtime ${timeMs}`
      : `mcts simulations 4096 softtime ${timeMs} strategy learned-puct seed ${seed}`);
    const match = /^bestmove ([a-i][1-9]) (?:depth \d+ score -?\d+ nodes (\d+)|simulations (\d+) nodes \d+ score [\d.]+ strategy learned-puct)$/.exec(line);
    if (!match) throw new Error(`Unexpected move response: ${line}`);
    return { move: action(match[1]), work: Number(match[2] ?? match[3]), elapsedMs: performance.now() - started };
  }

  close(): void { this.child.stdin.end('quit\n'); this.lines.close(); this.child.kill(); }
}

async function sha(path: string): Promise<string> { return createHash('sha256').update(await readFile(path)).digest('hex'); }

async function parity(binary: string): Promise<void> {
  const engine = new Engine(binary), next = random(8841);
  let checked = 0;
  try {
    for (let game = 0; game < 100; game++) {
      let state = createGame();
      const history: number[] = [];
      while (true) {
        const theirs = await engine.state(history);
        const marks = Array<null | Player>(81).fill(null);
        for (let move = 0; move < 81; move++) {
          const text = coord(move);
          const col = text.charCodeAt(0) - 97, row = 9 - Number(text[1]);
          marks[row * 9 + col] = state.boards[Math.floor(move / 9)][move % 9];
        }
        const externalMoves = (theirs.legalMoves as string[]).map(action).sort((a, b) => a - b);
        const moves = legalMoves(state).sort((a, b) => a - b);
        const winner = theirs.winner ?? (theirs.result === 'draw' ? 'draw' : null);
        if (JSON.stringify(theirs.board) !== JSON.stringify(marks)
          || JSON.stringify(theirs.miniBoards) !== JSON.stringify(state.results)
          || JSON.stringify(externalMoves) !== JSON.stringify(moves)
          || theirs.sideToMove !== state.turn || (state.winner === null && theirs.activeBoard !== state.forced)
          || winner !== state.winner) {
          throw new Error(`Rule mismatch at game ${game} history ${history.join(',')}: ${JSON.stringify(theirs)}`);
        }
        checked++;
        if (state.winner !== null) break;
        const move = moves[next() % moves.length];
        history.push(move);
        state = play(state, Math.floor(move / 9), move % 9);
      }
    }
  } finally { engine.close(); }
  process.stdout.write(`Rule parity passed: ${checked} positions across 100 games\n`);
}

interface BatchedRequest {
  features: Float32Array;
  resolve: (evaluation: Evaluation) => void;
  reject: (error: Error) => void;
}

/** Dynamic micro-batched evaluator for candidate models (R20 / Mini-HARNet / Teacher). */
async function modelEvaluator(path: string, maxBatchSize: number = 64): Promise<{ evaluate: Evaluator; close: () => Promise<void> }> {
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const useCuda = process.env.ORT_CUDA === '1';
  let session: InstanceType<typeof InferenceSession>;
  if (useCuda) {
    try {
      session = await InferenceSession.create(path, {
        executionProviders: [{ name: 'cuda', device_id: 0 }],
        graphOptimizationLevel: 'all',
      });
      process.stderr.write(`[ORT] Loaded model ${path} on CUDA (device 0, batched max=${maxBatchSize})\n`);
    } catch (err) {
      process.stderr.write(`[ORT] Model CUDA failed (${err}), falling back to CPU\n`);
      session = await InferenceSession.create(path, {
        executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
        graphOptimizationLevel: 'all',
      });
    }
  } else {
    session = await InferenceSession.create(path, {
      executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
      graphOptimizationLevel: 'all',
    });
  }

  if (session.inputNames.join() !== 'state' || session.outputNames.join() !== 'policy,value') throw new Error('Bad model interface');

  let closed = false;
  let inFlight: Promise<void> | null = null;
  let scheduled: NodeJS.Immediate | null = null;
  const queue: BatchedRequest[] = [];

  const flush = () => {
    scheduled = null;
    if (closed || inFlight || queue.length === 0) return;
    const batch = queue.splice(0, maxBatchSize);
    inFlight = (async () => {
      let input: InstanceType<typeof Tensor> | undefined;
      let output: any;
      try {
        const tensorData = new Float32Array(batch.length * 280);
        for (let i = 0; i < batch.length; i++) tensorData.set(batch[i].features, i * 280);
        input = new Tensor('float32', tensorData, [batch.length, 280]);
        output = await session.run({ state: input });
        const policyData = output.policy.data as Float32Array;
        const valueData = output.value.data as Float32Array;
        for (let i = 0; i < batch.length; i++) {
          batch[i].resolve({
            policy: policyData.slice(i * 81, (i + 1) * 81),
            value: Number(valueData[i]),
          });
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        for (const req of batch) req.reject(error);
      } finally {
        try { input?.dispose(); } catch {}
        if (output) for (const t of Object.values(output) as any[]) { try { t?.dispose(); } catch {} }
      }
    })();
    void inFlight.finally(() => {
      inFlight = null;
      if (queue.length > 0 && !scheduled) scheduled = setImmediate(flush);
    });
  };

  const evaluate: Evaluator = state => {
    if (closed) return Promise.reject(new Error('Evaluator closed'));
    return new Promise<Evaluation>((resolve, reject) => {
      queue.push({ features: encodeState(state), resolve, reject });
      if (!scheduled && !inFlight) scheduled = setImmediate(flush);
    });
  };

  return {
    evaluate,
    close: async () => {
      closed = true;
      if (scheduled) clearImmediate(scheduled);
      if (inFlight) await inFlight;
      await session.release();
    },
  };
}

/** Dynamic micro-batched evaluator for uttt.ai champion. */
async function utttaiEvaluator(path: string, maxBatchSize: number = 64): Promise<{ evaluate: Evaluator; close: () => Promise<void> }> {
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const useCuda = process.env.ORT_CUDA === '1';
  let session: InstanceType<typeof InferenceSession>;
  if (useCuda) {
    try {
      session = await InferenceSession.create(path, {
        executionProviders: [{ name: 'cuda', device_id: 0 }],
        graphOptimizationLevel: 'all',
      });
      process.stderr.write(`[ORT] Loaded uttt.ai ${path} on CUDA (device 0, batched max=${maxBatchSize})\n`);
    } catch (err) {
      process.stderr.write(`[ORT] uttt.ai CUDA failed (${err}), falling back to CPU\n`);
      session = await InferenceSession.create(path, {
        executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
        graphOptimizationLevel: 'all',
      });
    }
  } else {
    session = await InferenceSession.create(path, {
      executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1,
      graphOptimizationLevel: 'all',
    });
  }

  const shape = session.inputMetadata[0]?.shape?.join();
  if (session.inputNames.join() !== 'input' || session.outputNames.join() !== 'policy_logits,state_value'
      || (shape !== '1,4,9,9' && shape !== 'batch,4,9,9' && !shape?.endsWith(',4,9,9'))) {
    throw new Error(`Bad uttt.ai model interface: shape=${shape}`);
  }

  let closed = false;
  let inFlight: Promise<void> | null = null;
  let scheduled: NodeJS.Immediate | null = null;
  const queue: BatchedRequest[] = [];

  const flush = () => {
    scheduled = null;
    if (closed || inFlight || queue.length === 0) return;
    const batch = queue.splice(0, maxBatchSize);
    inFlight = (async () => {
      let input: InstanceType<typeof Tensor> | undefined;
      let output: any;
      try {
        const tensorData = new Float32Array(batch.length * 324);
        for (let i = 0; i < batch.length; i++) tensorData.set(batch[i].features, i * 324);
        input = new Tensor('float32', tensorData, [batch.length, 4, 9, 9]);
        output = await session.run({ input });
        const logits = output.policy_logits.data as Float32Array;
        const values = output.state_value.data as Float32Array;
        for (let i = 0; i < batch.length; i++) {
          const policy = new Float32Array(81);
          const offset = i * 81;
          for (let move = 0; move < 81; move++) {
            const board = Math.floor(move / 9), cell = move % 9;
            const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
            const col = board % 3 * 3 + cell % 3;
            policy[move] = logits[offset + row * 9 + col];
          }
          batch[i].resolve({
            policy,
            value: Number(values[i]),
          });
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        for (const req of batch) req.reject(error);
      } finally {
        try { input?.dispose(); } catch {}
        if (output) for (const t of Object.values(output) as any[]) { try { t?.dispose(); } catch {} }
      }
    })();
    void inFlight.finally(() => {
      inFlight = null;
      if (queue.length > 0 && !scheduled) scheduled = setImmediate(flush);
    });
  };

  const evaluate: Evaluator = state => {
    if (closed) return Promise.reject(new Error('Evaluator closed'));
    const features = new Float32Array(324), legal = legalMoves(state);
    for (let move = 0; move < 81; move++) {
      const board = Math.floor(move / 9), cell = move % 9;
      const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
      const col = board % 3 * 3 + cell % 3;
      const flat = row * 9 + col, mark = state.boards[board][cell];
      if (mark === state.turn) features[flat] = 1;
      else if (mark !== null) features[81 + flat] = 1;
      features[162 + flat] = state.turn === 'X' ? 1 : -1;
    }
    for (const move of legal) {
      const board = Math.floor(move / 9), cell = move % 9;
      const row = Math.floor(board / 3) * 3 + Math.floor(cell / 3);
      const col = board % 3 * 3 + cell % 3;
      features[243 + row * 9 + col] = 1;
    }
    return new Promise<Evaluation>((resolve, reject) => {
      queue.push({ features, resolve, reject });
      if (!scheduled && !inFlight) scheduled = setImmediate(flush);
    });
  };

  return {
    evaluate,
    close: async () => {
      closed = true;
      if (scheduled) clearImmediate(scheduled);
      if (inFlight) await inFlight;
      await session.release();
    },
  };
}

function opening(seed: number): { state: GameState; history: number[] } {
  let state = createGame();
  const history: number[] = [], next = random(seed);
  for (let i = 0; i < 2; i++) {
    const moves = legalMoves(state), move = moves[Math.floor(next() / 0x1_0000_0000 * moves.length)];
    history.push(move);
    state = play(state, Math.floor(move / 9), move % 9);
  }
  return { state, history };
}

async function game(
  initial: { state: GameState; history: number[] },
  modelSeat: Player,
  engine: Engine | undefined,
  evaluate: Evaluator,
  opponent: Opponent,
  timeMs: number,
  seed: number,
  externalEvaluate?: Evaluator,
  modelSimulations: number = 4096,
  opponentSimulations: number = 4096,
): Promise<{ winner: GameState['winner']; moves: number[]; modelTurns: number; opponentTurns: number; modelWork: number; opponentWork: number }> {
  let state = initial.state;
  const history = [...initial.history];
  const next = random(seed);
  let modelTurns = 0, opponentTurns = 0, modelWork = 0, opponentWork = 0;
  while (state.winner === null) {
    const legal = legalMoves(state);
    let move: number;
    if (state.turn === modelSeat || opponent === 'utttai') {
      const endgame = solveEndgame(state, { maxEmptySquares: 18, maxNodes: 32_000, timeMs: 2_000 });
      if (endgame?.move !== null && endgame !== null && (endgame.solved || endgame.guaranteedValue !== null)) move = endgame.move;
      else {
        const modelTurn = state.turn === modelSeat;
        const simBudget = modelTurn ? modelSimulations : opponentSimulations;
        const result = await search(state, {
          simulations: simBudget,
          timeMs,
          tacticalProof: true,
          evaluate: modelTurn ? evaluate : externalEvaluate,
          seed: next(),
        });
        if (result.move === null) throw new Error(`${modelTurn ? 'R20' : 'uttt.ai'} returned no move`);
        move = result.move;
        if (modelTurn) modelWork += result.simulations;
        else opponentWork += result.simulations;
      }
      if (state.turn === modelSeat) modelTurns++;
      else opponentTurns++;
    } else {
      const result = await engine!.move(history, opponent, timeMs, next());
      move = result.move;
      opponentWork += result.work;
      opponentTurns++;
    }
    if (!legal.includes(move)) throw new Error(`${state.turn === modelSeat ? 'R20' : opponent} illegal move ${move} after ${history.join(',')}`);
    history.push(move);
    state = play(state, Math.floor(move / 9), move % 9);
  }
  return { winner: state.winner, moves: history, modelTurns, opponentTurns, modelWork, opponentWork };
}

async function calibrateModel140ms(modelPath: string): Promise<number> {
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const session = await InferenceSession.create(modelPath, {
    executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1, graphOptimizationLevel: 'all',
  });
  try {
    const evaluate: Evaluator = async (state) => {
      const input = new Tensor('float32', encodeState(state), [1, 280]);
      const out = await session.run({ state: input });
      try {
        return { policy: Float32Array.from(out.policy.data as Float32Array), value: Number((out.value.data as Float32Array)[0]) };
      } finally { input.dispose(); for (const t of Object.values(out)) t.dispose(); }
    };
    let state = createGame();
    await search(state, { simulations: 50, timeMs: 50, evaluate });
    const samples: number[] = [];
    for (let ply = 0; ply < 8; ply++) {
      const res = await search(state, { simulations: 10000, timeMs: 140, evaluate, seed: 100 + ply });
      samples.push(res.simulations);
      const moves = legalMoves(state);
      state = play(state, Math.floor(moves[0] / 9), moves[0] % 9);
    }
    return Math.round(samples.reduce((a, b) => a + b, 0) / samples.length);
  } finally {
    await session.release();
  }
}

async function main(): Promise<void> {
  const options = args();
  if (options.opponent === 'utttai' && (options.simulations === 1253 || options.modelSimulations === 1253)) {
    if (options.model.includes('wide') || options.model.includes('fast') || options.model.includes('37b') || options.model.includes('gen38')) {
      console.log(`[Calibration] Detected architectural variant (${options.model}). Calibrating exact 140ms single-thread CPU throughput...`);
      const calibratedSims = await calibrateModel140ms(options.model);
      console.log(`[Calibration] Dynamic 140ms hardware equivalent: ${calibratedSims} simulations/move (was default ${options.modelSimulations})`);
      options.modelSimulations = calibratedSims;
      if (options.simulations !== undefined) options.simulations = calibratedSims;
    }
  }
  const [externalHash, modelHash] = await Promise.all([sha((options.engine ?? options.externalModel)!), sha(options.model)]);
  if (options.engine) await parity(options.engine);
  if (options.parityOnly) return;
  const config = {
    format: 'external-arena-v1',
    opponent: options.opponent,
    externalHash,
    modelHash,
    pairs: options.pairs,
    seed: options.seed,
    timeMs: options.timeMs,
    simulations: options.simulations,
    modelSimulations: options.modelSimulations,
    opponentSimulations: options.opponentSimulations,
    openingPlies: 2,
    model: 'R20 production Hard search with bounded exact endgame',
  };
  const existing = await stat(options.output).then(() => readFile(options.output, 'utf8')).catch(() => '');
  const lines = existing.trim() ? existing.trim().split('\n').map(line => JSON.parse(line)) : [];
  if (lines.length && JSON.stringify(lines[0]) !== JSON.stringify(config)) throw new Error('Existing report configuration differs');
  if (!lines.length) await appendFile(options.output, `${JSON.stringify(config)}\n`);
  const finished = new Set<number>(lines.slice(1).map(row => row.pair));
  const seedSource = random(options.seed);
  const seeds = Array.from({ length: options.pairs }, () => seedSource());
  
  // Set batch size based on parallel concurrency (2x parallel because of paired games)
  const batchSize = Math.max(32, options.parallel * 2);
  const model = await modelEvaluator(options.model, batchSize);
  const external = options.externalModel ? await utttaiEvaluator(options.externalModel, batchSize) : undefined;
  const engine = options.engine ? new Engine(options.engine) : undefined;

  try {
    const parallel = options.parallel;
    for (let first = options.offset; first < options.offset + options.count; first += parallel) {
      const last = Math.min(options.offset + options.count, first + parallel);
      const tasks = Array.from({ length: last - first }, async (_, offset) => {
        const pair = first + offset;
        if (finished.has(pair)) return;
        const op = opening(seeds[pair]);
        const [x, o] = await Promise.all([
          game(op, 'X', engine, model.evaluate, options.opponent, options.timeMs, seeds[pair] ^ 0x9e3779b9, external?.evaluate, options.modelSimulations, options.opponentSimulations),
          game(op, 'O', engine, model.evaluate, options.opponent, options.timeMs, seeds[pair] ^ 0x85ebca6b, external?.evaluate, options.modelSimulations, options.opponentSimulations),
        ]);
        const score = (result: typeof x, seat: Player) => result.winner === 'draw' ? 0.5 : result.winner === seat ? 1 : 0;
        const row = {
          pair,
          openingSeed: seeds[pair],
          opening: op.history,
          modelX: x,
          modelO: o,
          pairScore: (score(x, 'X') + score(o, 'O')) / 2,
        };
        await appendFile(options.output, `${JSON.stringify(row)}\n`);
        process.stderr.write(`${options.opponent} pair ${pair + 1}/${options.pairs}: score ${row.pairScore} (model: ${x.modelWork + o.modelWork} sims, opp: ${x.opponentWork + o.opponentWork} sims)\n`);
      });
      await Promise.all(tasks);
    }
  } finally { engine?.close(); await model.close(); await external?.close(); }
}

main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`); process.exitCode = 1; });
