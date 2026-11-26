#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { access, mkdir, open, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, extname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, type Browser, type BrowserContext, type Page } from '@playwright/test';
import { pairSeedGenerator, parseOpeningSchedule, type OpeningSchedule } from './arena-openings.ts';
import { legalMoves } from '../src/ai.ts';
import { solveEndgame } from '../src/endgame.ts';
import { createGame, play } from '../src/game.ts';
import type { GameState, Player } from '../src/game.ts';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SITE = '/play/ultimate-tic-tac-toe/';
const DIST = resolve(ROOT, 'dist/play/ultimate-tic-tac-toe');
const HARD_SETTINGS = { difficulty: 'hard', simulations: 384, timeMs: 220, tacticalProof: true } as const;
const EASY_SETTINGS = { simulations: 16, timeMs: 60, tacticalProof: false } as const;
const BASELINE_SETTINGS = { difficulty: 'hard', simulations: 384, timeMs: 650, tacticalProof: true } as const;
const DEFAULT_PAIRS = 32;
const DEFAULT_SEED = 15406102;

type Options = Record<string, string>;

interface BrowserReply {
  move: number | null;
  source: 'model' | 'search';
  simulations: number;
  elapsedMs: number;
  settings?: { simulations: number; timeMs: number; tacticalProof: boolean };
}

interface Observation {
  elapsedMs: number;
  simulations: number;
  source: BrowserReply['source'];
}

interface Agent {
  label: 'challenger' | 'champion';
  modelPath: string;
  modelSha256: string;
  page: Page;
  context: BrowserContext;
  modelUrl: string;
  modelLoads: number;
  observations: Observation[];
  settings: typeof HARD_SETTINGS | typeof BASELINE_SETTINGS;
  pageErrors: string[];
  warmupMs: number;
}

interface GameResult {
  winner: GameState['winner'];
  plies: number;
  challengerTurns: number;
  championTurns: number;
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

  integer(maxExclusive: number): number {
    return Math.floor((this.nextUint32() / 0x1_0000_0000) * maxExclusive);
  }
}

function parseArgs(argv: string[]): { pairs: number; seed: number; openingSchedule: OpeningSchedule; challenger: string; champion: string; output: string; championWorker?: string } {
  const options: Options = {};
  if (argv.length === 1 && ['--help', '-h'].includes(argv[0])) {
    process.stdout.write(
      'Usage: node --experimental-strip-types scripts/browser-arena.ts ' +
      '--challenger model.onnx --champion model.onnx --output result.json ' +
      `[--pairs ${DEFAULT_PAIRS}] [--seed ${DEFAULT_SEED}] [--champion-worker baseline384.js] [--opening-schedule legacy|independent]\n` +
      'Runs a paired arena through the production dist worker with hard settings.\n',
    );
    process.exit(0);
  }
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const equal = token.indexOf('=');
    const key = token.slice(2, equal < 0 ? undefined : equal);
    const value = equal < 0 ? argv[++index] : token.slice(equal + 1);
    if (!key || value === undefined || value.startsWith('--')) throw new Error(`Expected a value for ${token}`);
    if (options[key] !== undefined) throw new Error(`Option --${key} was provided more than once`);
    if (!['pairs', 'seed', 'challenger', 'champion', 'output', 'champion-worker', 'opening-schedule'].includes(key)) {
      throw new Error(`Unknown option: --${key}`);
    }
    options[key] = value;
  }

  const integerOption = (name: 'pairs' | 'seed', fallback: number, minimum: number): number => {
    const raw = options[name];
    if (raw === undefined) return fallback;
    if (!/^\d+$/.test(raw)) throw new Error(`--${name} must be an integer >= ${minimum}`);
    const value = Number(raw);
    if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`--${name} must be an integer >= ${minimum}`);
    return value;
  };
  for (const name of ['challenger', 'champion', 'output'] as const) {
    if (!options[name]) throw new Error(`--${name} is required`);
  }
  return {
    pairs: integerOption('pairs', DEFAULT_PAIRS, 1),
    seed: integerOption('seed', DEFAULT_SEED, 0),
    openingSchedule: parseOpeningSchedule(options['opening-schedule']),
    challenger: resolve(process.cwd(), options.challenger!),
    champion: resolve(process.cwd(), options.champion!),
    output: resolve(process.cwd(), options.output!),
    championWorker: options['champion-worker'] ? resolve(process.cwd(), options['champion-worker']) : undefined,
  };
}

function contentType(filePath: string): string {
  switch (extname(filePath)) {
    case '.html': return 'text/html; charset=utf-8';
    case '.js':
    case '.mjs': return 'text/javascript; charset=utf-8';
    case '.css': return 'text/css; charset=utf-8';
    case '.wasm': return 'application/wasm';
    case '.json': return 'application/json; charset=utf-8';
    default: return 'application/octet-stream';
  }
}

async function startStaticServer(dist = DIST): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://127.0.0.1').pathname);
      if (!pathname.startsWith(SITE)) throw new Error('Outside site prefix');
      const relativePath = pathname.slice(SITE.length) || 'index.html';
      const filePath = resolve(dist, relativePath);
      if (filePath !== dist && !filePath.startsWith(`${dist}${sep}`)) throw new Error('Path outside dist');
      const bytes = await readFile(filePath);
      response.writeHead(200, { 'Content-Type': contentType(filePath), 'Cache-Control': 'no-store' });
      response.end(bytes);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  });
  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Could not read local static server address');
  return {
    baseUrl: `http://127.0.0.1:${address.port}${SITE}`,
    close: () => new Promise<void>((resolveClose, reject) => server.close(error => error ? reject(error) : resolveClose())),
  };
}

async function hashDistAssets(dist = DIST): Promise<Record<string, string>> {
  const files: string[] = [];
  const walk = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const filePath = resolve(directory, entry.name);
      if (entry.isDirectory()) await walk(filePath);
      else if (entry.isFile()) files.push(filePath);
    }
  };
  await walk(dist);
  const result: Record<string, string> = {};
  for (const filePath of files.sort()) {
    result[relative(dist, filePath).split(sep).join('/')] = createHash('sha256')
      .update(await readFile(filePath)).digest('hex');
  }
  return result;
}

function assertNoPageErrors(agents: Agent[]): void {
  const errors = agents.flatMap(agent => agent.pageErrors.map(message => `${agent.label}: ${message}`));
  if (errors.length) throw new Error(`Browser page errors: ${errors.join(' | ')}`);
}

async function makeAgent(
  label: Agent['label'],
  modelPath: string,
  modelSha256: string,
  modelBytes: Buffer,
  baseUrl: string,
  workerAsset: string,
  browser: Browser,
  overrideWorker?: Buffer,
): Promise<Agent> {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  let modelLoads = 0;
  await context.route(new URL('models/browser-arena.onnx', baseUrl).href, async route => {
    modelLoads++;
    await route.fulfill({ status: 200, contentType: 'application/octet-stream', body: modelBytes });
  });
  if (overrideWorker) await context.route(`**/assets/${workerAsset}`, route => route.fulfill({ status: 200, contentType: 'text/javascript', body: overrideWorker }));
  await page.goto(baseUrl, { waitUntil: 'load' });
  const modelUrl = new URL('models/browser-arena.onnx', baseUrl).href;
  const workerUrl = new URL(`assets/${workerAsset}`, baseUrl).href;
  await page.evaluate(({ workerUrl, modelUrl }) => {
    const worker = new Worker(workerUrl, { type: 'module' });
    let nextId = 0;
    const pending = new Map<number, {
      resolve: (value: any) => void;
      reject: (reason: Error) => void;
      started: number;
      timer: number;
    }>();
    const rejectAll = (reason: Error) => {
      for (const item of pending.values()) {
        window.clearTimeout(item.timer);
        item.reject(reason);
      }
      pending.clear();
    };
    worker.addEventListener('message', event => {
      const item = pending.get(event.data.id);
      if (!item) return;
      pending.delete(event.data.id);
      window.clearTimeout(item.timer);
      if (event.data.error) item.reject(new Error(event.data.error));
      else item.resolve({ ...event.data, elapsedMs: performance.now() - item.started });
    });
    worker.addEventListener('error', (event: ErrorEvent) => rejectAll(new Error(event.message || 'Bundled worker failed')));
    (window as any).__browserArena = {
      request(state: GameState, difficulty: 'easy' | 'hard') {
        return new Promise((resolve, reject) => {
          const id = ++nextId;
          const started = performance.now();
          const timer = window.setTimeout(() => {
            pending.delete(id);
            reject(new Error(`Worker response timeout for request ${id}`));
          }, 20_000);
          pending.set(id, { resolve, reject, started, timer });
          try {
            worker.postMessage({ id, action: 'move', state, difficulty, modelUrl });
          } catch (error) {
            window.clearTimeout(timer);
            pending.delete(id);
            reject(error);
          }
        });
      },
      dispose() {
        worker.terminate();
        rejectAll(new Error('Worker disposed'));
      },
    };
  }, { workerUrl, modelUrl });
  return { label, modelPath, modelSha256, page, context, modelUrl, get modelLoads() { return modelLoads; }, observations: [], settings: overrideWorker ? BASELINE_SETTINGS : HARD_SETTINGS, pageErrors, warmupMs: 0 };
}

/** Independently verify a zero-visit return from the worker's bounded Hard endgame search. */
function isEndgameMove(state: GameState, move: number): boolean {
  const options = { maxEmptySquares: 14, maxNodes: 16_000, timeMs: 2_000 };
  const root = solveEndgame(state, options);
  if (!root) return false;
  if (root.move === move && (root.solved || root.guaranteedValue !== null)) return true;
  // The worker can time out after proving a draw, even when a later root move
  // wins. Verify its actual child separately. A proved root loss admits any legal move.
  if (root.solved && root.value === -1) return true;
  const child = play(state, Math.floor(move / 9), move % 9);
  if (child.winner !== null) return child.winner === 'draw' || child.winner === state.turn;
  const reply = solveEndgame(child, { ...options, maxNodes: 200_000, optimizeMateDistance: false });
  return !!reply?.solved && reply.value <= 0;
}

async function requestMove(agent: Agent, state: GameState, difficulty: 'easy' | 'hard', scored = false): Promise<BrowserReply> {
  const reply = await agent.page.evaluate(
    async ({ state, difficulty }) => await (window as any).__browserArena.request(state, difficulty),
    { state, difficulty },
  ) as BrowserReply;
  const legacy = agent.settings === BASELINE_SETTINGS;
  const expected = difficulty === 'hard' ? agent.settings : legacy ? undefined : EASY_SETTINGS;
  // The archival override is identified by its plan and binary hash. Some
  // historical workers did not echo a settings object at all.
  if ((!reply.settings && !legacy) || (expected && reply.settings && (
    reply.settings.simulations !== expected.simulations || reply.settings.timeMs !== expected.timeMs
    || reply.settings.tacticalProof !== expected.tacticalProof
  ))) throw new Error(`${agent.label} ${difficulty} worker settings mismatch: ${JSON.stringify(reply.settings)}`);
  const cap = expected?.simulations ?? reply.settings?.simulations ?? agent.settings.simulations;
  if (!Number.isInteger(reply.simulations) || reply.simulations < 0 || reply.simulations > cap) {
    throw new Error(`${agent.label} worker returned invalid simulation count: ${reply.simulations}`);
  }
  if (!legacy && difficulty === 'easy' && state.moves === 0 && reply.simulations === 0) {
    throw new Error(`${agent.label} model probe completed no simulations`);
  }
  if (!Number.isFinite(reply.elapsedMs) || reply.elapsedMs < 0) {
    throw new Error(`${agent.label} worker returned invalid latency: ${reply.elapsedMs}`);
  }
  if (reply.move === null || !legalMoves(state).includes(reply.move)) throw new Error(`${agent.label} returned illegal move ${reply.move}`);
  if (reply.source !== 'model') {
    if (reply.source !== 'search' || difficulty !== 'hard' || legacy || reply.simulations !== 0 || !isEndgameMove(state, reply.move)) {
      throw new Error(`${agent.label} used an unverified search fallback`);
    }
    // A failed model session is sticky in the production worker. Easy on an
    // empty board must exercise it, and an exact Hard return already cleared its subtree.
    await requestMove(agent, createGame(), 'easy');
  }
  if (scored) agent.observations.push({ elapsedMs: reply.elapsedMs, simulations: reply.simulations, source: reply.source });
  return reply;
}

function makeOpening(seed: number): { state: GameState; moves: [number, number] } {
  let state = createGame();
  const random = new SeededRandom(seed);
  const moves: number[] = [];
  for (let ply = 0; ply < 2; ply++) {
    const actions = legalMoves(state);
    const move = actions[random.integer(actions.length)];
    moves.push(move);
    state = play(state, Math.floor(move / 9), move % 9);
  }
  return { state, moves: [moves[0], moves[1]] };
}

async function playGame(
  opening: GameState,
  challengerSeat: Player,
  challenger: Agent,
  champion: Agent,
): Promise<GameResult> {
  // Easy moves clear the current worker's saved Hard subtree while preserving
  // the warmed inference session. Run these probes outside scored timings.
  for (const agent of [challenger, champion]) await requestMove(agent, createGame(), 'easy');
  let state = opening;
  let challengerTurns = 0;
  let championTurns = 0;
  while (state.winner === null) {
    const actions = legalMoves(state);
    if (!actions.length) throw new Error('Browser arena reached a nonterminal state without legal actions');
    const isChallengerTurn = state.turn === challengerSeat;
    const agent = isChallengerTurn ? challenger : champion;
    const reply = await requestMove(agent, state, 'hard', true);
    if (reply.move === null || !actions.includes(reply.move)) {
      throw new Error(`${agent.label} selected an illegal move ${reply.move}`);
    }
    if (isChallengerTurn) challengerTurns++;
    else championTurns++;
    state = play(state, Math.floor(reply.move / 9), reply.move % 9);
  }
  return { winner: state.winner, plies: state.moves - opening.moves, challengerTurns, championTurns };
}

function score(winner: GameState['winner'], seat: Player): number {
  if (winner === 'draw') return 0.5;
  return winner === seat ? 1 : 0;
}

function summarize(values: number[]): Record<string, number> {
  if (!values.length) return { count: 0, mean: 0, median: 0, p95: 0, min: 0, max: 0 };
  const sorted = [...values].sort((left, right) => left - right);
  const quantile = (q: number) => sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
  return {
    count: values.length,
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    median: quantile(0.5),
    p95: quantile(0.95),
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

function summarizeAgent(agent: Agent): Record<string, unknown> {
  const capHits = agent.observations.filter(item => item.simulations === agent.settings.simulations).length;
  return {
    label: agent.label,
    model: relative(ROOT, agent.modelPath).split(sep).join('/'),
    modelSha256: agent.modelSha256,
    moveRequests: agent.observations.length,
    modelMoves: agent.observations.filter(item => item.source === 'model').length,
    verifiedEndgameMoves: agent.observations.filter(item => item.source === 'search').length,
    warmupMs: agent.warmupMs,
    simulationCounts: summarize(agent.observations.map(item => item.simulations)),
    elapsedMs: summarize(agent.observations.map(item => item.elapsedMs)),
    simulationCapHits: capHits,
    simulationCapHitRate: agent.observations.length ? capHits / agent.observations.length : 0,
    pageErrors: agent.pageErrors,
    settings: agent.settings,
  };
}

async function bundledWorker(dist: string) {
  const assets = await readdir(resolve(dist, 'assets'));
  const workerAssets = assets.filter(name => /^ai\.worker-[A-Za-z0-9_-]+\.js$/.test(name));
  if (workerAssets.length !== 1) throw new Error(`Expected exactly one bundled ai.worker asset, found ${workerAssets.length}`);
  return workerAssets[0];
}

// One warmed production session, with a verified Easy reset before each game.
// The official harness serializes games so no other game consumes CPU during a timed Hard turn.
export async function createHardBrowserCandidate(distPath: string, modelPath: string, modelBytes: Buffer) {
  const dist = resolve(distPath);
  const workerAsset = await bundledWorker(dist);
  const assets = await hashDistAssets(dist);
  const modelSha256 = createHash('sha256').update(modelBytes).digest('hex');
  const server = await startStaticServer(dist);
  let browser: Browser | undefined;
  async function close() {
    try { await browser?.close(); } finally { await server.close(); }
  }
  try {
    browser = await chromium.launch({ headless: true });
    const agent = await makeAgent('challenger', modelPath, modelSha256, modelBytes, server.baseUrl, workerAsset, browser);
    const warm = await requestMove(agent, createGame(), 'easy');
    if (warm.source !== 'model' || !agent.modelLoads) throw new Error('Browser worker did not load the requested candidate model');
    await requestMove(agent, createGame(), 'hard');
    assertNoPageErrors([agent]);
    return {
      evidence: {
        modelSha256, workerSha256: assets['assets/' + workerAsset],
        distSha256: createHash('sha256').update(JSON.stringify(assets)).digest('hex'),
        chromiumVersion: browser.version(), settings: HARD_SETTINGS, warmupMs: warm.elapsedMs,
      },
      async beginGame() {
        await requestMove(agent, createGame(), 'easy');
        assertNoPageErrors([agent]);
      },
      async move(state: GameState) {
        const reply = await requestMove(agent, state, 'hard');
        assertNoPageErrors([agent]);
        return { move: reply.move!, source: reply.source, simulations: reply.simulations, elapsedMs: reply.elapsedMs };
      },
      close,
    };
  } catch (error) { await close(); throw error; }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const challengerPath = options.challenger;
  const championPath = options.champion;
  const outputPath = options.output;
  const progressPath = outputPath.endsWith('.json')
    ? `${outputPath.slice(0, -5)}.progress.json`
    : `${outputPath}.progress.json`;
  await Promise.all([access(challengerPath), access(championPath), access(resolve(DIST, 'index.html'))]);
  for (const path of [outputPath, progressPath]) {
    try {
      await access(path);
      throw new Error(`Refusing to overwrite existing file: ${path}`);
    } catch (error) {
      if (error instanceof Error && error.message.startsWith('Refusing to overwrite')) throw error;
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  }

  const workerAsset = await bundledWorker(DIST);
  const workerPath = resolve(DIST, 'assets', workerAsset);
  const [challengerBytes, championBytes, workerBytes] = await Promise.all([
    readFile(challengerPath), readFile(championPath), readFile(workerPath),
  ]);
  const hash = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
  const challengerSha256 = hash(challengerBytes);
  const championSha256 = hash(championBytes);
  const workerSha256 = hash(workerBytes);
  const championWorkerBytes = options.championWorker ? await readFile(options.championWorker) : undefined;
  if (championWorkerBytes) {
    const baselinePlan = JSON.parse(await readFile(resolve(dirname(options.championWorker!), 'plan.json'), 'utf8'));
    if (hash(championWorkerBytes) !== baselinePlan.baselineWorkerSha256 || baselinePlan.baseline.simulations !== 384 || baselinePlan.baseline.timeMs !== 650) throw new Error('Champion worker does not match the pinned baseline plan');
  }
  const workerComparison = { challengerSha256: workerSha256, championSha256: championWorkerBytes ? hash(championWorkerBytes) : workerSha256, challengerSettings: HARD_SETTINGS, championSettings: championWorkerBytes ? BASELINE_SETTINGS : HARD_SETTINGS };

  const sourcePaths = [
    resolve(ROOT, 'scripts/browser-arena.ts'),
    resolve(ROOT, 'scripts/arena-openings.ts'),
    resolve(ROOT, 'src/game.ts'),
    resolve(ROOT, 'src/ai.ts'),
    resolve(ROOT, 'src/endgame.ts'),
    resolve(ROOT, 'src/ai.worker.ts'),
    resolve(ROOT, 'src/onnx.ts'),
  ];
  const sourceHashes: Record<string, string> = {};
  for (const filePath of sourcePaths) {
    sourceHashes[relative(ROOT, filePath).split(sep).join('/')] = createHash('sha256')
      .update(await readFile(filePath)).digest('hex');
  }
  const distAssetHashes = await hashDistAssets();
  await mkdir(dirname(outputPath), { recursive: true });
  const initialProgress = {
    format: 'ultimate-tic-tac-toe-browser-paired-arena-progress-v1',
    status: 'starting',
    createdAt: new Date().toISOString(),
    output: outputPath,
    pairsTarget: options.pairs,
    pairsCompleted: 0,
    gamesCompleted: 0,
    seed: options.seed,
    openingSchedule: options.openingSchedule,
    settings: HARD_SETTINGS,
    workerComparison,
    challengerModel: { path: relative(ROOT, challengerPath).split(sep).join('/'), sha256: challengerSha256 },
    championModel: { path: relative(ROOT, championPath).split(sep).join('/'), sha256: championSha256 },
    sourceHashes,
    distAssetHashes,
    pairScores: [],
    perPair: [],
  };
  const initialProgressFile = await open(progressPath, 'wx');
  try {
    await initialProgressFile.writeFile(`${JSON.stringify(initialProgress, null, 2)}\n`, 'utf8');
  } finally {
    await initialProgressFile.close();
  }

  const staticServer = await startStaticServer();
  let browser: Browser | undefined;
  let challengerAgent: Agent | undefined;
  let championAgent: Agent | undefined;
  try {
    browser = await chromium.launch({ headless: true });
    challengerAgent = await makeAgent(
      'challenger', challengerPath, challengerSha256, challengerBytes,
      staticServer.baseUrl, workerAsset, browser,
    );
    championAgent = await makeAgent(
      'champion', championPath, championSha256, championBytes,
      staticServer.baseUrl, workerAsset, browser, championWorkerBytes,
    );

    // Load each exact pinned model session before scoring; record startup separately.
    const warmState = createGame();
    const challengerWarm = await requestMove(challengerAgent, warmState, 'easy');
    const championWarm = await requestMove(championAgent, warmState, 'easy');
    challengerAgent.warmupMs = challengerWarm.elapsedMs;
    championAgent.warmupMs = championWarm.elapsedMs;
    // Warm the actual Hard search path as well as loading the WASM sessions.
    // Exclude these requests from scored-game timing and simulation summaries.
    for (const agent of [challengerAgent, championAgent]) {
      await requestMove(agent, warmState, 'hard');
    }
    if (challengerWarm.source !== 'model' || championWarm.source !== 'model') {
      throw new Error('Model warm-up used a fallback evaluator');
    }
    assertNoPageErrors([challengerAgent, championAgent]);

    const random = new SeededRandom(options.seed);
    const seedForPair = pairSeedGenerator(options.openingSchedule, options.seed, () => random.nextUint32());
    const perPair: Record<string, unknown>[] = [];
    const pairScores: number[] = [];
    const gameScores: number[] = [];
    const saveProgress = async (status: 'running' | 'complete', finalOutput?: string): Promise<void> => {
      assertNoPageErrors([challengerAgent!, championAgent!]);
      const progress = {
        format: 'ultimate-tic-tac-toe-browser-paired-arena-progress-v1',
        status,
        createdAt: startedAt,
        updatedAt: new Date().toISOString(),
        output: outputPath,
        finalOutput: finalOutput ?? null,
        pairsTarget: options.pairs,
        pairsCompleted: perPair.length,
        gamesCompleted: gameScores.length,
        seed: options.seed,
        openingSchedule: options.openingSchedule,
        settings: HARD_SETTINGS,
        workerComparison,
        challengerModel: { path: relative(ROOT, challengerPath).split(sep).join('/'), sha256: challengerSha256 },
        championModel: { path: relative(ROOT, championPath).split(sep).join('/'), sha256: championSha256 },
        sourceHashes,
        distAssetHashes,
        pairScores: [...pairScores],
        perPair: [...perPair],
        timing: {
          challenger: summarizeAgent(challengerAgent!),
          champion: summarizeAgent(championAgent!),
        },
      };
      const temporaryPath = `${progressPath}.${process.pid}.${randomUUID()}.tmp`;
      await writeFile(temporaryPath, `${JSON.stringify(progress, null, 2)}\n`, { flag: 'wx' });
      await rename(temporaryPath, progressPath);
    };
    const startedAt = new Date().toISOString();
    for (let pair = 0; pair < options.pairs; pair++) {
      const pairStarted = performance.now();
      const openingSeed = seedForPair(pair);
      const opening = makeOpening(openingSeed);
      let asX: GameResult;
      let asO: GameResult;
      if (pair % 2 === 0) {
        asX = await playGame(opening.state, 'X', challengerAgent, championAgent);
        asO = await playGame(opening.state, 'O', challengerAgent, championAgent);
      } else {
        asO = await playGame(opening.state, 'O', challengerAgent, championAgent);
        asX = await playGame(opening.state, 'X', challengerAgent, championAgent);
      }
      const challengerAsX = score(asX.winner, 'X');
      const challengerAsO = score(asO.winner, 'O');
      const pairScore = (challengerAsX + challengerAsO) / 2;
      pairScores.push(pairScore);
      gameScores.push(challengerAsX, challengerAsO);
      perPair.push({
        pair,
        openingSeed,
        openingMoves: opening.moves,
        gameOrder: pair % 2 === 0 ? ['challenger-X', 'challenger-O'] : ['challenger-O', 'challenger-X'],
        challengerAsX: { ...asX, score: challengerAsX },
        challengerAsO: { ...asO, score: challengerAsO },
        pairScore,
        elapsedMs: performance.now() - pairStarted,
      });
      assertNoPageErrors([challengerAgent, championAgent]);
      await saveProgress('running');
      const runningMean = pairScores.reduce((sum, value) => sum + value, 0) / pairScores.length;
      process.stderr.write(`browser arena: ${pair + 1}/${options.pairs} pairs; current mean ${runningMean.toFixed(3)}\n`);
    }

    const pairedMean = pairScores.reduce((sum, value) => sum + value, 0) / pairScores.length;
    const variance = pairScores.length > 1
      ? pairScores.reduce((sum, value) => sum + (value - pairedMean) ** 2, 0) / (pairScores.length - 1)
      : 0;
    const margin = pairScores.length > 1 ? 1.96 * Math.sqrt(variance / pairScores.length) : 1;
    const wins = gameScores.filter(value => value === 1).length;
    const draws = gameScores.filter(value => value === 0.5).length;
    const losses = gameScores.filter(value => value === 0).length;
    const pageMeta = await challengerAgent.page.evaluate(() => ({
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      platform: navigator.platform,
    }));
    const result = {
      format: 'ultimate-tic-tac-toe-browser-paired-arena-v1',
      createdAt: new Date().toISOString(),
      runtime: {
        browser: 'Chromium',
        browserVersion: browser.version(),
        headless: true,
        ...pageMeta,
        workerAsset: relative(ROOT, workerPath).split(sep).join('/'),
        workerSha256,
      },
      workerComparison,
      settings: {
        ...HARD_SETTINGS,
        parallelGames: 1,
        searchSeed: 1,
        openingPlyCount: 2,
        openingProtocol: 'same seeded two-ply random opening in both games of each pair; challenger and champion swap X/O',
        gameOrder: 'alternate challenger-X-first and challenger-O-first across pairs',
        warmup: 'Initial Easy and Hard warm-up. Easy model probes before each game clear the Hard subtree; a further probe follows each verified endgame move. All probes are excluded from scored timings.',
        valueModes: { challenger: 'model', champion: 'model' },
      },
      challengerModel: { model: relative(ROOT, challengerPath).split(sep).join('/'), sha256: challengerSha256 },
      championModel: { model: relative(ROOT, championPath).split(sep).join('/'), sha256: championSha256 },
      pairs: options.pairs,
      games: gameScores.length,
      seed: options.seed,
      openingSchedule: options.openingSchedule,
      challenger: {
        wins,
        draws,
        losses,
        games: gameScores.length,
        scoreRate: pairedMean,
      },
      pairScores,
      pairedConfidence95: {
        lower: Math.max(0, pairedMean - margin),
        upper: Math.min(1, pairedMean + margin),
      },
      timing: {
        note: 'Latency is measured in the page from worker postMessage through reply; the warm-up request is excluded.',
        challenger: summarizeAgent(challengerAgent),
        champion: summarizeAgent(championAgent),
      },
      sourceHashes,
      distAssetHashes,
      perPair,
    };

    const handle = await open(outputPath, 'wx');
    try {
      await handle.writeFile(`${JSON.stringify(result, null, 2)}\n`, 'utf8');
    } finally {
      await handle.close();
    }
    await saveProgress('complete', outputPath);
    process.stdout.write(`${JSON.stringify({ output: outputPath, pairs: options.pairs, scoreRate: pairedMean, pairedConfidence95: result.pairedConfidence95 }, null, 2)}\n`);
  } finally {
    await Promise.allSettled([
      challengerAgent?.page.evaluate(() => (window as any).__browserArena?.dispose()),
      championAgent?.page.evaluate(() => (window as any).__browserArena?.dispose()),
      challengerAgent?.context.close(),
      championAgent?.context.close(),
    ]);
    await browser?.close();
    await staticServer.close();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
