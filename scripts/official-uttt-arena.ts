/** Offline match against uttt.ai's published stage-2 search. See --help. */
import { createHash, randomUUID } from 'node:crypto';
import { link, mkdir, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { basename, dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { pairSeedGenerator, parseOpeningSchedule } from './arena-openings.ts';
import type { InferenceSession, Tensor } from 'onnxruntime-node';
import { advanceSubtree, encodeState, legalMoves, search, type Evaluator, type Node } from '../src/ai.ts';
import { solveEndgame } from '../src/endgame.ts';
import { createGame, play, type GameState, type Player } from '../src/game.ts';

const hardEndgameSettings = { maxEmptySquares: 14, maxNodes: 16_000, timeMs: 250 };
const suiteSha256 = 'feb92e42ff53472bd8a4a04f4e7d31254c1d87cbdb7ab9f40d110932d001acbe';
const upstreamCommit = '622dae66b18198f6f12aef0a0bccd5bd40de2d8e';
// Git blob IDs at this revision. Checking source bytes also supports extracted release trees.
const officialSources: Record<string, string> = {
  'game/Action.js': '1834e8b4aa51bfaa5b80d4cc4631370b111ab7b2',
  'game/UltimateTicTacToe.js': 'e369850d74c0aa82b3dc933f98a8ec166494ff93',
  'game/UltimateTicTacToeError.js': 'ae9f9a17a08ee52019f65c0d2de2ac7fa29d1612',
  'game/constants.js': '094213657b8d61f50fe4193e03b10969b5d48af4',
  'game/helpers.js': '49aedd93332eb767d98a44a2a4049e56b1bed2de',
  'global/device.js': '2ec57f005528e76e861f87bcb4fb79c55459b189',
  'global/sleeping.js': 'df53f0df8547a431b4f654d129fc51bf45ba4661',
  'selfplay/NeuralMonteCarloTreeSearch.js': '2bb6dbfdd56de2f4aefddae8fb14f26d251489b8',
  'selfplay/NeuralMonteCarloTreeSearchError.js': '2f7f6cae14cdcd6d012c7d24e37a9e5db73089cb',
  'selfplay/helpers.js': '7128462d8022aae60715e0f9882cbc9fe0b359f6',
};
// The dynamic export differs in batch handling; both hashes were checked
// against the published weights and raw-model fixture outputs.
const officialModelHashes = new Set([
  '81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2',
  'b4f92e809e1eaa2053b202202d3853e1ed10be6d4c6980c2c46339f6e14f70b7',
]);
const sha256 = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

function random(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

class InputTensor {
  data: Float32Array;
  constructor(_type: string, data: Float32Array) { this.data = data; }
}

async function loadOfficial(upstream: string) {
  const { build } = await import('esbuild');
  const source = resolve(upstream, 'website/src');
  const result = await build({
    stdin: { contents: "export {default as Game} from './game/UltimateTicTacToe.js'; export {default as Engine} from './selfplay/NeuralMonteCarloTreeSearch.js'", resolveDir: source },
    write: false, bundle: true, format: 'cjs', target: 'node22', platform: 'node', external: ['onnxruntime-web'],
    define: { 'Math.random': 'gameRandom' },
    plugins: [{ name: 'verify-upstream', setup(builder) {
      builder.onLoad({ filter: /\.js$/ }, async args => {
        const path = relative(source, args.path);
        const bytes = await readFile(args.path);
        const blob = createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
        if (blob !== officialSources[path]) throw new Error(`Upstream file does not match ${upstreamCommit}: ${path}`);
        return { contents: bytes, resolveDir: dirname(args.path), loader: 'js' };
      });
    } }],
  });
  // Each game's lexical RNG is independent of batching and interleaving. The website's
  // download/timer UI is unused. Its unmodified simulate() receives a local inference session.
  const load = new Function('module', 'require', 'gameRandom', 'navigator', result.outputFiles[0].text);
  return (gameRandom: () => number) => {
    const module = { exports: {} as any };
    load(module, (name: string) => {
      if (name !== 'onnxruntime-web') throw new Error(`Unexpected import ${name}`);
      return { Tensor: InputTensor };
    }, gameRandom, { userAgent: 'offline-arena' });
    return module.exports;
  };
}

interface WorkerMove { move: number; source: 'model' | 'search'; simulations: number; elapsedMs: number }
interface BrowserPlayer { beginGame(): Promise<void>; move(state: GameState): Promise<WorkerMove> }

interface Prediction { policy: Float32Array; value: number }
interface Request { input: Float32Array; resolve: (prediction: Prediction) => void; reject: (reason: unknown) => void }

async function inference(path: string, bytes: Buffer, official: boolean, provider: string) {
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: [provider], intraOpNumThreads: 1, interOpNumThreads: 1 });
  const [input, policy, value] = official ? ['input', 'policy_logits', 'state_value'] : ['state', 'policy', 'value'];
  const shape = session.inputMetadata[0].shape;
  const dims = official ? [4, 9, 9] : [280];
  const batchDimension = shape[0];
  if (session.inputNames.join() !== input || shape.slice(1).join() !== dims.join() ||
      !session.outputNames.includes(policy) || !session.outputNames.includes(value) ||
      (typeof batchDimension === 'number' && batchDimension !== 1)) {
    await session.release(); throw new Error(`Unsupported ONNX interface for ${path}`);
  }
  const pending: Request[] = [];
  async function flush() {
    const requests = pending.splice(0);
    // An unmodified published model can have a fixed batch of one. For dynamic
    // models, batch only across games. Every tree keeps exactly one outstanding leaf.
    const groupSize = batchDimension === 1 ? 1 : requests.length;
    for (let start = 0; start < requests.length; start += groupSize) {
      const group = requests.slice(start, start + groupSize);
      let tensor: Tensor | undefined;
      let result: InferenceSession.ReturnType | undefined;
      try {
        const width = group[0].input.length;
        const data = new Float32Array(group.length * width);
        group.forEach((item, index) => data.set(item.input, index * width));
        tensor = new ort.Tensor('float32', data, [group.length, ...dims]);
        result = await session.run({ [input]: tensor }, [policy, value]);
        const logits = result[policy].data as Float32Array, values = result[value].data as Float32Array;
        if (logits.length !== 81 * group.length || values.length !== group.length ||
            !logits.every(Number.isFinite) || !values.every(v => Number.isFinite(v) && Math.abs(v) <= 1.001)) {
          throw new Error(`Nonfinite or malformed output from ${path}`);
        }
        group.forEach((item, i) => item.resolve({ policy: logits.slice(81 * i, 81 * (i + 1)), value: values[i] }));
      } catch (error) { group.forEach(item => item.reject(error)); }
      finally { tensor?.dispose(); if (result) Object.values(result).forEach(item => item.dispose()); }
    }
  }
  return {
    sha256: sha256(bytes), name: basename(path), inputShape: shape,
    predict(data: Float32Array): Promise<Prediction> {
      return new Promise((resolve, reject) => {
        pending.push({ input: data, resolve, reject });
        if (pending.length === 1) setImmediate(flush);
      });
    }, close: () => session.release(),
  };
}

function opening(seed: number) {
  const next = random(seed), moves: number[] = [];
  let state = createGame();
  while (moves.length < 2) {
    const legal = legalMoves(state), move = legal[Math.floor(next() * legal.length)];
    moves.push(move); state = play(state, Math.floor(move / 9), move % 9);
  }
  return moves;
}

// All 93 digits are significant, including subboard outcomes, next player,
// forced subboard and global result. Distinct histories can reach the same state.
function projectStateDigits(state: GameState) {
  const digit = (mark: 'X' | 'O' | 'draw' | null) => mark === null ? 0 : mark === 'X' ? 1 : mark === 'O' ? 2 : 3;
  return [...state.boards.flat().map(digit), ...state.results.map(digit), digit(state.turn),
    state.forced ?? 9, digit(state.winner)].join('');
}

export function recoverPublishedOpening(target: string): number[] {
  if (!/^[012]{81}[0123]{9}[12][0-9][0123]$/.test(target)) throw new Error('Invalid published state encoding');
  const plies = [...target.slice(0, 81)].filter(digit => digit !== '0').length;
  if (plies < 1 || plies > 6) throw new Error('Unexpected published opening depth');
  function visit(state: GameState, history: number[]): number[] | null {
    if (history.length === plies) return projectStateDigits(state) === target ? history : null;
    const mark = state.turn === 'X' ? '1' : '2';
    for (const move of legalMoves(state)) {
      if (target[move] !== mark) continue;
      const result = visit(play(state, Math.floor(move / 9), move % 9), [...history, move]);
      if (result !== null) return result;
    }
    return null;
  }
  const history = visit(createGame(), []);
  if (history === null) throw new Error('Published opening has no legal project history');
  return history;
}

export async function publishedSuite(upstream: string) {
  const bytes = await readFile(resolve(upstream, 'utttpy/selfplay/evaluation_uttt_states.py'));
  if (sha256(bytes) !== suiteSha256) throw new Error('Published opening suite hash mismatch');
  const states = Array.from(bytes.toString('utf8').matchAll(/^    "([0-9]{93})",$/gm), match => match[1]);
  if (states.length !== 50 || new Set(states).size !== 50) throw new Error('Expected 50 distinct published opening positions');
  return states.map(state => ({ state, moves: recoverPublishedOpening(state) }));
}

export async function playGame(openingMoves: number[], seat: Player, seed: number, cap: number,
                        officialCap: number, hardEndgame: boolean,
                        load: Awaited<ReturnType<typeof loadOfficial>>, evaluate: Evaluator | undefined,
                        opponent: Pick<Awaited<ReturnType<typeof inference>>, 'predict'>, publishedState?: string,
                        browser?: BrowserPlayer) {
  if (!browser && !evaluate) throw new Error('Missing candidate evaluator');
  const next = random(seed), officialRandom = random(seed ^ 0x6a09e667);
  const { Game, Engine } = load(officialRandom);
  const game = new Game();
  let state = createGame();
  const advance = (move: number) => {
    const action = game.getLegalActions().find((item: {index: number}) => item.index === move);
    if (!action) throw new Error(`Upstream rules rejected move ${move}`);
    game.execute(action);
    state = play(state, Math.floor(move / 9), move % 9);
    const winner = [null, 'X', 'O', 'draw'][game.result];
    if (game.getLegalIndexes().join() !== legalMoves(state).join() || state.winner !== winner ||
        game.nextSymbol !== (state.turn === 'X' ? 1 : 2)) throw new Error(`Rules disagree after move ${move}`);
  };
  openingMoves.forEach(advance);
  if (publishedState !== undefined && game.state.join('') !== publishedState) {
    throw new Error('Replayed position differs from the complete published upstream state');
  }
  const engine = new Engine(game.clone(), officialCap, 2);
  engine.policyValueNetSession = { async run({ input }: {input: InputTensor}) {
    const p = await opponent.predict(input.data);
    return { policy_logits: { data: p.policy }, state_value: { data: [p.value] } };
  } };
  const moves = [...openingMoves];
  const browserTurns: WorkerMove[] = [];
  if (browser) await browser.beginGame();
  let root: Node | undefined, lastMove: number | undefined;
  let ourVisits = 0, theirVisits = 0, ourReuse = 0, theirReuse = 0, tacticalMoves = 0;
  let exactMoves = 0, retainedEndgameMoves = 0;
  while (!state.winner) {
    let move: number;
    if (state.turn === seat && browser) {
      const reply = await browser.move(state);
      move = reply.move;
      ourVisits += reply.simulations;
      if (reply.source === 'search') exactMoves++;
      browserTurns.push(reply);
    } else if (state.turn === seat) {
      const endgame = hardEndgame ? solveEndgame(state, hardEndgameSettings) : null;
      if (endgame && endgame.move !== null && (endgame.solved || endgame.guaranteedValue !== null)) {
        move = endgame.move;
        exactMoves++;
        if (!endgame.solved) retainedEndgameMoves++;
        root = undefined;
        lastMove = undefined;
      } else {
        const reused = lastMove === undefined ? undefined : advanceSubtree(root, lastMove, moves[moves.length - 1]);
        const inheritedVisits = reused?.visits ?? 0;
        const result = await search(state, { simulations: cap, timeMs: null, tacticalProof: true,
          seed: Math.floor(next() * 0x100000000), evaluate, reusedRoot: reused });
        if (result.move === null) throw new Error('Project search returned no move');
        if (result.root && result.simulations !== cap) throw new Error(`Project search hit its time limit at ${result.simulations}/${cap}`);
        if (!result.root) tacticalMoves++;
        ourVisits += result.simulations;
        if (result.root && result.root === reused) ourReuse += inheritedVisits;
        move = result.move; root = result.root; lastMove = move;
      }
    } else {
      const root = engine.tree.root;
      theirReuse += root.visitCount;
      while (root.visitCount < officialCap) await engine.simulate();
      theirVisits += root.visitCount;
      const highest = Math.max(...root.childNodes.map((child: any) => child.visitCount));
      const best = root.childNodes.filter((child: any) => child.visitCount === highest);
      move = best[Math.floor(officialRandom() * best.length)].action.index;
    }
    if (!legalMoves(state).includes(move)) throw new Error(`Illegal move ${move}`);
    moves.push(move); advance(move); engine.tree.synchronize(game);
  }
  const score = state.winner === 'draw' ? 0.5 : state.winner === seat ? 1 : 0;
  return { seat, score, winner: state.winner, moves, ourVisits, theirVisits, theirReuse, exactMoves,
    ourReuse: browser ? null : ourReuse, tacticalMoves: browser ? null : tacticalMoves,
    retainedEndgameMoves: browser ? null : retainedEndgameMoves, ...(browser ? { browserTurns } : {}) };
}

async function main() {
  const { values } = parseArgs({ options: {
    model: { type: 'string' }, 'official-model': { type: 'string' }, upstream: { type: 'string' }, output: { type: 'string' },
    'browser-dist': { type: 'string' }, 'opening-suite': { type: 'string', default: 'generated' }, 'opening-schedule': { type: 'string' }, 'official-simulations': { type: 'string' }, 'hard-endgame': { type: 'boolean', default: false },
    pairs: { type: 'string', default: '50' }, simulations: { type: 'string' }, seed: { type: 'string', default: '20261126' },
    parallel: { type: 'string' }, provider: { type: 'string', default: 'cpu' }, help: { type: 'boolean' },
  } });
  if (values.help) { console.log('node scripts/official-uttt-arena.ts --model candidate.onnx --official-model policy_value_net_stage2.onnx --upstream /path/to/utttai --output new-report.json [--pairs 50 --simulations 256 --official-simulations 1000 --hard-endgame --seed 20261126 --parallel 4 --provider cpu|cuda --opening-schedule legacy|independent --opening-suite generated|published] [--browser-dist compiled-site (production Hard worker; --parallel 1)]'); return; }
  for (const k of ['model', 'official-model', 'upstream', 'output'] as const) if (!values[k]) throw new Error(`Missing --${k}`);
  const integer = (key: 'pairs'|'simulations'|'official-simulations'|'seed'|'parallel', min: number, max: number) => {
    const n = Number(values[key]);
    if (!Number.isSafeInteger(n) || n < min || n > max) throw new Error(`Invalid --${key}`);
    return n;
  };
  const browserDist = values['browser-dist'];
  if (browserDist !== undefined && !browserDist.trim()) throw new Error('Expected a compiled site for --browser-dist');
  if (browserDist && (values.simulations !== undefined || values['hard-endgame'])) {
    throw new Error('The production worker supplies its own Hard settings; omit --simulations and --hard-endgame with --browser-dist');
  }
  const cap = browserDist ? 384 : values.simulations === undefined ? 256 : integer('simulations', 2, 10000);
  const pairs = integer('pairs', 1, 1000000), seed = integer('seed', 0, 0xffffffff);
  const officialCap = values['official-simulations'] === undefined ? cap : integer('official-simulations', 2, 10000);
  const hardEndgame = !!browserDist || values['hard-endgame'];
  const parallel = values.parallel === undefined ? (browserDist ? 1 : 4) : integer('parallel', 1, 128);
  if (browserDist && parallel !== 1) throw new Error('Timed browser evaluation requires --parallel 1');
  const provider = values.provider!;
  if (!['cpu', 'cuda'].includes(provider)) throw new Error('Expected --provider cpu or cuda');
  const openingSuite = values['opening-suite'];
  if (openingSuite !== 'generated' && openingSuite !== 'published') throw new Error('Expected --opening-suite generated or published');
  if (openingSuite === 'published' && pairs !== 50) throw new Error('The published suite requires exactly 50 pairs');
  const openingSchedule = parseOpeningSchedule(values['opening-schedule']);
  const output = resolve(values.output!);
  await mkdir(dirname(output), { recursive: true });
  const existing = await stat(output).catch(error => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  });
  if (existing) throw new Error(`Report already exists: ${output}`);
  const [ourBytes, officialBytes] = await Promise.all([readFile(values.model!), readFile(values['official-model']!)]);
  const officialDigest = sha256(officialBytes);
  if (!officialModelHashes.has(officialDigest)) {
    throw new Error(`Unrecognized official uttt.ai Stage 2 model SHA-256: ${officialDigest}`);
  }
  const published = openingSuite === 'published' ? await publishedSuite(values.upstream!) : null;
  const progress = `${output}.progress.json`;
  await writeFile(progress, JSON.stringify({ status: 'starting' }), { flag: 'wx' });
  const [load, ours, theirs] = await Promise.all([loadOfficial(values.upstream!), browserDist ? undefined : inference(values.model!, ourBytes, false, provider), inference(values['official-model']!, officialBytes, true, provider)]);
  const next = random(seed);
  const seedForPair = pairSeedGenerator(openingSchedule, seed, () => Math.floor(next() * 0x100000000));
  const rows = Array.from({ length: pairs }, (_, pair) => {
    const gameSeed = seedForPair(pair);
    // Native/browser xorshift maps a zero opening seed to this nonzero state.
    const openingSeed = openingSchedule === 'independent' && gameSeed === 0 ? 0x9e3779b9 : gameSeed;
    return { pair, seed: gameSeed, opening: published ? published[pair].moves : opening(openingSeed), games: [] as Awaited<ReturnType<typeof playGame>>[] };
  });
  const openingPlyDistribution: Record<number, number> = {};
  for (const row of rows) openingPlyDistribution[row.opening.length] = (openingPlyDistribution[row.opening.length] ?? 0) + 1;
  let browser: Awaited<ReturnType<typeof import('./browser-arena.ts')['createHardBrowserCandidate']>> | undefined;
  const evidence = {
    candidateBackend: browserDist ? 'chromium-worker' : 'native',
    openingSuite, ...(published ? { openingSuiteSha256: suiteSha256, openingPositions: published.length, openingPlyDistribution } : {}),
    format: 'official-uttt-arena-v1', upstreamCommit, provider, pairs, seed, openingSchedule, parallel, maxRootVisits: cap, officialMaxRootVisits: officialCap,
    hardEndgame, hardEndgameSettings: hardEndgame ? hardEndgameSettings : null,
    officialProtocol: 'Published website NMCTS simulation; exploration 2.0, 0.01 prior floor, subtree synchronized each ply, tournament max visits with seeded random ties. Published quick / strong caps are 1000 / 10000.',
    candidateProtocol: browserDist
      ? 'Production Chromium Hard worker, 384 root visits and 220 ms PUCT deadline, separate bounded Hard endgame solver. A warmed session resets before every game. The official opponent uses its fixed visit cap.'
      : 'Project PUCT with tactical proof and subtree carryover. Both visit caps count prior root work. Candidate PUCT has a fixed visit cap and no wall-clock deadline. ' +
      (hardEndgame ? 'Hard endgame solver runs first under its own node and time limits.' : 'Separate endgame solver disabled.'),
    projectEndgameSha256: sha256(await readFile(new URL('../src/endgame.ts', import.meta.url))),
    openings: published
      ? 'Fixed 50-position uttt.ai evaluation suite in published order. The seed controls search ties. Each position is played from both seats. Draws score 0.5.'
      : 'Two seeded uniform legal plies per pair; swap sides and reuse game seed. Draws score 0.5.',
    candidate: { file: basename(values.model!), sha256: sha256(ourBytes), ...(ours ? { inputShape: ours.inputShape } : {}) },
    official: { file: theirs.name, sha256: theirs.sha256, inputShape: theirs.inputShape },
    harnessSha256: sha256(await readFile(fileURLToPath(import.meta.url))),
    openingScheduleSha256: sha256(await readFile(new URL('./arena-openings.ts', import.meta.url))),
    projectSearchSha256: sha256(await readFile(new URL('../src/ai.ts', import.meta.url))),
    projectRulesSha256: sha256(await readFile(new URL('../src/game.ts', import.meta.url))),
  };
  let cursor = 0;
  let failed = false;
  // A single writer serializes progress snapshots so finishes out of order cannot rewind the file.
  let writing = Promise.resolve();
  const run = async () => {
    while (!failed && cursor < 2 * pairs) {
      const index = cursor++, row = rows[Math.floor(index / 2)];
      const seat = (index + row.pair) % 2 === 0 ? 'X' : 'O';
      const game = await playGame(row.opening, seat, row.seed, cap, officialCap, hardEndgame, load, ours ? state => ours.predict(encodeState(state)) : undefined, theirs, published?.[row.pair].state, browser);
      row.games.push(game);
      if (row.games.length === 2) {
        const done = rows.filter(item => item.games.length === 2);
        console.error(`Completed ${done.length}/${pairs} pairs`);
        const json = JSON.stringify({ status: 'running', ...evidence, completed: done.length, results: done });
        writing = writing.then(async () => { const tmp = `${progress}.${randomUUID()}`; await writeFile(tmp, json); await rename(tmp, progress); });
        await writing;
      }
    }
  };
  try {
    if (browserDist) {
      const { createHardBrowserCandidate } = await import('./browser-arena.ts');
      browser = await createHardBrowserCandidate(browserDist, values.model!, ourBytes);
      Object.assign(evidence, { browserCandidate: browser.evidence,
        browserHarnessSha256: sha256(await readFile(new URL('./browser-arena.ts', import.meta.url))) });
    }
    const workers = await Promise.allSettled(Array.from({ length: Math.min(parallel, 2 * pairs) }, () =>
      run().catch(error => { failed = true; throw error; })));
    for (const worker of workers) if (worker.status === 'rejected') throw worker.reason;
    const pairScores = rows.map(row => row.games.reduce((sum, g) => sum + g.score / 2, 0));
    const score = pairScores.reduce((sum, x) => sum + x, 0) / pairs;
    const varianceSum = pairScores.reduce((sum, x) => sum + (x-score)**2, 0);
    const margin = pairs < 2 || varianceSum === 0 ? null : 1.96 * Math.sqrt(varianceSum / ((pairs-1)*pairs));
    // Distribution-free 95% interval for the sampled opening-pair score (in [0,1]).
    // The usual normal approximation degenerates with identical pair results.
    const hoeffdingMargin = Math.sqrt(Math.log(40) / (2 * pairs));
    const games = rows.flatMap(row => row.games);
    const scores = games.map(game => game.score);
    const exactMoves = games.reduce((sum, game) => sum + game.exactMoves, 0);
    const retainedEndgameMoves = browser ? null : games.reduce((sum, game) => sum + game.retainedEndgameMoves!, 0);
    const summary = { exactMoves, retainedEndgameMoves, wins: scores.filter(x => x === 1).length, draws: scores.filter(x => x === 0.5).length,
      losses: scores.filter(x => x === 0).length, score, pairedNormal95: published || margin === null ? null : [Math.max(0, score-margin), Math.min(1, score+margin)],
      pairedHoeffding95: published ? null : [Math.max(0, score-hoeffdingMargin), Math.min(1, score+hoeffdingMargin)] };
    const json = JSON.stringify({ status: 'complete', ...evidence, ...summary, pairScores, results: rows }, null, 2);
    const tmp = `${output}.${randomUUID()}`;
    await writeFile(tmp, json); try { await link(tmp, output); } finally { await unlink(tmp); }
    await unlink(progress);
    console.log(JSON.stringify(summary));
  } finally { await Promise.all([ours?.close(), theirs.close(), browser?.close()]); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
