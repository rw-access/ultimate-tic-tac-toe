// Both engines take turns in warmed single-threaded WASM workers in one Chromium instance.
import { createHash, randomUUID } from 'node:crypto';
import { link, mkdir, readFile, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHardBrowserCandidate } from './browser-arena.ts';
import { createPublishedSearch, projectStateDigits, publishedSuite, verifiedOfficialBundle } from './official-uttt-arena.ts';
import { pairSeedGenerator } from './arena-openings.ts';
import { createGame, play, type GameState, type Player } from '../src/game.ts';
import { legalMoves } from '../src/ai.ts';

const websiteModel = '81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2';
const digest = (bytes: Buffer | string) => createHash('sha256').update(bytes).digest('hex');
const absolute = (name: string) => fileURLToPath(new URL(name, import.meta.url));

async function peer(project: Awaited<ReturnType<typeof createHardBrowserCandidate>>, dist: string,
                    upstream: string, original: Buffer) {
  const context = await project.browser.newContext();
  const page = await context.newPage();
  const prefix = new URL('arena-timed/', project.baseUrl).href;
  const checked = await verifiedOfficialBundle(upstream);
  const worker = await readFile(absolute('./timed-official-worker.mjs'));
  const runtime = await readFile(absolute('../node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'));
  const sources = new Map<string, {body: string | Buffer; contentType: string}>([
    ['worker.mjs', { body: worker, contentType: 'text/javascript' }],
    ['ort.mjs', { body: runtime, contentType: 'text/javascript' }],
    ['published-engine.mjs', { body: 'export default function(module,require,gameRandom,navigator){\n' + checked + '\n}', contentType: 'text/javascript' }],
    ['original.onnx', { body: original, contentType: 'application/octet-stream' }],
  ]);
  const files = await readdir(resolve(dist, 'assets'));
  function asset(pattern: RegExp) {
    const matches = files.filter(name => pattern.test(name));
    if (matches.length !== 1) throw new Error('Missing or ambiguous WASM asset in the project build');
    return new URL('assets/' + matches[0], project.baseUrl).href;
  }
  const runtimeFiles = { wasm: asset(/^ort-wasm-simd-threaded-[\w-]+\.wasm$/), mjs: asset(/^ort-wasm-simd-threaded-[\w-]+\.mjs$/) };
  await context.route(prefix + '**', route => {
    const item = sources.get(route.request().url().slice(prefix.length));
    return item ? route.fulfill({ status: 200, ...item }) : route.abort();
  });
  await page.goto(project.baseUrl, { waitUntil: 'load' });
  await page.evaluate(({ prefix }) => {
    const worker = new Worker(prefix + 'worker.mjs', { type: 'module' });
    let id = 0;
    let pending: ((value: any) => void) | undefined;
    let fail: ((error: Error) => void) | undefined;
    let fatal: Error | undefined;
    worker.onerror = event => {
      fatal = new Error(event.message || 'Original engine worker failed');
      fail?.(fatal);
    };
    worker.onmessage = event => {
      if (event.data.id !== id || !pending) { fail?.(new Error('Unmatched original worker response')); return; }
      if (event.data.error) fail?.(new Error(event.data.error));
      else pending(event.data.result);
    };
    (window as any).__timedOriginal = (message: any) => new Promise((resolve, reject) => {
      if (fatal || pending) { reject(fatal ?? new Error('Overlapping worker requests')); return; }
      const start = performance.now();
      const timeout = setTimeout(() => finish(reject, new Error('Timed original worker request expired')), 90000);
      function finish(done: Function, value: any) { clearTimeout(timeout); pending = undefined; fail = undefined; done(value); }
      pending = result => finish(resolve, { result, elapsedMs: performance.now() - start });
      fail = error => finish(reject, error);
      worker.postMessage({ id: ++id, message });
    });
  }, {prefix});
  const send = (message: any): Promise<any> => page.evaluate(msg => (window as any).__timedOriginal(msg), message);
  const loaded = (await send({ type: 'load', runtimeFiles, sha256: websiteModel })).result;
  if (loaded.threads !== 1 || loaded.sha256 !== websiteModel) throw new Error('Unverified original runtime');
  return { send, evidence: { sourceBundleSha256: digest(checked), adapterSha256: digest(worker),
    runtimeModuleSha256: digest(runtime), modelSha256: websiteModel, wasmThreads: 1 }, close: () => context.close() };
}

function position(moves: number[]) {
  return moves.reduce((state, m) => play(state, Math.floor(m / 9), m % 9), createGame());
}

async function tournament() {
  const { values } = parseArgs({ options: {
    model: {type:'string'}, 'official-model': {type:'string'}, upstream: {type:'string'}, dist: {type:'string'},
    output: {type:'string'}, pairs: {type:'string', default:'50'}, seed: {type:'string', default:'26315110'},
  } });
  for (const k of ['model','official-model','upstream','dist','output'] as const) {
    if (!values[k]) throw new Error('Missing --' + k);
  }
  const upstream = resolve(values.upstream!), dist = resolve(values.dist!), out = resolve(values.output!);
  const original = await readFile(values['official-model']!), model = await readFile(values.model!);
  if (digest(original) !== websiteModel) throw new Error('Use the original fixed-batch website model');
  const pairs = Number(values.pairs), seed = Number(values.seed);
  if (!Number.isInteger(pairs) || pairs < 1 || pairs > 50 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Invalid pairs or seed');
  const suite = await publishedSuite(upstream);
  const seedForPair = pairSeedGenerator('independent', seed, () => { throw new Error('Expected independent schedule'); });
  const progress = out + '.progress.json';
  const spec = { format: 'uttt-browser-timed-original-v1', seed, pairs, projectModelSha256: digest(model),
    originalModelSha256: websiteModel, projectCap: 384, originalCap: 1000, projectSearchMs: 220, originalSearchMs: 220,
    harnessSha256: digest(await readFile(fileURLToPath(import.meta.url))) };
  async function existing(path: string) {
    try { return JSON.parse(await readFile(path, 'utf8')); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error; }
  }
  if (await existing(out)) throw new Error('The completed output already exists');
  const saved = await existing(progress);
  if (saved && JSON.stringify(saved.spec) !== JSON.stringify(spec)) throw new Error('Progress was made with a different protocol');
  const results: any[] = saved?.results ?? [];
  await mkdir(dirname(out), {recursive:true});
  const project = await createHardBrowserCandidate(dist, values.model!, model);
  try {
    const official = await peer(project, dist, upstream, original);
    const reference = await createPublishedSearch(upstream, values['official-model']!, 'cpu');
    const fixtures = [];
    try {
      for (const index of [0, 17, 36]) {
        const moves = suite[index].moves, state = position(moves), fixtureSeed = 93671 + index;
        await official.send({type:'reset', opening:moves, seed:fixtureSeed});
        const budget = 16;
        const browser = (await official.send({type:'fixture', moves, state:projectStateDigits(state), visits:budget})).result;
        const native = await reference.analyse(state, budget, fixtureSeed);
        const qError = Math.max(...native.q.map((v, i) => Math.abs(v - browser.q[i])));
        if (native.n.join() !== browser.n.join() || qError > .0002 || Math.abs(native.v - browser.v) > .0002) {
          throw new Error('Original browser simulation and native reference diverged at fixture ' + index + ': ' + JSON.stringify({native,browser}));
        }
        fixtures.push({openingIndex:index, visits:budget, maximumActionValueError:qError, rootValueError:Math.abs(native.v-browser.v)});
      }
    } finally { await reference.close(); }
    const evidence = { project:project.evidence, original:official.evidence, fixtures, upstreamCommit:reference.upstreamCommit };
    if (saved && (saved.evidence.project.workerSha256 !== evidence.project.workerSha256 ||
        saved.evidence.project.distSha256 !== evidence.project.distSha256 || saved.evidence.project.chromiumVersion !== evidence.project.chromiumVersion ||
        JSON.stringify(saved.evidence.original) !== JSON.stringify(evidence.original))) throw new Error('Worker or runtime changed since progress was recorded');
    for (let i=0;i<results.length;i++) {
      if (i>=pairs || results[i].pair!==i || results[i].seed!==seedForPair(i) || results[i].opening.join()!==suite[i].moves.join() || results[i].games.length!==2) throw new Error('Invalid saved opening pair');
    }
    for (let i=results.length;i<pairs;i++) {
      const opening = suite[i].moves, gameSeed = seedForPair(i);
      const games = [];
      for (let side=0;side<2;side++) {
        const seat: Player = (i+side)%2===0 ? 'X':'O';
        let state: GameState = position(opening);
        const moves=[...opening], ourTurns=[], originalTurns=[];
        await project.beginGame();
        const reset = await official.send({type:'reset',opening,seed:(gameSeed^0x6a09e667)>>>0});
        if (reset.result!==suite[i].state) throw new Error('Original did not replay the published starting state');
        while (!state.winner) {
          let move: number;
          if (state.turn===seat) {
            const answer=await project.move(state); move=answer.move; ourTurns.push(answer);
          } else {
            const response=await official.send({type:'move',moves,state:projectStateDigits(state)});
            const answer=response.result; move=answer.move;
            if (answer.visits<2 || answer.visits>1000 || answer.addedVisits!==answer.visits-answer.startVisits ||
                (answer.visits<1000 && answer.searchMs<220)) throw new Error('Unverified original time or visit budget');
            originalTurns.push({...answer,elapsedMs:response.elapsedMs});
          }
          if (!legalMoves(state).includes(move)) throw new Error('Illegal timed move');
          moves.push(move); state=play(state,Math.floor(move/9),move%9);
        }
        const end=await official.send({type:'finish',moves,state:projectStateDigits(state)});
        if (end.result!==({X:1,O:2,draw:3} as const)[state.winner]) throw new Error('Terminal outcome differs between engines');
        games.push({seat,winner:state.winner,score:state.winner==='draw'?.5:Number(state.winner===seat),moves,ourTurns,originalTurns});
      }
      results.push({pair:i,seed:gameSeed,opening,games});
      const temp=progress+'.'+randomUUID(); await writeFile(temp,JSON.stringify({spec,evidence,results})); await rename(temp,progress);
      console.error('Completed',i+1,'of',pairs,'published pairs');
    }
    const all=results.flatMap(row=>row.games);
    const wins=all.filter(g=>g.score===1).length,draws=all.filter(g=>g.score===.5).length,losses=all.length-wins-draws;
    const summary={wins,draws,losses,score:(wins+draws/2)/all.length};
    const temp=out+'.'+randomUUID(); await writeFile(temp,JSON.stringify({status:'complete',spec,evidence,summary,results}));
    try { await link(temp,out); } finally { await unlink(temp); }
    await unlink(progress); await official.close();
    console.log(JSON.stringify(summary));
  } finally { await project.close(); }
}
tournament().catch(error=>{console.error(error);process.exitCode=1;});
