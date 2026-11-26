/** Play greedy policy and exhaustive one-ply games in a single-threaded Chromium worker. */
import { createHash, randomUUID } from 'node:crypto';
import { link, readFile, unlink, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { stripTypeScriptTypes } from 'node:module';
import { parseArgs } from 'node:util';
import { chromium } from '@playwright/test';

const { values } = parseArgs({
  options: Object.fromEntries(['student', 'teacher', 'original', 'openings', 'output'].map(name => [name, { type: 'string' }])),
});
if (!values.student || !values.teacher || !values.original || !values.openings || !values.output) {
  throw new Error('Supply --student, --teacher, --original, --openings and --output. Openings are a JSON array of board-major move sequences.');
}
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const openingBytes = await readFile(values.openings);
const openings = JSON.parse(openingBytes);
if (!Array.isArray(openings) || !openings.length || !openings.every(moves =>
  Array.isArray(moves) && moves.length < 81 && moves.every(a => Number.isInteger(a) && a >= 0 && a < 81))) {
  throw new Error('Expected at least one list of legal opening moves.');
}
const assets = new Map([['/', ['', 'text/html']], ['/worker.mjs', ['', 'text/javascript']]]);
for (const name of ['game', 'ai']) {
  const source = await readFile(new URL('../src/' + name + '.ts', import.meta.url), 'utf8');
  assets.set('/' + name + '.mjs', [stripTypeScriptTypes(source).replaceAll('./game.ts', './game.mjs'), 'text/javascript']);
}
for (const file of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  assets.set('/' + file, [await readFile(new URL('../node_modules/onnxruntime-web/dist/' + file, import.meta.url)),
    file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript']);
}
const models = {};
for (const name of ['student', 'teacher', 'original']) {
  const bytes = await readFile(values[name]);
  models[name] = hash(bytes);
  assets.set('/' + name + '.onnx', [bytes, 'application/octet-stream']);
}
const server = createServer((request, response) => {
  const asset = assets.get(request.url);
  if (!asset) { response.writeHead(404).end(); return; }
  response.writeHead(200, {
    'Content-Type': asset[1], 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
  }).end(asset[0]);
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:' + server.address().port);
  const started = page.waitForEvent('worker');
  await page.evaluate(() => { self.arena = new Worker('/worker.mjs', { type: 'module' }); });
  const worker = await started;
  const panels = await worker.evaluate(async ({ models, openings }) => {
    const ort = await import('/ort.wasm.min.mjs');
    const { createGame, play } = await import('/game.mjs');
    const { legalMoves, encodeState } = await import('/ai.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = { wasm: location.origin + '/ort-wasm-simd-threaded.wasm', mjs: location.origin + '/ort-wasm-simd-threaded.mjs' };
    const sessions = new Map();
    const boardToGrid = Array.from({ length: 81 }, (_, action) => {
      const board = Math.floor(action / 9), cell = action % 9;
      return (3 * Math.floor(board / 3) + Math.floor(cell / 3)) * 9 + 3 * (board % 3) + cell % 3;
    });
    async function predict(role, state, policy) {
      const encoded = encodeState(state), original = role === 'original';
      let input = encoded;
      if (original) {
        input = new Float32Array(324);
        for (let action = 0; action < 81; action++) {
          const i = boardToGrid[action];
          input[i] = encoded[action];
          input[81 + i] = encoded[81 + action];
          input[162 + i] = state.turn === 'X' ? 1 : -1;
          input[243 + i] = encoded[162 + action];
        }
      }
      const session = sessions.get(role);
      const tensor = new ort.Tensor('float32', input, original ? [1, 4, 9, 9] : [1, 280]);
      let out;
      try {
        out = await session.run({ [session.inputNames[0]]: tensor });
        const value = out[original ? 'state_value' : 'value'].data[0];
        if (!Number.isFinite(value)) throw new Error('Non-finite state value');
        if (!policy) return value;
        const logits = out[original ? 'policy_logits' : 'policy'].data;
        if (logits.length !== 81 || !logits.every(Number.isFinite)) throw new Error('Invalid policy scores');
        return original ? Float32Array.from(boardToGrid, i => logits[i]) : Float32Array.from(logits);
      } finally {
        tensor.dispose();
        if (out) Object.values(out).forEach(output => output.dispose());
      }
    }
    async function choose(role, state, mode) {
      const policy = mode === 'policy' ? await predict(role, state, true) : null;
      let best = -Infinity, chosen = -1;
      for (const move of legalMoves(state)) {
        let value;
        if (policy) value = policy[move];
        else {
          const child = play(state, Math.floor(move / 9), move % 9);
          value = child.winner === 'draw' ? 0 : child.winner ? 1 : -await predict(role, child, false);
        }
        if (value > best || (value === best && move < chosen)) { best = value; chosen = move; }
      }
      if (chosen < 0) throw new Error('No legal move for a live game');
      return chosen;
    }
    async function match(role, mode, seat, opening) {
      let state = createGame();
      const moves = [...opening];
      for (const action of opening) {
        if (!legalMoves(state).includes(action)) throw new Error('Illegal opening');
        state = play(state, Math.floor(action / 9), action % 9);
      }
      if (state.winner) throw new Error('Expected a live opening');
      while (!state.winner) {
        const move = await choose(state.turn === seat ? role : 'original', state, mode);
        moves.push(move);
        state = play(state, Math.floor(move / 9), move % 9);
      }
      return { winner: state.winner, moves };
    }
    try {
      for (const [role, expected] of Object.entries(models)) {
        const response = await fetch('/' + role + '.onnx');
        if (!response.ok) throw new Error('Model fetch failed');
        const bytes = await response.arrayBuffer();
        const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
        if (Array.from(digest, n => n.toString(16).padStart(2, '0')).join('') !== expected) throw new Error('Model digest mismatch');
        sessions.set(role, await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }));
      }
      const panels = {};
      for (const role of ['student', 'teacher']) {
        for (const mode of ['policy', 'one-ply']) {
          const games = [];
          for (const opening of openings) {
            games.push({ opening, firstAsX: await match(role, mode, 'X', opening), firstAsO: await match(role, mode, 'O', opening) });
          }
          panels[role + '-' + mode] = games;
        }
      }
      return panels;
    } finally {
      for (const session of sessions.values()) await session.release();
    }
  }, { models, openings });
  const result = { format: 'uttt-browser-policy-games-v1', models, openingSha256: hash(openingBytes),
    pairs: openings.length, runtime: browser.version(), panels };
  const temp = values.output + '.' + randomUUID();
  await writeFile(temp, JSON.stringify(result) + '\n', { flag: 'wx' });
  try { await link(temp, values.output); } finally { await unlink(temp); }
  console.log('Completed', 8 * openings.length, 'browser games across four comparisons.');
} finally {
  try { await browser?.close(); } finally { await new Promise(resolve => server.close(resolve)); }
}
