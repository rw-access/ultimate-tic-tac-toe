// The benchmark serves the checked upstream module and the project's WASM runtime at these URLs.
import * as ort from './ort.mjs';
import loadPublished from './published-engine.mjs';

let session;
let game;
let engine;
let history;
let nextRandom;

function reset(opening, seed) {
  let state = seed >>> 0 || 1;
  nextRandom = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
  const mod = { exports: {} };
  loadPublished(mod, name => {
    if (name !== 'onnxruntime-web') throw new Error('Unexpected upstream dependency');
    return ort;
  }, nextRandom, navigator);
  const { Game, Engine } = mod.exports;
  game = new Game();
  history = [];
  function apply(move) {
    const action = game.getLegalActions().find(item => item.index === move);
    if (!action) throw new Error('Opening contains an illegal move');
    game.execute(action);
    history.push(move);
  }
  opening.forEach(apply);
  engine = new Engine(game.clone(), 1000, 2);
  engine.policyValueNetSession = {
    async run(feeds) {
      let output;
      try {
        output = await session.run(feeds);
        return {
          policy_logits: { data: Float32Array.from(output.policy_logits.data) },
          state_value: { data: [Number(output.state_value.data[0])] },
        };
      } finally {
        feeds.input.dispose();
        if (output) Object.values(output).forEach(t => t.dispose());
      }
    },
  };
  return game.state.join('');
}

function advance(moves) {
  if (moves.length < history.length || !history.every((move, i) => moves[i] === move)) {
    throw new Error('Mismatched history for the original engine');
  }
  for (const move of moves.slice(history.length)) {
    const action = game.getLegalActions().find(item => item.index === move);
    if (!action) throw new Error('The original engine rejected a recorded move');
    game.execute(action);
    engine.tree.synchronize(game);
    history.push(move);
  }
}

async function handle(message) {
  if (message.type === 'load') {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = message.runtimeFiles;
    const response = await fetch('./original.onnx');
    if (!response.ok) throw new Error('Original network unavailable');
    const bytes = await response.arrayBuffer();
    const address = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    const sha256 = Array.from(address, n => n.toString(16).padStart(2, '0')).join('');
    if (sha256 !== message.sha256) throw new Error('Original network digest changed');
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    if (session.inputNames.join() !== 'input' || !session.outputNames.includes('policy_logits') ||
        !session.outputNames.includes('state_value')) throw new Error('Unexpected original network interface');
    return { sha256, threads: ort.env.wasm.numThreads };
  }
  if (message.type === 'reset') return reset(message.opening, message.seed);
  advance(message.moves);
  if (game.state.join('') !== message.state) throw new Error('Game state differs from the project rules');
  if (message.type === 'finish') return game.result;
  const root = engine.tree.root;
  const startVisits = root.visitCount;
  const start = performance.now();
  if (message.type === 'fixture') {
    for (let i = 0; i < message.visits; i++) await engine.simulate();
    const n = Array(81).fill(0);
    const q = Array(81).fill(0);
    for (const child of root.childNodes) {
      n[child.action.index] = child.visitCount;
      q[child.action.index] = -child.stateValueMean;
    }
    return { n, q, v: root.stateValueMean };
  }
  if (message.type !== 'move' || !game.getLegalActions().length) throw new Error('Expected a live original-engine turn');
  const deadline = start + 220;
  while (root.visitCount < 1000 && (root.visitCount < 2 || performance.now() < deadline)) {
    await engine.simulate();
  }
  const top = Math.max(...root.childNodes.map(child => child.visitCount));
  const choices = root.childNodes.filter(child => child.visitCount === top);
  const move = choices[Math.floor(nextRandom() * choices.length)].action.index;
  return { move, startVisits, visits: root.visitCount, addedVisits: root.visitCount - startVisits,
    searchMs: performance.now() - start };
}

self.onmessage = async ({ data }) => {
  try { self.postMessage({ id: data.id, result: await handle(data.message) }); }
  catch (error) { self.postMessage({ id: data.id, error: String(error?.stack ?? error) }); }
};
