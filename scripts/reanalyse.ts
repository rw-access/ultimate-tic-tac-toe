/** Persistent local PUCT worker for training/reanalyse_positions.py. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createModelClient } from './ai.ts';
import { encodeState, legalMoves, search, type Evaluator } from '../src/ai.ts';
import { play, restoreGame, type Player } from '../src/game.ts';

export function decode(features: Uint8Array) {
  if (features.length !== 280 || features.some(value => value > 1)) throw new Error('Expected 280 binary features');
  let own = 0, other = 0;
  for (let i = 0; i < 81; i++) {
    if (features[i] && features[81 + i]) throw new Error('Overlapping marks');
    own += features[i]; other += features[81 + i];
  }
  if (other !== own && other !== own + 1) throw new Error('Invalid side-to-move mark counts');
  const turn: Player = own === other ? 'X' : 'O';
  const opponent: Player = turn === 'X' ? 'O' : 'X';
  const boards = Array.from({ length: 9 }, (_, board) => Array.from({ length: 9 }, (_, cell) => {
    const i = board * 9 + cell;
    if (features[i]) return turn;
    return features[81 + i] ? opponent : null;
  }));
  const forced = features.subarray(270).indexOf(1);
  const state = restoreGame({ boards, turn, forced: forced === 9 ? null : forced });
  if (!state || state.winner || !encodeState(state).every((value, i) => value === features[i])) {
    throw new Error('Position is terminal or disagrees with canonical rules/feature encoding');
  }
  return state;
}

async function analyse(features: Uint8Array, row: number, seed: number, simulations: number, evaluate: Evaluator) {
  try {
    const state = decode(features);
    // A stable seed for this source row, irrespective of sampling order or concurrency.
    const rowSeed = createHash('sha256').update(`${seed}:${row}`).digest().readUInt32LE(0);
    const result = await search(state, { simulations, timeMs: null, tacticalProof: true, seed: rowSeed, evaluate });
    if (result.move === null || !legalMoves(state).includes(result.move)) throw new Error('No legal search result');
    if (!result.root) {
      if (result.simulations !== 0 || play(state, Math.floor(result.move / 9), result.move % 9).winner !== state.turn) {
        throw new Error('Zero-visit result was not a verified immediate win');
      }
      const p = Array<number>(81).fill(0); p[result.move] = 1;
      return { p, v: 1, visits: 0, edgeVisits: 0, move: result.move };
    }
    const root = result.root;
    if (root.visits !== simulations || result.simulations !== simulations) {
      throw new Error(`Search reached ${root.visits}/${simulations} root visits before the time limit`);
    }
    const p = Array<number>(81).fill(0);
    let edgeVisits = 0;
    for (const child of root.children ?? []) {
      const visits = child.node?.visits ?? 0;
      p[child.move] = visits; edgeVisits += visits;
    }
    if (!edgeVisits || edgeVisits >= root.visits) throw new Error('Invalid root child visits');
    const v = root.valueSum / root.visits;
    if (!Number.isFinite(v) || Math.abs(v) > 1) throw new Error('Invalid root mean value');
    return { p: p.map(n => n / edgeVisits), v, visits: root.visits, edgeVisits, move: result.move };
  } catch (error) { throw new Error(`Source row ${row}: ${error instanceof Error ? error.message : error}`); }
}

async function main() {
  const { values } = parseArgs({ options: { model: { type: 'string' }, simulations: { type: 'string' },
    parallel: { type: 'string' }, seed: { type: 'string' } } });
  const simulations = Number(values.simulations), parallel = Number(values.parallel), seed = Number(values.seed);
  if (!values.model || !Number.isInteger(simulations) || simulations < 2 || simulations > 10000 ||
      !Number.isInteger(parallel) || parallel < 1 || parallel > 256 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Required: --model PATH --simulations 2..10000 --parallel 1..256 --seed UINT32');
  }
  const modelSha256 = createHash('sha256').update(await readFile(values.model)).digest('hex');
  const controller = new AbortController();
  const client = await createModelClient(values.model, controller.signal, parallel);
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  try {
    console.log(JSON.stringify({ type: 'ready', modelSha256, simulations, seed, parallel }));
    for await (const line of lines) {
      const request = JSON.parse(line) as { ids: number[]; x: string };
      const features = Buffer.from(request.x, 'base64');
      if (!Array.isArray(request.ids) || !request.ids.length || features.length !== 280 * request.ids.length ||
          !request.ids.every(id => Number.isSafeInteger(id) && id >= 0)) throw new Error('Invalid work request');
      const targets = [];
      for (let start = 0; start < request.ids.length; start += parallel) {
        const rows = request.ids.slice(start, start + parallel);
        targets.push(...await Promise.all(rows.map((id, j) => analyse(features.subarray((start+j)*280, (start+j+1)*280),
          id, seed, simulations, client.evaluate))));
      }
      console.log(JSON.stringify({ type: 'targets', targets, inference: client.stats }));
    }
  } finally { controller.abort(); lines.close(); await client.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
