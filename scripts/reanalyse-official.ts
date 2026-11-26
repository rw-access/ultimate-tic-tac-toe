/** Produce independent labelled roots from the published Stage 2 network and NMCTS. */
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { pairSeedGenerator } from './arena-openings.ts';
import { createPublishedSearch } from './official-uttt-arena.ts';
import { decode } from './reanalyse.ts';

async function main() {
  const { values } = parseArgs({ options: {
    model: { type: 'string' }, upstream: { type: 'string' }, provider: { type: 'string' },
    visits: { type: 'string' }, parallel: { type: 'string' }, seed: { type: 'string' },
  } });
  const visits = Number(values.visits), parallel = Number(values.parallel), seed = Number(values.seed);
  if (!values.model || !values.upstream || !['cuda', 'cpu'].includes(values.provider ?? '') ||
      !Number.isInteger(visits) || visits < 2 || visits > 10000 || !Number.isInteger(parallel) ||
      parallel < 1 || parallel > 128 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Pass published model, upstream, provider, fixed visits, parallel roots and a 32-bit seed');
  }
  const backend = await createPublishedSearch(values.upstream, values.model, values.provider as 'cuda' | 'cpu');
  const rowSeed = pairSeedGenerator('independent', seed, () => { throw new Error('Invalid schedule'); });
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  try {
    console.log(JSON.stringify({ type: 'ready', visits, parallel, seed, modelSha256: backend.modelSha256,
      upstreamCommit: backend.upstreamCommit }));
    for await (const line of lines) {
      const request = JSON.parse(line) as { ids: number[]; x: string };
      const features = Buffer.from(request.x, 'base64');
      if (!Array.isArray(request.ids) || !request.ids.length || features.length !== 280 * request.ids.length ||
          !request.ids.every(i => Number.isSafeInteger(i) && i >= 0)) throw new Error('Invalid root request');
      const targets = [];
      for (let first = 0; first < request.ids.length; first += parallel) {
        const chunk = request.ids.slice(first, first + parallel);
        targets.push(...await Promise.all(chunk.map((id, i) => backend.analyse(
          decode(features.subarray((first + i) * 280, (first + i + 1) * 280)), visits, rowSeed(id)))));
      }
      console.log(JSON.stringify({ type: 'targets', targets }));
    }
  } finally {
    lines.close();
    await backend.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
