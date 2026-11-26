import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';

// The shipped worker exposes moves and analysis. Build its raw evaluator as
// an additional in-memory test entry so every policy logit can be compared.
export async function prepareParityProbe(root, modelBytes, fixturePath) {
  const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
  assert.equal(createHash('sha256').update(modelBytes).digest('hex'), fixture.modelSha256,
    'Parity fixture must describe the exact model in the build');
  assert.ok(fixture.cases.length >= 12);
  const project = fileURLToPath(new URL('../', import.meta.url));
  const { output } = await build({
    root: project, configFile: false, publicDir: false, base: './', logLevel: 'silent',
    build: {
      write: false,
      rollupOptions: { input: resolve(project, 'src/onnx.ts'), preserveEntrySignatures: 'strict' },
    },
  });
  const entries = output.filter(item => item.type === 'chunk');
  assert.equal(entries.length, 1);
  assert.deepEqual(entries[0].exports, ['createEvaluator']);
  for (const asset of output.filter(item => item.type === 'asset')) {
    assert.ok(Buffer.from(asset.source).equals(await readFile(resolve(root, asset.fileName))),
      `Browser inference asset differs from the production build: ${asset.fileName}`);
  }
  return { entry: entries[0], fixture };
}

export async function checkWorkerParity(page, site, probe) {
  const worker = page.workers().find(item => /\/ai\.worker-[^/]+\.js$/.test(item.url()));
  assert.ok(worker, 'A production game worker must be active');
  const result = await worker.evaluate(async ({ moduleUrl, modelUrl, cases }) => {
    const { createEvaluator } = await import(moduleUrl);
    const evaluate = await createEvaluator(modelUrl);
    let maxPolicy = 0;
    let maxValue = 0;
    try {
      for (const item of cases) {
        const output = await evaluate(item.state);
        if (item.policy.length !== 81) throw new Error(`Invalid policy fixture for ${item.name}`);
        for (let move = 0; move < 81; move++) {
          const difference = Math.abs(output.policy[move] - item.policy[move]);
          if (!(difference < 1e-4)) throw new Error(`${item.name} policy[${move}] error ${difference}`);
          maxPolicy = Math.max(maxPolicy, difference);
        }
        const difference = Math.abs(output.value - item.value);
        if (!(difference < 1e-4)) throw new Error(`${item.name} value error ${difference}`);
        maxValue = Math.max(maxValue, difference);
      }
    } finally {
      await evaluate.dispose();
    }
    return { maxPolicy, maxValue };
  }, {
    moduleUrl: new URL(probe.entry.fileName, site).href,
    modelUrl: new URL('models/policy-value.onnx', site).href,
    cases: probe.fixture.cases,
  });
  console.log(`Production worker parity: ${probe.fixture.cases.length} cases, max policy error ${result.maxPolicy}, max value error ${result.maxValue}, SHA-256 ${probe.fixture.modelSha256}`);
}
