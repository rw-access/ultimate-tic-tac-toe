import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('TypeScript self-play CLI emits Python-compatible JSONL and reproducibility metadata', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-training-'));
  const output = join(directory, 'tiny.jsonl');
  try {
    const child = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '1', '--simulations', '1', '--seed', '42', '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(child.error);
    assert.equal(child.status, 0, child.stderr);
    const firstDataset = await readFile(output, 'utf8');

    const [line] = (await readFile(output, 'utf8')).trim().split('\n');
    const record = JSON.parse(line) as {
      game: number;
      winner: number;
      moves: number;
      samples: [number[], number[], number][];
    };
    assert.deepEqual(Object.keys(record), ['game', 'winner', 'moves', 'samples']);
    assert.equal(record.game, 0);
    assert.ok([1, -1, 2].includes(record.winner));
    assert.equal(record.moves, record.samples.length);
    assert.ok(record.moves > 0 && record.moves <= 81);

    for (const [features, policy, value] of record.samples) {
      assert.equal(features.length, 280);
      assert.equal(policy.length, 81);
      assert.ok(features.every(Number.isFinite));
      assert.ok(policy.every(Number.isFinite));
      assert.ok(value === -1 || value === 0 || value === 1);
      assert.ok(Math.abs(policy.reduce((sum, mass) => sum + mass, 0) - 1) < 1e-5);
      for (let action = 0; action < 81; action++) {
        if (features[162 + action] === 0) assert.equal(policy[action], 0);
      }
    }

    const metadataPath = join(directory, 'tiny.metadata.json');
    const metadata = JSON.parse(await readFile(metadataPath, 'utf8'));
    assert.equal(metadata.format, 'ultimate-tic-tac-toe-selfplay-v1');
    assert.equal(metadata.games, 1);
    assert.equal(metadata.simulationsPerPosition, 1);
    assert.equal(metadata.seed, 42);
    assert.equal(metadata.rulesSource, 'src/game.ts');
    assert.deepEqual(metadata.rootNoise, {
      epsilon: 0,
      concentrationMass: 2.7,
      alphaFormula: 'concentrationMass / legalRootMoves',
    });

    const repeated = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '1', '--simulations', '1', '--seed', '42', '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(repeated.error);
    assert.equal(repeated.status, 0, repeated.stderr);
    assert.equal(await readFile(output, 'utf8'), firstDataset);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('TypeScript self-play CLI rejects misspelled options before writing a dataset', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-training-'));
  const output = join(directory, 'should-not-exist.jsonl');
  try {
    const child = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '1', '--simulations', '1', '--simulatons', '7', '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
    assert.ifError(child.error);
    assert.equal(child.status, 1);
    assert.match(child.stderr, /Unknown option for generate: --simulatons/);
    await assert.rejects(readFile(output));

    const mixWithoutModel = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '1', '--simulations', '1', '--teacher-mix', '0.25', '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
    assert.ifError(mixWithoutModel.error);
    assert.equal(mixWithoutModel.status, 1);
    assert.match(mixWithoutModel.stderr, /--teacher-mix requires --model/);
    await assert.rejects(readFile(output));

    for (const [flag, value, message] of [
      ['--root-noise', '1.01', /--root-noise must be a number between 0 and 1/],
      ['--root-noise', 'NaN', /--root-noise must be a number between 0 and 1/],
      ['--root-noise-mass', '0', /--root-noise-mass must be a finite number greater than 0/],
      ['--root-noise-mass', 'Infinity', /--root-noise-mass must be a finite number greater than 0/],
    ] as const) {
      const invalidNoise = spawnSync(process.execPath, [
        '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
        '--games', '1', '--simulations', '1', flag, value, '--output', output,
      ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
      assert.ifError(invalidNoise.error);
      assert.equal(invalidNoise.status, 1);
      assert.match(invalidNoise.stderr, message);
      await assert.rejects(readFile(output));
    }

    const invalidCaps = [
      { args: ['--fast-simulations', '1'], message: /must be provided together/ },
      { args: ['--full-search-probability', '0.5'], message: /must be provided together/ },
      { args: ['--fast-simulations', '5', '--full-search-probability', '0.5'], message: /must be <= --simulations/ },
      { args: ['--fast-simulations', '0', '--full-search-probability', '0.5'], message: /--fast-simulations must be an integer >= 1/ },
      { args: ['--fast-simulations', '1', '--full-search-probability', '0'], message: /must be a finite number in \(0, 1\]/ },
      { args: ['--fast-simulations', '1', '--full-search-probability', 'NaN'], message: /must be a finite number in \(0, 1\]/ },
      { args: ['--fast-simulations', '1', '--full-search-probability', '1.01'], message: /must be a finite number in \(0, 1\]/ },
    ] as const;
    for (const invalid of invalidCaps) {
      const invalidCap = spawnSync(process.execPath, [
        '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
        '--games', '1', '--simulations', '4', ...invalid.args, '--output', output,
      ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
      assert.ifError(invalidCap.error);
      assert.equal(invalidCap.status, 1);
      assert.match(invalidCap.stderr, invalid.message);
      await assert.rejects(readFile(output));
    }

    const arenaNoise = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
      '--root-noise', '0.25',
    ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
    assert.ifError(arenaNoise.error);
    assert.equal(arenaNoise.status, 1);
    assert.match(arenaNoise.stderr, /Unknown option for arena: --root-noise/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('randomized search caps use an isolated schedule stream and save only full-cap targets', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-random-caps-'));
  const defaults = join(directory, 'full-default.jsonl');
  const allFull = join(directory, 'full-schedule.jsonl');
  const sampled = join(directory, 'sampled.jsonl');
  const repeated = join(directory, 'sampled-repeated.jsonl');
  const zeroSamples = join(directory, 'zero-samples.jsonl');
  try {
    const defaultArgs = [
      '--games', '2', '--simulations', '2', '--seed', '9162',
    ];
    const defaultRun = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      ...defaultArgs, '--output', defaults,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(defaultRun.error);
    assert.equal(defaultRun.status, 0, defaultRun.stderr);

    const allFullRun = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      ...defaultArgs, '--fast-simulations', '1', '--full-search-probability', '1', '--output', allFull,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(allFullRun.error);
    assert.equal(allFullRun.status, 0, allFullRun.stderr);
    assert.equal(await readFile(allFull, 'utf8'), await readFile(defaults, 'utf8'));

    const sampledArgs = [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '4', '--simulations', '4', '--fast-simulations', '1',
      '--full-search-probability', '0.5', '--parallel-games', '2', '--seed', '2026',
    ];
    const first = spawnSync(process.execPath, [...sampledArgs, '--output', sampled], {
      cwd: ROOT, encoding: 'utf8', timeout: 30_000,
    });
    assert.ifError(first.error);
    assert.equal(first.status, 0, first.stderr);
    const repeatedRun = spawnSync(process.execPath, [...sampledArgs, '--output', repeated], {
      cwd: ROOT, encoding: 'utf8', timeout: 30_000,
    });
    assert.ifError(repeatedRun.error);
    assert.equal(repeatedRun.status, 0, repeatedRun.stderr);
    assert.equal(await readFile(repeated, 'utf8'), await readFile(sampled, 'utf8'));

    const records = (await readFile(sampled, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
    const metadata = JSON.parse(await readFile(join(directory, 'sampled.metadata.json'), 'utf8'));
    const capStats = metadata.randomizedSearchCaps;
    assert.equal(capStats.fullSimulations, 4);
    assert.equal(capStats.fastSimulations, 1);
    assert.equal(capStats.fullSearchProbability, 0.5);
    assert.match(capStats.scheduleSeedFormula, /generationSeed XOR 0x51ed270b/);
    assert.ok(capStats.requestedCapMoves.full > 0);
    assert.ok(capStats.requestedCapMoves.fast > 0);
    assert.deepEqual(capStats.observedSearchCalls, capStats.requestedCapMoves);
    assert.equal(capStats.savedPositions, records.reduce((sum, record) => sum + record.samples.length, 0));
    assert.equal(capStats.savedPositions, capStats.requestedCapMoves.full);
    assert.ok(capStats.gamesWithZeroSamples >= 0 && capStats.gamesWithZeroSamples <= records.length);
    assert.equal(capStats.actualSimulations.total,
      capStats.actualSimulations.fullCapTotal + capStats.actualSimulations.fastCapTotal);
    assert.equal(capStats.requestedSimulations.total,
      capStats.requestedSimulations.fullCapTotal + capStats.requestedSimulations.fastCapTotal);
    assert.ok(capStats.actualSimulations.fullCapTotal <= capStats.requestedSimulations.fullCapTotal);
    assert.ok(capStats.actualSimulations.fastCapTotal <= capStats.requestedSimulations.fastCapTotal);
    assert.ok(capStats.actualSimulations.fullCapTotal <= capStats.actualSimulations.fullCapCalls * 4);
    assert.ok(capStats.actualSimulations.fastCapTotal <= capStats.actualSimulations.fastCapCalls);

    let observedFastMove = false;
    for (const record of records) {
      assert.ok(record.moves >= record.samples.length);
      if (record.moves > record.samples.length) observedFastMove = true;
      for (const [features, policy, value] of record.samples as [number[], number[], number][]) {
        assert.equal(features.length, 280);
        assert.equal(policy.length, 81);
        assert.ok(Math.abs(policy.reduce((sum, mass) => sum + mass, 0) - 1) < 1e-5);
        for (let action = 0; action < 81; action++) {
          if (features[162 + action] === 0) assert.equal(policy[action], 0);
        }
        const totalMarks = features.slice(0, 162).reduce((sum, mark) => sum + mark, 0);
        const currentPlayer = totalMarks % 2 === 0 ? 'X' : 'O';
        const expectedValue = record.winner === 2 ? 0
          : record.winner === 1 ? (currentPlayer === 'X' ? 1 : -1)
            : (currentPlayer === 'O' ? 1 : -1);
        assert.equal(value, expectedValue);
      }
    }
    assert.ok(observedFastMove);

    const zeroSampleRun = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
      '--games', '1', '--simulations', '4', '--fast-simulations', '1',
      '--full-search-probability', '1e-12', '--seed', '77', '--output', zeroSamples,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(zeroSampleRun.error);
    assert.equal(zeroSampleRun.status, 0, zeroSampleRun.stderr);
    const [zeroRecord] = (await readFile(zeroSamples, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
    const zeroMetadata = JSON.parse(await readFile(join(directory, 'zero-samples.metadata.json'), 'utf8'));
    assert.ok(zeroRecord.moves > 0);
    assert.deepEqual(zeroRecord.samples, []);
    assert.equal(zeroMetadata.randomizedSearchCaps.savedPositions, 0);
    assert.equal(zeroMetadata.randomizedSearchCaps.gamesWithZeroSamples, 1);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('model-guided self-play batches concurrent games and remains deterministic and legal', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-model-training-'));
  const output = join(directory, 'batched.jsonl');
  const model = resolve(ROOT, 'public/models/policy-value.onnx');
  const command = [
    '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'generate',
    '--games', '2', '--simulations', '2', '--seed', '420',
    '--parallel-games', '2', '--model', model, '--teacher-mix', '0.25',
    '--root-noise', '0.25', '--root-noise-mass', '2.7', '--output', output,
  ];
  try {
    const first = spawnSync(process.execPath, command, {
      cwd: ROOT, encoding: 'utf8', timeout: 30_000,
    });
    assert.ifError(first.error);
    assert.equal(first.status, 0, first.stderr);
    const firstDataset = await readFile(output, 'utf8');
    const records = firstDataset.trim().split('\n').map(line => JSON.parse(line));
    assert.deepEqual(records.map(record => record.game), [0, 1]);
    for (const record of records) {
      assert.equal(record.moves, record.samples.length);
      for (const [features, policy, value] of record.samples as [number[], number[], number][]) {
        assert.equal(features.length, 280);
        assert.equal(policy.length, 81);
        assert.ok(value === -1 || value === 0 || value === 1);
        assert.ok(Math.abs(policy.reduce((sum, mass) => sum + mass, 0) - 1) < 1e-5);
        for (let action = 0; action < 81; action++) {
          if (features[162 + action] === 0) assert.equal(policy[action], 0);
        }
      }
    }
    const metadata = JSON.parse(await readFile(join(directory, 'batched.metadata.json'), 'utf8'));
    assert.equal(metadata.parallelGames, 2);
    assert.equal(metadata.teacherMix, 0.25);
    assert.equal(metadata.teacher, 'blended-onnx-typescript-mcts');
    assert.deepEqual(metadata.rootNoise, {
      epsilon: 0.25,
      concentrationMass: 2.7,
      alphaFormula: 'concentrationMass / legalRootMoves',
    });
    assert.match(metadata.modelSha256, /^[a-f0-9]{64}$/);
    assert.ok(metadata.inference.positions > 0);
    assert.ok(metadata.inference.batches > 0);
    assert.ok(metadata.inference.positions > metadata.inference.batches);
    assert.ok(metadata.inference.meanBatchSize > 1);
    assert.ok(metadata.inference.positionsPerSecond > 0);

    const repeated = spawnSync(process.execPath, command, {
      cwd: ROOT, encoding: 'utf8', timeout: 30_000,
    });
    assert.ifError(repeated.error);
    assert.equal(repeated.status, 0, repeated.stderr);
    assert.equal(await readFile(output, 'utf8'), firstDataset);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('arena ranges preserve the full seeded opening stream and absolute pair identities', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-arena-ranges-'));
  const model = resolve(ROOT, 'public/models/policy-value.onnx');
  const run = async (name: string, range: string[]) => {
    const output = join(directory, `${name}.json`);
    const child = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
      '--challenger', model, '--champion', model, '--pairs', '4',
      '--simulations', '4', '--seed', '4281', '--parallel-games', '2',
      ...range, '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(child.error);
    assert.equal(child.status, 0, child.stderr);
    return JSON.parse(await readFile(output, 'utf8'));
  };
  try {
    const complete = await run('complete', []);
    const first = await run('first', ['--pair-offset', '0', '--pair-count', '2']);
    const second = await run('second', ['--pair-offset', '2', '--pair-count', '2']);
    assert.equal(complete.arenaShard, undefined);
    assert.deepEqual(first.arenaShard, { plannedPairs: 4, pairOffset: 0, pairCount: 2 });
    assert.deepEqual(second.arenaShard, { plannedPairs: 4, pairOffset: 2, pairCount: 2 });
    assert.equal(first.games, 4);
    assert.equal(second.games, 4);
    assert.deepEqual([...first.perPair, ...second.perPair], complete.perPair);
    assert.deepEqual([...first.pairScores, ...second.pairScores], complete.pairScores);
    for (const range of [
      ['--pair-offset', '4'], ['--pair-offset', '3', '--pair-count', '2'],
      ['--pair-offset', '-1'], ['--pair-count', '0'],
    ]) {
      const output = join(directory, 'invalid.json');
      const child = spawnSync(process.execPath, [
        '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
        '--challenger', model, '--champion', model, '--pairs', '4',
        ...range, '--output', output,
      ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
      assert.equal(child.status, 1);
      await assert.rejects(readFile(output));
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('paired arena writes swapped-seat results, model hashes, and per-pair scores', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ultimate-tic-tac-toe-arena-'));
  const output = join(directory, 'arena.json');
  const model = resolve(ROOT, 'public/models/policy-value.onnx');
  try {
    const child = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
      '--challenger', model, '--champion', model,
      '--pairs', '1', '--simulations', '1', '--seed', '420',
      '--parallel-games', '2', '--output', output,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(child.error);
    assert.equal(child.status, 0, child.stderr);
    const result = JSON.parse(await readFile(output, 'utf8'));
    assert.equal(result.format, 'ultimate-tic-tac-toe-paired-arena-v1');
    assert.equal(result.pairs, 1);
    assert.equal(result.games, 2);
    assert.equal(result.parallelGames, 2);
    assert.equal(result.challengerValueMode, 'model');
    assert.equal(result.championValueMode, 'model');
    assert.equal(result.tacticalProof, false);
    assert.equal(result.openingPlyCount, 2);
    assert.match(result.challengerModel.sha256, /^[a-f0-9]{64}$/);
    assert.equal(result.challengerModel.sha256, result.championModel.sha256);
    assert.deepEqual(result.challenger, {
      wins: result.challenger.wins,
      draws: result.challenger.draws,
      losses: result.challenger.losses,
      games: 2,
      scoreRate: result.challenger.scoreRate,
    });
    assert.equal(result.challenger.wins + result.challenger.draws + result.challenger.losses, 2);
    assert.equal(result.pairScores.length, 1);
    assert.equal(result.pairScores[0], result.perPair[0].pairScore);
    assert.ok(result.pairedConfidence95.lower >= 0 && result.pairedConfidence95.lower <= 1);
    assert.ok(result.pairedConfidence95.upper >= 0 && result.pairedConfidence95.upper <= 1);
    assert.ok(result.pairedConfidence95.lower <= result.pairedConfidence95.upper);
    assert.equal(result.perPair.length, 1);
    const pair = result.perPair[0];
    assert.equal(pair.openingMoves.length, 2);
    assert.ok(pair.openingMoves.every((move: number) => Number.isInteger(move) && move >= 0 && move < 81));
    assert.ok(['win', 'draw', 'loss'].includes(pair.challengerAsX.result));
    assert.ok(['win', 'draw', 'loss'].includes(pair.challengerAsO.result));
    assert.equal(pair.pairScore, pair.points / 2);
    assert.equal(result.results.pairs.wins + result.results.pairs.draws + result.results.pairs.losses, 1);
    assert.ok(result.inference.challenger.positions > 0);
    assert.ok(result.inference.champion.positions > 0);

    const ablationOutput = join(directory, 'arena-ablation.json');
    const ablation = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
      '--challenger', model, '--champion', model,
      '--pairs', '1', '--simulations', '1', '--seed', '420',
      '--parallel-games', '2', '--challenger-value', 'zero', '--champion-value', 'heuristic',
      '--tactical-proof', '--output', ablationOutput,
    ], { cwd: ROOT, encoding: 'utf8', timeout: 30_000 });
    assert.ifError(ablation.error);
    assert.equal(ablation.status, 0, ablation.stderr);
    const ablationResult = JSON.parse(await readFile(ablationOutput, 'utf8'));
    assert.equal(ablationResult.challengerValueMode, 'zero');
    assert.equal(ablationResult.championValueMode, 'heuristic');
    assert.equal(ablationResult.tacticalProof, true);

    const invalidMode = spawnSync(process.execPath, [
      '--experimental-strip-types', resolve(ROOT, 'scripts/ai.ts'), 'arena',
      '--challenger-value', 'policy',
    ], { cwd: ROOT, encoding: 'utf8', timeout: 10_000 });
    assert.ifError(invalidMode.error);
    assert.equal(invalidMode.status, 1);
    assert.match(invalidMode.stderr, /--challenger-value must be one of: model, zero, heuristic/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
