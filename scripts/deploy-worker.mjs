#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaultName = 'ultimate-tic-tac-toe';

function usage() {
  process.stdout.write('Usage: npm run deploy:worker -- [--name NAME] [--dry-run]\n');
  process.stdout.write('Environment: UTTT_WORKER_NAME, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID\n');
}

function parseArguments(args) {
  let name = process.env.UTTT_WORKER_NAME || defaultName;
  let dryRun = false;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--help' || argument === '-h') return { help: true };
    if (argument === '--dry-run') {
      dryRun = true;
      continue;
    }
    if (argument === '--name') {
      const value = args[index + 1];
      if (!value || value.startsWith('--')) throw new Error(`Expected a value after ${argument}.`);
      index += 1;
      name = value;
      continue;
    }
    throw new Error(`Unknown option: ${argument}`);
  }
  if (!/^[a-z0-9][a-z0-9-]{0,62}$/i.test(name)) throw new Error('Worker name must contain only letters, digits, and hyphens.');
  return { name, dryRun };
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit', env: process.env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

try {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    usage();
    process.exit(0);
  }
  process.stdout.write(`Building ${options.name} before ${options.dryRun ? 'dry-run validation' : 'deployment'}…\n`);
  run(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build']);

  const wrangler = resolve(root, 'node_modules', 'wrangler', 'bin', 'wrangler.js');
  const args = [wrangler, 'deploy', '--config', 'wrangler.jsonc', '--name', options.name];
  if (options.dryRun) args.push('--dry-run');
  run(process.execPath, args);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  usage();
  process.exit(1);
}
