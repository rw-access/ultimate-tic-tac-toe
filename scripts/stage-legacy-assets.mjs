import { cpSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Keep assets at the old prefix so existing room links keep their origin and
// browser-stored seat tokens while new games use /play/ultimate-tic-tac-toe/.
const source = fileURLToPath(new URL('../dist/play/ultimate-tic-tac-toe/', import.meta.url));
const destination = fileURLToPath(new URL('../dist/ultimate-tic-tac-toe/', import.meta.url));
rmSync(destination, { recursive: true, force: true });
cpSync(source, destination, { recursive: true });
