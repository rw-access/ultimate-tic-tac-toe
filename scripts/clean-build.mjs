import { rmSync } from 'node:fs';

// Clear generated output so retired URL prefixes are not uploaded again.
rmSync(new URL('../dist/', import.meta.url), { recursive: true, force: true });
