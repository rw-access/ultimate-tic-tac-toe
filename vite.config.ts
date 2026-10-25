import { defineConfig } from 'vite';

export default defineConfig({
  worker: { format: 'es' },
  server: { watch: { ignored: ['**/.venv/**'] } },
});
