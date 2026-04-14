import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

export default defineConfig({
  server: {
    host: true,
    port: 5173,
    fs: {
      allow: [workspaceRoot],
    },
  },
});
