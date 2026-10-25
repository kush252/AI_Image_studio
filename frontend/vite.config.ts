import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: '../backend/app/static',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        // Split the heavy, rarely-changing libraries out of the app chunk:
        // charts and KaTeX together are most of the bundle, and neither is
        // needed before the catalogue loads.
        // React itself is not split out: Rollup hoists it into whichever
        // chunk needs it first, so naming it here only produces an empty file.
        manualChunks: {
          charts: ['recharts'],
          katex: ['katex'],
        },
      },
    },
  },
})
