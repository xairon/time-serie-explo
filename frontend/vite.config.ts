import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-plotly': ['react-plotly.js', 'plotly.js-dist-min'],
          'vendor-maplibre': ['maplibre-gl'],
          'vendor-query': ['@tanstack/react-query'],
        },
      },
    },
    sourcemap: false,
  },
  server: {
    proxy: {
      // Override with e.g. VITE_PROXY_TARGET=http://localhost:49514 to hit a deployed API.
      '/api': process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000',
    },
  },
})
