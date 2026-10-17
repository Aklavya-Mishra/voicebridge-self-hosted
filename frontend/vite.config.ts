import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.VOICEBRIDGE_BACKEND ?? 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    // host: true lets a second machine on the LAN open the dev server.
    host: true,
    // 5273/8001 so this edition runs next to the main voicebridge repo (5173/8000).
    port: 5273,
    strictPort: true,
    proxy: {
      '/api': backend,
      '/socket.io': { target: backend, ws: true },
    },
  },
})
