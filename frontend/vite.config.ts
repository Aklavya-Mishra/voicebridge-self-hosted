import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.VOICEBRIDGE_BACKEND ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    // host: true lets a second machine on the LAN open the dev server.
    host: true,
    proxy: {
      '/api': backend,
      '/socket.io': { target: backend, ws: true },
    },
  },
})
