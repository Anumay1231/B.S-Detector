import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, the Python server (python run.py --no-browser) answers /api on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
