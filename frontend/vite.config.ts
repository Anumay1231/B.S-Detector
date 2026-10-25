import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// In dev, the Python server (python run.py --no-browser) answers /api on :8000.
// Point elsewhere with BSD_API in frontend/.env.local.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { '/api': env.BSD_API || 'http://127.0.0.1:8000' },
    },
  }
})
