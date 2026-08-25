import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    // Used in local development — proxy /api to the local FastAPI server
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  define: {
    // Expose VITE_API_URL for production builds (set this in Vercel env vars)
    // In dev, this is undefined and the proxy above handles routing.
    'import.meta.env.VITE_API_URL': JSON.stringify(
      mode === 'production' ? process.env.VITE_API_URL || '' : ''
    ),
  }
}))
