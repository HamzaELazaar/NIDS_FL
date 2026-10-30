import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    // Force esbuild to pre-bundle these CJS packages so Rolldown sees clean ESM
    include: ['react-plotly.js', 'plotly.js'],
  },
})