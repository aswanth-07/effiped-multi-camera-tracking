import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  publicDir: "../../docs",
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: { allow: ["../.."] },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true
      }
    }
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rolldownOptions: {
      output: {
        // The replay fixtures are data, not code: ship them as their own chunk
        // so the app bundle stays small and they cache independently.
        codeSplitting: {
          groups: [{ name: "fixtures", test: /[\\/]src[\\/]data[\\/].+\.json$/ }]
        }
      }
    }
  }
});

