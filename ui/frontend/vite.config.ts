import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Standalone (non-monorepo) Vite app. In dev, /api is proxied to the FastAPI
// backend (default http://localhost:8000) so the EventSource("/api/...") calls
// work without CORS. In prod, FastAPI serves the built dist/ and same-origin /api.
const apiTarget = process.env.API_PROXY_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
    dedupe: ["react", "react-dom"],
  },
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: Number(process.env.PORT) || 5173,
    host: "0.0.0.0",
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
  preview: { port: Number(process.env.PORT) || 5173, host: "0.0.0.0" },
});
