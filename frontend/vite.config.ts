import mdx from "@mdx-js/rollup";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [
    // MDX compiles at build time. `jsxImportSource: "react"` with no `providerImportSource` means
    // the emitted module imports nothing but `react/jsx-runtime` - no MDX package reaches the
    // bundle, so the three runtime dependencies stay three. scripts/check-runtime-deps.mjs proves
    // it rather than trusting it.
    //
    // enforce: "pre" is required: without it @vitejs/plugin-react sees the raw .mdx first.
    { enforce: "pre", ...mdx({ jsxImportSource: "react" }) },
    react({ include: /\.(jsx|js|mdx|md|tsx|ts)$/ }),
  ],
  server: {
    port: 5173,
    proxy: {
      // changeOrigin so the backend sees a consistent Host and CORS never enters the picture.
      "/api": { target: backend, changeOrigin: true },
      "/health": { target: backend, changeOrigin: true },
    },
  },
});
