import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

export default tseslint.config(
  // `.vite` is Vite's dependency-optimizer cache, written by the first `npm run dev`. It holds
  // pre-bundled React, which lints as several hundred errors about browser globals. A fresh CI
  // checkout does not have it, so leaving it out of this list made the gate pass in CI and fail
  // on every machine that had ever started the dev server.
  { ignores: ["dist", "node_modules", ".vite", "coverage", "test-results", "playwright-report"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    },
  },
  {
    // Build-time scripts run under node, not in the browser: console and process are theirs.
    files: ["scripts/**/*.mjs"],
    languageOptions: {
      globals: { console: "readonly", process: "readonly" },
    },
  },
);
