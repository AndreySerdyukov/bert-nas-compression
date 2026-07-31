/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,mdx}"],
  // Light is the default and dark is the class, the same way the two playground siblings do it.
  // (billboard-planner inverts this; there the subject is a night-time map, here it is a text.)
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Semantic tokens on CSS variables; values flip under .dark (see index.css).
        canvas: "rgb(var(--canvas) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        ink: "rgb(var(--ink) / <alpha-value>)",
        slate: "rgb(var(--slate) / <alpha-value>)",
        hair: "rgb(var(--hair) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
      },
      fontFamily: {
        // No webfont anywhere: the Playwright harness aborts every unstubbed request, and a font
        // CDN is one more thing that can be slow, blocked or different in CI than on a laptop.
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        // Chapters are set in a serif; Georgia is on macOS and Windows and reads well at 18px.
        serif: ["ui-serif", "Georgia", "Cambria", '"Times New Roman"', "serif"],
        // Layer masks, parameter counts and latencies all want fixed advance width.
        mono: ['"SF Mono"', "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      borderRadius: {
        row: "6px",
        panel: "10px",
      },
      maxWidth: {
        // The measure for prose; wide figures break out of it deliberately.
        prose: "68ch",
        content: "1120px",
      },
    },
  },
  plugins: [],
};
