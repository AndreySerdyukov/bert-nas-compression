import { defineConfig, devices } from "@playwright/test";

/**
 * The browser tests run against the production bundle, not the dev server: the class of defect
 * they exist for is computed layout and bundling, and neither is the same under Vite's dev
 * transform. The network is stubbed inside the tests, so no backend is needed.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? "list" : "html",
  use: {
    baseURL: "http://127.0.0.1:4174",
    trace: "retain-on-failure",
    // Lets a machine without a Playwright download use its own Chrome:
    //   PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
    channel: process.env.PLAYWRIGHT_CHANNEL,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // --host 127.0.0.1 explicitly: vite preview otherwise binds ::1 and the tests ask for 127.0.0.1.
    command: "npm run build && npx vite preview --host 127.0.0.1 --port 4174 --strictPort",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
