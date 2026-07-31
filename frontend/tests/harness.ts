import { test as base, expect, type Page } from "@playwright/test";

import {
  ARCHITECTURES,
  COMPARE,
  EXAMPLES,
  MODELS,
  REPORTED_RESULTS,
  SCAN_FRAMES,
  SCAN_STARTED,
  TRAJECTORIES,
} from "./fixtures";

/**
 * The stubbed world every browser test runs in.
 *
 * Nothing is allowed to reach the network. The catch-all route below aborts anything the specific
 * stubs did not claim and records it, and the fixture asserts that list is empty at teardown - so a
 * new fetch added to the app shows up as a failing test rather than as a request that quietly went
 * to whatever was listening on port 8000.
 *
 * The cost is that these tests cannot catch contract drift between the two sides. That is what
 * backend/tests/test_api.py is for.
 */
export interface Harness {
  /** console.error text and uncaught exceptions collected for the whole test. */
  consoleErrors: string[];
  /** Requests that reached the catch-all, i.e. that nothing above stubbed. Must stay empty. */
  unstubbed: string[];
  /** API paths the page asked for, so a test can assert something was *not* fetched. */
  apiCalls: string[];
}

const API_STUBS: Record<string, unknown> = {
  "/api/architectures": ARCHITECTURES,
  "/api/trajectories": TRAJECTORIES,
  "/api/reported-results": REPORTED_RESULTS,
  // The serving endpoints. A test that wants a different registry state registers its own route
  // afterwards: Playwright checks handlers in reverse registration order, so the later one wins.
  "/api/models": MODELS,
  "/api/examples": EXAMPLES,
  "/api/compare": COMPARE,
};

async function guardTheNetwork(page: Page, harness: Harness, origin: string): Promise<void> {
  // Registered first so the specific stubs below take precedence over it.
  await page.route("**/*", async (route) => {
    const url = route.request().url();
    // Same-origin document, bundle and styles are the app itself.
    if (url.startsWith(origin) && !url.includes("/api/")) return route.continue();
    harness.unstubbed.push(url);
    await route.abort();
  });

  // The job endpoints need shapes the flat map cannot express: a POST that returns a started job,
  // and a stream. The stream is fulfilled as one body holding every frame - the app closes it on
  // the terminal snapshot, so a stub that ends immediately is the same story told faster.
  await page.route("**/api/jobs", async (route) => {
    harness.apiCalls.push("/api/jobs");
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify(SCAN_STARTED),
    });
  });
  await page.route("**/api/jobs/*/events", async (route) => {
    harness.apiCalls.push("/api/jobs/events");
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: SCAN_FRAMES.map((frame) => `event: update\ndata: ${JSON.stringify(frame)}\n\n`).join(
        "",
      ),
    });
  });

  for (const [path, body] of Object.entries(API_STUBS)) {
    await page.route(`**${path}`, async (route) => {
      harness.apiCalls.push(path);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    });
  }
}

export const test = base.extend<{ harness: Harness }>({
  harness: [
    async ({ page, baseURL }, use) => {
      const harness: Harness = { consoleErrors: [], unstubbed: [], apiCalls: [] };

      // Errors only. Warnings are noise; an uncaught exception is not.
      page.on("console", (message) => {
        if (message.type() === "error") harness.consoleErrors.push(message.text());
      });
      page.on("pageerror", (error) => harness.consoleErrors.push(String(error)));

      await guardTheNetwork(page, harness, baseURL ?? "http://127.0.0.1:4174");

      await use(harness);

      expect(harness.unstubbed, "requests nothing stubbed").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
