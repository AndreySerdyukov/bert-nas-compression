import { test as base, expect, type Page } from "@playwright/test";

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
}

async function guardTheNetwork(page: Page, harness: Harness, origin: string): Promise<void> {
  await page.route("**/*", async (route) => {
    const url = route.request().url();
    // Same-origin document, bundle and styles are the app itself.
    if (url.startsWith(origin)) return route.continue();
    harness.unstubbed.push(url);
    await route.abort();
  });
}

export const test = base.extend<{ harness: Harness }>({
  harness: [
    async ({ page, baseURL }, use) => {
      const harness: Harness = { consoleErrors: [], unstubbed: [] };

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
