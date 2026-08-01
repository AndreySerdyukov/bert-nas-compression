import type { Page } from "@playwright/test";

import { ABLATION_FRAMES, ABLATION_RESULT, ABLATION_STARTED } from "./fixtures";
import { expect, test } from "./harness";

/**
 * The explorer, whose result is worthless without its framing.
 *
 * An amputated four-layer model scores in the sixties. Read alone that says "four layers cannot do
 * this task", which is the opposite of what this project measured - four *trained* layers reach
 * 90%. So the tests here are as much about the words as about the numbers: the warning has to be
 * on screen before anything is run, and the trained counterpart has to appear beside the ablation.
 */

/** Stub the job endpoints with an ablation run, optionally over a different mask. */
async function stubAblation(page: Page, layers?: number[]) {
  const frames =
    layers === undefined
      ? ABLATION_FRAMES
      : ABLATION_FRAMES.map((frame) =>
          frame.result === null
            ? frame
            : {
                ...frame,
                result: {
                  ...ABLATION_RESULT,
                  layers,
                  n_layers: layers.length,
                } as unknown as Record<string, unknown>,
              },
        );

  await page.route("**/api/jobs", async (route) => {
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify(ABLATION_STARTED),
    });
  });
  await page.route("**/api/jobs/*/events", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: frames.map((frame) => `event: update\ndata: ${JSON.stringify(frame)}\n\n`).join(""),
    });
  });
}

test("the warning is on screen before anything has been run", async ({ page }) => {
  await page.goto("/explorer");

  await expect(page.getByText("Read this before you run it")).toBeVisible();
  await expect(page.getByText(/nothing is trained afterwards/i)).toBeVisible();
  // And it is above the button, not under a result that does not exist yet.
  await expect(page.getByRole("button", { name: /Score \d+ layers/ })).toBeVisible();
});

test("a preset names the architecture it belongs to", async ({ page }) => {
  await page.goto("/explorer");

  await page.getByRole("button", { name: "AlphaNAS" }).click();
  await expect(page.getByText(/this is the mask of AlphaNAS/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Score 4 layers" })).toBeVisible();
});

test("an empty mask cannot be scored, and says why", async ({ page }) => {
  await page.goto("/explorer");

  // The default keeps four; turn them all off.
  for (const index of [0, 1, 2, 3]) {
    await page.getByRole("switch", { name: `Layer ${index}, kept` }).click();
  }
  await expect(page.getByRole("button", { name: /Score 0 layers/ })).toBeDisabled();
  await expect(page.getByText(/an encoder with none is not a model/)).toBeVisible();
});

test("the result carries three columns and an interval on each measured one", async ({ page }) => {
  await stubAblation(page);
  await page.goto("/explorer");
  await page.getByRole("button", { name: "Score 4 layers" }).click();

  // The amputation, collapsed.
  await expect(page.getByText("62.15%")).toBeVisible();
  // The same weights at full depth, scored on the same reviews in the same pass.
  await expect(page.getByText("93.25%")).toBeVisible();
  // The same mask trained properly - from controls.json, so the reader sees what training buys.
  await expect(page.getByText("90.62%")).toBeVisible();

  // Wilson bounds, not bare percentages.
  await expect(page.getByText("60.00% – 64.27%")).toBeVisible();
  await expect(page.getByText(/different set of reviews/)).toBeVisible();
});

test("a mask nobody trained says so instead of comparing against something else", async ({
  page,
}) => {
  // Layer 5 on top of the default four: a mask with no counterpart anywhere.
  await stubAblation(page, [0, 1, 2, 3, 5]);
  await page.goto("/explorer");
  await page.getByRole("switch", { name: "Layer 5, dropped" }).click();
  await page.getByRole("button", { name: "Score 5 layers" }).click();

  await expect(page.getByText(/Nobody has trained this mask/)).toBeVisible();
  await expect(page.getByText(/4 096 of them/)).toBeVisible();
});

test("the trained column follows the mask that was scored, not the one now on screen", async ({
  page,
}) => {
  await stubAblation(page);
  await page.goto("/explorer");
  await page.getByRole("button", { name: "Score 4 layers" }).click();
  await expect(page.getByText("90.62%")).toBeVisible();

  // Editing the strip afterwards must not relabel a result it no longer describes.
  await page.getByRole("switch", { name: "Layer 9, dropped" }).click();
  await expect(page.getByText("90.62%")).toBeVisible();
  await expect(page.getByText("62.15%")).toBeVisible();
});
