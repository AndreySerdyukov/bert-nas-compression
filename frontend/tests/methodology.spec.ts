import { CHAPTER_META as CHAPTERS } from "../src/content/chapters";
import { expect, test } from "./harness";

test("the contents page lists every chapter", async ({ page }) => {
  await page.goto("/methodology");
  for (const chapter of CHAPTERS) {
    await expect(page.getByRole("link", { name: new RegExp(chapter.title, "i") })).toBeVisible();
  }
});

for (const chapter of CHAPTERS) {
  test(`chapter ${chapter.number} renders real prose`, async ({ page, harness }) => {
    await page.goto(`/methodology/${chapter.slug}`);
    await expect(page.locator("h1")).toContainText(chapter.title);

    // A broken MDX import compiles to an empty component and `tsc` never notices, so the guard has
    // to be on the rendered word count rather than on the module existing.
    const words = await page.locator("article.prose-nas").innerText();
    expect(words.split(/\s+/).length).toBeGreaterThan(150);

    expect(harness.consoleErrors).toEqual([]);
  });
}

test("j and k walk through the chapters", async ({ page }) => {
  await page.goto(`/methodology/${CHAPTERS[0]!.slug}`);
  await page.locator("body").press("j");
  await expect(page.locator("h1")).toContainText(CHAPTERS[1]!.title);
  await page.locator("body").press("k");
  await expect(page.locator("h1")).toContainText(CHAPTERS[0]!.title);
});

test("the layer mask widget prices a mask without asking the server", async ({ page, harness }) => {
  await page.goto("/methodology/the-search-space");

  const widget = page.getByRole("group", { name: /encoder layers kept/i });
  await expect(widget).toBeVisible();

  // The preset is Random Search's shipped mask: five layers, and the parameter count the formula
  // gives for five. This is what pins lib/arch.ts against the same ladder the Python tests pin.
  await page.getByRole("button", { name: "Random Search" }).click();
  await expect(page.getByText("59.9M")).toBeVisible();
  await expect(page.getByText("5 of 12")).toBeVisible();

  const before = harness.apiCalls.length;
  await page.getByRole("switch", { name: /layer 2,/i }).click();
  await expect(page.getByText("6 of 12")).toBeVisible();
  await expect(page.getByText("67.0M")).toBeVisible();
  // Six layers is 66 956 546 by the formula, and no request went out to learn that.
  expect(harness.apiCalls.length).toBe(before);
});

test("the search log renders its data as a table for screen readers", async ({ page }) => {
  await page.goto("/methodology/random-search");
  const table = page.locator("table.sr-only").first();
  // Three candidates in the Random Search log, and the third is the one that shipped.
  await expect(table.locator("tbody tr")).toHaveCount(3);
  await expect(table).toContainText("0 2 3 4 5 6 7 8 9 10");
});

test("a rejected candidate is drawn differently from a scored one", async ({ page }) => {
  await page.goto("/methodology/alphanas");
  const table = page.locator("table.sr-only").first();
  await expect(table).toContainText("parameter cap of 1e8 exceeded");
});
