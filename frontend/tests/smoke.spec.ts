import { expect, test } from "./harness";

const ROUTES = ["/", "/playground", "/explorer", "/benchmark", "/methodology"] as const;

test("every route renders a heading and leaves the console clean", async ({ page, harness }) => {
  for (const route of ROUTES) {
    await page.goto(route);
    await expect(page.locator("h1")).toBeVisible();
  }
  expect(harness.consoleErrors).toEqual([]);
});

test("the methodology route renders compiled MDX", async ({ page }) => {
  await page.goto("/methodology");
  // The prose comes from an .mdx file. If the MDX plugin stopped compiling, the build would fail;
  // if it compiled to something empty, only an assertion on rendered text would notice.
  await expect(page.locator(".prose-nas p")).toContainText("MDX pipeline");
});

test("the theme toggle flips the root class and survives a reload", async ({ page }) => {
  await page.goto("/");
  const root = page.locator("html");
  await expect(root).not.toHaveClass(/dark/);

  await page.getByRole("button", { name: /switch to the dark theme/i }).click();
  await expect(root).toHaveClass(/dark/);

  await page.reload();
  await expect(root).toHaveClass(/dark/);
});

test("the page does not scroll sideways at a laptop width", async ({ page }) => {
  await page.setViewportSize({ width: 1100, height: 780 });
  await page.goto("/methodology");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
