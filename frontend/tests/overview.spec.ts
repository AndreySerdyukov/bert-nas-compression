import { expect, test } from "./harness";

/**
 * The front page, whose job is to not mislead.
 *
 * "Compressed to under half the size for two and a half points" is true and is the sentence a
 * reader would leave with if the page led with the NAS table. These tests pin the arrangement that
 * prevents it: the control that beats every searched model is on the page, in the accent colour,
 * beside the best searched figure rather than below the fold.
 */

test("the headline puts a control beside the best searched model", async ({ page }) => {
  await page.goto("/");

  // Exact: the phrase also appears inside DistilBERT's caption, which compares against it.
  await expect(page.getByText("Best searched model", { exact: true })).toBeVisible();
  await expect(page.getByText("90.77%")).toBeVisible();

  // The control that outscores it, and the fact that it is not a transformer.
  await expect(page.getByText("TF-IDF + logistic regression")).toBeVisible();
  await expect(page.getByText("91.41%")).toBeVisible();
  await expect(page.getByText(/No transformer at all/)).toBeVisible();
});

test("the accuracy given up is computed against the baseline, not typed in", async ({ page }) => {
  await page.goto("/");

  // 0.933 - 0.9295 and 0.933 - 0.9077, to two decimals, from the fixtures.
  await expect(page.getByText(/Gives up 0\.35 points to full BERT/)).toBeVisible();
  await expect(page.getByText(/gives up 2\.53/)).toBeVisible();
});

test("without the controls the page still stands, minus the claims they support", async ({
  page,
}) => {
  await page.route("**/api/controls", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "not measured yet" }),
    });
  });
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByText("TF-IDF + logistic regression")).toHaveCount(0);
  // The paragraph that only makes sense with the controls goes with them.
  await expect(page.getByText(/inside the spread of masks chosen at random/)).toHaveCount(0);
});
