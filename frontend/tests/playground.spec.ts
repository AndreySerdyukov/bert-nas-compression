import { COMPARE, EXAMPLES, MODELS, MODELS_WITHOUT_WEIGHTS } from "./fixtures";
import { expect, test } from "./harness";

/**
 * The playground, against a stubbed backend.
 *
 * The assertions are mostly about honesty rather than about layout: that a timing is never shown
 * without the machine behind it, that the token count travels with it, that a disagreement with
 * the baseline is the only thing marked, and that a clone with no weights gets an explanation
 * rather than an error.
 */

const REVIEW = EXAMPLES.examples[1]!;

test("scoring one review shows every model, with the machine behind the timings", async ({
  page,
  harness,
}) => {
  await page.goto("/playground");
  await page.getByLabel("The review").fill("A complete waste of time.");
  await page.getByRole("button", { name: /score with every model/i }).click();

  for (const model of MODELS.models) {
    await expect(page.locator(`[data-model="${model.name}"]`)).toBeVisible();
  }

  // The rule this whole project is built around: no millisecond figure without its context.
  const runtime = MODELS.runtime;
  await expect(
    page.getByText(`1 thread, cpu, ${runtime.machine}, torch ${runtime.torch_version}`),
  ).toBeVisible();

  expect(harness.consoleErrors).toEqual([]);
});

test("latency is labelled as a median and p95 over named repeats, never as a bare number", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("The review").fill("A complete waste of time.");
  await page.getByRole("button", { name: /score with every model/i }).click();

  const baseline = page.locator('[data-model="bert-imdb"]');
  await expect(baseline).toContainText("Latency");
  await expect(baseline).toContainText("20.0 ms median");
  await expect(baseline).toContainText("25.1 ms p95");
  await expect(baseline).toContainText("5 repeats, 3 warm-up passes discarded");
});

test("the token count travels with the timing, and it is not the same for every model", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("The review").fill("A complete waste of time.");
  await page.getByRole("button", { name: /score with every model/i }).click();

  // AdaBERT reads a padded 128-token window; the others read the 32 tokens the review is worth.
  await expect(page.locator('[data-model="bert-imdb"]')).toContainText("32 tokens");
  await expect(page.locator('[data-model="adabert"]')).toContainText("128 tokens");
});

test("only the model that parts from the baseline is marked", async ({ page }) => {
  await page.goto("/playground");
  await page.getByLabel("The review").fill("A mixed review.");
  await page.getByRole("button", { name: /score with every model/i }).click();

  await expect(page.getByRole("heading", { name: /1 of 4 part from the baseline/i })).toBeVisible();
  await expect(page.locator('[data-model="adabert"]')).toContainText("disagrees");
  await expect(page.locator('[data-model="bert-imdb"]')).toContainText("baseline");
  for (const agreeing of ["alphanas", "bananas", "random-search"]) {
    await expect(page.locator(`[data-model="${agreeing}"]`)).not.toContainText("disagrees");
  }
});

test("an example carries its corpus label, and editing it takes the label away", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByRole("button", { name: new RegExp(`#${REVIEW.id}`) }).click();
  await expect(page.getByLabel("The review")).toHaveValue(REVIEW.text);

  await page.getByRole("button", { name: /score with every model/i }).click();
  // The stubbed verdict for the baseline is negative and this example is labelled negative.
  await expect(page.locator('[data-model="bert-imdb"]')).toContainText("matches the corpus label");
  // AdaBERT said positive, so against the same label it is wrong.
  await expect(page.locator('[data-model="adabert"]')).toContainText("wrong");

  await page.getByLabel("The review").fill(`${REVIEW.text} and one more sentence.`);
  await expect(page.getByText("The review has been edited since these were scored.")).toBeVisible();
  await expect(page.locator('[data-model="bert-imdb"]')).not.toContainText("corpus label");
});

test("corpus markup is left in the review, and the page says why", async ({ page }) => {
  await page.goto("/playground");
  await page.getByLabel("The review").fill("A film that <br /><br />goes nowhere.");
  await expect(page.getByText(/left in on purpose/i)).toBeVisible();
  // And it is sent as-is: stripping it would score the models on text they never saw.
  await expect(page.getByLabel("The review")).toHaveValue("A film that <br /><br />goes nowhere.");

  await page.getByLabel("The review").fill("A film that goes nowhere.");
  await expect(page.getByText(/left in on purpose/i)).toHaveCount(0);
});

test("a clone with no weights is explained rather than broken", async ({ page, harness }) => {
  await page.route("**/api/models", async (route) => {
    harness.apiCalls.push("/api/models");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MODELS_WITHOUT_WEIGHTS),
    });
  });

  await page.goto("/playground");
  await expect(page.getByText("No checkpoints have been downloaded")).toBeVisible();
  await expect(page.getByText("python scripts/fetch_models.py", { exact: true })).toBeVisible();
  // Every skipped model says what to run for it specifically.
  await expect(page.getByText("python scripts/fetch_models.py --only adabert")).toBeVisible();
  // Nothing to score with, so the button stays out of reach.
  await expect(page.getByRole("button", { name: /score with every model/i })).toBeDisabled();

  expect(harness.consoleErrors).toEqual([]);
});

test("serving switched off says so instead of looking like a failure", async ({ page }) => {
  await page.route("**/api/models", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "serving is switched off (APP_MODEL_TIER=none)" }),
    });
  });

  await page.goto("/playground");
  await expect(page.getByText(/serving is switched off in this deployment/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "One review, every model" })).toBeVisible();
});

test("a refused request is shown as its reason, not as an empty result", async ({ page }) => {
  await page.route("**/api/compare", async (route) => {
    await route.fulfill({
      status: 429,
      contentType: "application/json",
      body: JSON.stringify({ detail: "another request is being scored." }),
    });
  });

  await page.goto("/playground");
  await page.getByLabel("The review").fill("A complete waste of time.");
  await page.getByRole("button", { name: /score with every model/i }).click();

  await expect(page.getByText("another request is being scored.")).toBeVisible();
  await expect(page.locator(`[data-model="${COMPARE.results[0]!.name}"]`)).toHaveCount(0);
});

test("the scan streams its progress and then shows the rows that split the models", async ({
  page,
  harness,
}) => {
  await page.goto("/playground");
  await page.getByRole("button", { name: "Scan all 2,000" }).click();

  await expect(
    page.getByRole("heading", { name: /412 of 2,000 reviews split the models/i }),
  ).toBeVisible();

  // The summary is per model, and the baseline is not compared against itself.
  await expect(page.locator('[data-scan-model="bert-imdb"]')).toContainText("96.00%");
  await expect(page.locator('[data-scan-model="adabert"]')).toContainText("87.00%");

  // The rows are the point of the whole feature.
  await expect(page.locator("[data-disagreement]")).toHaveCount(2);
  await expect(page.locator('[data-disagreement="8761"]')).toContainText("adabert: negative");

  // A cap that went unmentioned would read as though these were all of them.
  await expect(page.getByText(/2 of the 412 disagreeing reviews/i)).toBeVisible();
  // And the scan says it is not a timing, because it dropped the thread pin to run.
  await expect(page.getByText(/not a measurement of speed/i)).toBeVisible();

  expect(harness.consoleErrors).toEqual([]);
});

test("the scan result is never confused with the published benchmark", async ({ page }) => {
  await page.goto("/playground");
  await page.getByRole("button", { name: "First 200" }).click();
  await expect(page.getByText(/not the full 15 000-row test split/i)).toBeVisible();
});
