import { expect, test } from "./harness";

/**
 * The benchmark page, which is where this project's own numbers meet the controls.
 *
 * The assertions are on figures rather than on layout. Two of them exist because of specific ways
 * this page could be wrong and still look right: accuracies rendered at one decimal would merge
 * AlphaNAS into BANANAS, and a percentile computed with "at or below" would flip the finding from
 * "the search sits below the median random mask" to "above it".
 */

test("the model table separates two models that differ in the second decimal", async ({ page }) => {
  await page.goto("/benchmark");

  // Scoped to the model table: the pareto chart and the latency chart each render their own
  // screen-reader table with a row per model, so an unscoped row locator matches three.
  const table = page.getByRole("table", { name: /models measured/i });
  await expect(table.getByRole("row", { name: /AlphaNAS/ })).toContainText("90.27%");
  await expect(table.getByRole("row", { name: /BANANAS/ })).toContainText("90.31%");
});

test("each depth gets its own verdict, and the two depths disagree", async ({ page }) => {
  await page.goto("/benchmark");

  const fourLayers = page.getByRole("heading", { name: "4 layers" }).locator("..");
  // BANANAS 0.9031 against the evenly spaced 0.8959, and against the bottom four at 0.9062.
  await expect(fourLayers).toContainText("+0.72 pp");
  await expect(fourLayers).toContainText("-0.31 pp");
  // One of the three random masks scores below BANANAS, so the percentile is 33 and not 67.
  await expect(fourLayers).toContainText("33th");

  const fiveLayers = page.getByRole("heading", { name: "5 layers" }).locator("..");
  // Random Search 0.9077 loses to the evenly spaced mask, by four reviews out of fifteen thousand.
  await expect(fiveLayers).toContainText("-0.03 pp");
  await expect(fiveLayers).toContainText("-0.37 pp");
});

test("a depth with no random masks drops the percentile rather than inventing one", async ({
  page,
}) => {
  await page.goto("/benchmark");

  const fiveLayers = page.getByRole("heading", { name: "5 layers" }).locator("..");
  await expect(fiveLayers).not.toContainText("percentile");
  await expect(fiveLayers).not.toContainText("NaN");
});

test("the pareto chart tells controls apart from shipped models in its data table", async ({
  page,
}) => {
  await page.goto("/benchmark");

  const table = page.locator("table.sr-only").first();
  await expect(table).toContainText("control");
  await expect(table).toContainText("baseline");
  // The control rows carry their own accuracies, so a screen reader gets the comparison too.
  await expect(table).toContainText("87.99%");
});

test("TF-IDF is on the page but never on the parameter axis", async ({ page }) => {
  await page.goto("/benchmark");

  // It has no parameter count comparable to a transformer's, so it cannot be a point on the chart.
  await expect(page.getByText("TF-IDF + logistic regression").first()).toBeVisible();
  const chartTable = page.locator("table.sr-only").first();
  await expect(chartTable).not.toContainText("TF-IDF");
});

test("latency is drawn with its p95 and never divided into a throughput", async ({ page }) => {
  await page.goto("/benchmark");

  const chart = page.getByRole("img", { name: /latency for \d+ models/i });
  await expect(chart).toBeVisible();

  const table = page.locator("table.sr-only").nth(1);
  await expect(table).toContainText("p95 ms");
  await expect(table).toContainText("tokens");
});

test("missing controls read as an explanation with a command, not as an error", async ({
  page,
}) => {
  await page.route("**/api/controls", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "controls.json is not present. Run python -m training.train_reference",
      }),
    });
  });
  await page.goto("/benchmark");

  await expect(page.getByText("python -m training.train_reference")).toBeVisible();
  // The half that was measured still renders: one absent document does not take the page down.
  const table = page.getByRole("table", { name: /models measured/i });
  await expect(table.getByRole("row", { name: /BERT-base/ })).toContainText("93.30%");
  await expect(page.getByRole("heading", { name: /Did searching beat/ })).toHaveCount(0);
});

test("a partial control run says so instead of showing the survivors as the plan", async ({
  page,
}) => {
  await page.route("**/api/controls", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: 1,
        generated_by: "training/train_reference.py",
        note: "stub",
        protocol: {
          epochs: 1,
          train_rows: 28_000,
          test_rows: 15_000,
          test_index_sha256: "33d7fee",
          max_length: 512,
          batch_size: 16,
          learning_rate: 3e-5,
          weight_decay: 0.01,
          warmup_ratio: 0.1,
          base_model: "bert-base-uncased",
          device: "mps",
          cooldown_seconds: 120,
          torch: "2.13.0",
          source: "stub",
        },
        planned: ["tfidf", "uniform-4", "uniform-5"],
        not_run: ["uniform-5"],
        weights_saved: false,
        weights_note: "stub",
        controls: [
          {
            key: "tfidf",
            label: "TF-IDF + logistic regression",
            question: "How much of this task needs a transformer at all?",
            priority: 1,
            kind: "tfidf",
            layers: null,
            n_layers: null,
            seed: null,
            train: { seconds: 9.78, steps: null },
            correct: 13_712,
            accuracy: 0.9141,
            macro_precision: 0.9142,
            macro_recall: 0.914,
            macro_f1: 0.9141,
            positive_precision: 0.9103,
            positive_recall: 0.9211,
            positive_f1: 0.9156,
            confusion: {
              true_negative: 6722,
              false_positive: 689,
              false_negative: 599,
              true_positive: 6990,
            },
          },
          {
            key: "uniform-4",
            label: "Evenly spaced, 4 layers",
            question: "Did searching beat keeping every third layer?",
            priority: 2,
            kind: "mask",
            layers: [0, 4, 7, 11],
            n_layers: 4,
            seed: null,
            train: { seconds: 1385.55, steps: 1750, params: 52_780_802 },
            correct: 13_439,
            accuracy: 0.8959,
            macro_precision: 0.8958,
            macro_recall: 0.8958,
            macro_f1: 0.8958,
            positive_precision: 0.8958,
            positive_recall: 0.8958,
            positive_f1: 0.8958,
            confusion: {
              true_negative: 6700,
              false_positive: 711,
              false_negative: 850,
              true_positive: 6739,
            },
          },
        ],
      }),
    });
  });
  await page.goto("/benchmark");

  await expect(page.getByText(/Not run: uniform-5/)).toBeVisible();
  await expect(page.getByText(/2 of 3 planned controls/)).toBeVisible();
});
