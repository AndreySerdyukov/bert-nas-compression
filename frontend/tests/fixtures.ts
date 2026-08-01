import type {
  Architectures,
  Benchmark,
  CompareResponse,
  Controls,
  ExamplesResponse,
  JobSnapshot,
  ModelsResponse,
  ReportedResults,
  ScanResult,
  Scored,
  Trajectories,
} from "../src/api";

/**
 * Stub payloads for the browser tests.
 *
 * Typed against `src/api.ts`, so a change to the contract breaks `tsc` here rather than producing
 * tests that pass against a shape the app no longer serves. The values are the real ones - the
 * shipped masks, the real Random Search log - because several assertions are about specific
 * numbers, and a fixture full of 1s and 2s would let a units bug through.
 */

export const ARCHITECTURES: Architectures = {
  schema: 1,
  baseline: {
    label: "BERT-base",
    hf_repo: "AndreySerdyukov/bert-imdb",
    n_layers: 12,
    params: 109_483_778,
    eval_notebook: "notebooks/bert_finetune/Bert_Finetune_Andrey.ipynb",
  },
  params_formula: { base: 24_429_314, per_layer: 7_087_872 },
  methods: {
    "random-search": {
      label: "Random Search",
      hf_repo: "AndreySerdyukov/random-search",
      eval_notebook: "notebooks/random_search/RandomSearch_Eval.ipynb",
      selection_notebook: "notebooks/random_search/RandomSearch_Selection.ipynb",
      shipped: {
        mask: [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        layers: [0, 1, 5, 7, 9],
        n_layers: 5,
        params: 59_868_674,
        finetune_epochs: 1,
      },
      search_best: {
        mask: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        layers: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      },
      shipped_matches_search: false,
    },
    alphanas: {
      label: "AlphaNAS",
      hf_repo: "AndreySerdyukov/alphanas",
      eval_notebook: "notebooks/alphanas/AlphaNas_Eval.ipynb",
      selection_notebook: "notebooks/alphanas/AlphaNas_Selection.ipynb",
      shipped: {
        mask: [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        layers: [1, 2, 6, 11],
        n_layers: 4,
        params: 52_780_802,
        finetune_epochs: 2,
      },
      search_best: { mask: [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], layers: [1, 2, 6, 11] },
      shipped_matches_search: true,
    },
    bananas: {
      label: "BANANAS",
      hf_repo: "alinaselivanets/bananas-bert",
      eval_notebook: "notebooks/bananas/BANANAS_Eval.ipynb",
      selection_notebook: "notebooks/bananas/BANANAS_Selection.ipynb",
      shipped: {
        mask: [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        layers: [0, 1, 6, 9],
        n_layers: 4,
        params: 52_780_802,
        finetune_epochs: 1,
      },
      search_best: { mask: [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], layers: [0, 1, 6, 9] },
      shipped_matches_search: true,
    },
    adabert: {
      label: "AdaBERT",
      hf_repo: "ilkonz/dnas",
      eval_notebook: "notebooks/dnas/Differentiable_NAS.ipynb",
      selection_notebook: "notebooks/dnas/Differentiable_NAS.ipynb",
      shipped: { mask: null, layers: null, n_layers: null, params: 7_814_146, finetune_epochs: 3 },
      search_best: null,
      shipped_matches_search: null,
      note: "Every searched operation was avg_pool, which is parameter-free.",
    },
  },
};

const BUDGET = {
  train_rows: 2800,
  val_rows: 700,
  max_length: 512,
  epochs_per_candidate: 1,
  fitness: "alpha * accuracy + beta * (1 - log(params + 1) / log(max_params))",
  fitness_params: { alpha: 0.3, beta: 0.7, max_params: 110_000_000 },
};

export const TRAJECTORIES: Trajectories = {
  schema: 1,
  params_formula: { base: 24_429_314, per_layer: 7_087_872 },
  methods: {
    "random-search": {
      notebook: "notebooks/random_search/RandomSearch_Selection.ipynb",
      budget: BUDGET,
      stages: [],
      candidates: [
        {
          mask: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          layers: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          n_layers: 10,
          accuracy: 0.9157,
          params: 95_308_034,
          fitness: 0.2801,
          trial: 1,
        },
        {
          mask: [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
          layers: [0, 2, 3, 4, 9],
          n_layers: 5,
          accuracy: 0.8329,
          params: 59_868_674,
          fitness: 0.2729,
          trial: 2,
        },
        {
          mask: [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
          layers: [0, 1, 6, 8, 10],
          n_layers: 5,
          accuracy: 0.8429,
          params: 59_868_674,
          fitness: 0.2759,
          trial: 3,
        },
      ],
    },
    alphanas: {
      notebook: "notebooks/alphanas/AlphaNas_Selection.ipynb",
      budget: { ...BUDGET, train_rows: 5600, val_rows: 1400 },
      stages: [],
      candidates: [
        {
          mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          n_layers: 12,
          accuracy: 0.8979,
          params: 109_483_778,
          fitness: null,
          generation: 1,
          rejected: true,
          rejection_reason: "parameter cap of 1e8 exceeded",
        },
        {
          mask: [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
          layers: [1, 2, 6, 11],
          n_layers: 4,
          accuracy: 0.8493,
          params: 52_780_802,
          fitness: 0.321478,
          generation: 1,
          rejected: false,
        },
      ],
    },
    bananas: {
      notebook: "notebooks/bananas/BANANAS_Selection.ipynb",
      budget: { ...BUDGET, train_rows: 560, val_rows: 140, max_length: 128 },
      stages: [],
      proposals: [
        {
          round: 0,
          mask: [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
          n_layers: 5,
          predicted_fitness: -0.031418,
        },
      ],
      candidates: [
        {
          mask: [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
          layers: [0, 1, 6, 9],
          n_layers: 4,
          accuracy: 0.6214,
          params: 52_780_802,
          fitness: 0.093621,
          phase: "bayesian",
        },
      ],
    },
  },
};

export const REPORTED_RESULTS: ReportedResults = {
  schema: 1,
  note: "stub",
  sources: {
    nas_results: {
      label: "NAS_Results.ipynb",
      kind: "notebook",
      canonical: true,
      protocol: "15 000 test rows",
    },
  },
  models: [
    {
      key: "bert-andrey",
      label: "BERT-base (Andrey)",
      hf_repo: "AndreySerdyukov/bert-imdb",
      n_layers: 12,
      params: 109_483_778,
      accuracy: { nas_results: 0.933 },
    },
    {
      key: "random-search",
      label: "Random Search",
      hf_repo: "AndreySerdyukov/random-search",
      n_layers: 5,
      params: 59_868_674,
      accuracy: { nas_results: 0.9077 },
    },
    {
      key: "alphanas",
      label: "AlphaNAS",
      hf_repo: "AndreySerdyukov/alphanas",
      n_layers: 4,
      params: 52_780_802,
      accuracy: { nas_results: 0.9027 },
    },
    {
      key: "adabert",
      label: "AdaBERT",
      hf_repo: "ilkonz/dnas",
      n_layers: null,
      params: 7_814_146,
      accuracy: { nas_results: 0.8801 },
    },
  ],
  latency_caveat: "stub",
};

// --- serving ------------------------------------------------------------------------------------

/**
 * The catalog, the offered reviews and one real comparison, captured from the running backend on
 * 2026-07-31 with all five checkpoints downloaded.
 *
 * The comparison is a real one and was not chosen to be tidy: AdaBERT calls this mixed review
 * positive while the baseline and the three searched descendants call it negative. That is the
 * page's whole subject, so the fixture has to contain an actual disagreement rather than five
 * models nodding along.
 */
export const MODELS: ModelsResponse = {
  baseline: "bert-imdb",
  runtime: {
    device: "cpu",
    threads: 1,
    interop_threads: 1,
    machine: "Darwin arm64",
    torch_version: "2.13.0",
  },
  skipped: [],
  models: [
    {
      name: "bert-imdb",
      label: "BERT-base",
      description: "The uncompressed reference: bert-base-uncased fine-tuned on IMDB.",
      method: null,
      params: 109_483_778,
      n_layers: 12,
      max_length: 512,
      hf_repo: "AndreySerdyukov/bert-imdb",
      hf_revision: "79678c3fb67dbaff65bba99a733ebd4f5e35265e",
      positive_index: 1,
      is_baseline: true,
    },
    {
      name: "adabert",
      label: "AdaBERT",
      description: "Differentiable NAS with distillation. Every searched operation was free.",
      method: "adabert",
      params: 7_814_146,
      n_layers: null,
      max_length: 128,
      hf_repo: "ilkonz/dnas",
      hf_revision: "36f59582dff22e51c507cbb73390457b0538f9e2",
      positive_index: 1,
      is_baseline: false,
    },
    {
      name: "alphanas",
      label: "AlphaNAS",
      description: "Four encoder layers found by evolutionary search over the layer mask.",
      method: "alphanas",
      params: 52_780_802,
      n_layers: 4,
      max_length: 512,
      hf_repo: "AndreySerdyukov/alphanas",
      hf_revision: "e9620fe43d485e31bc5f5f3604f6abf69f35460a",
      positive_index: 1,
      is_baseline: false,
    },
    {
      name: "bananas",
      label: "BANANAS",
      description: "Four encoder layers found by Bayesian optimisation over a surrogate.",
      method: "bananas",
      params: 52_780_802,
      n_layers: 4,
      max_length: 512,
      hf_repo: "alinaselivanets/bananas-bert",
      hf_revision: "05150dab0ec9b37c6f8d36ec5fe1b58e2c8f6f36",
      positive_index: 1,
      is_baseline: false,
    },
    {
      name: "random-search",
      label: "Random Search",
      description: "Five encoder layers found by random sampling over the layer mask.",
      method: "random-search",
      params: 59_868_674,
      n_layers: 5,
      max_length: 512,
      hf_repo: "AndreySerdyukov/random-search",
      hf_revision: "f76a82a54344974fb852f2e21f51db716b420d0d",
      positive_index: 1,
      is_baseline: false,
    },
  ],
};

/** The catalog a fresh clone gets: five manifests, no weights, a command for each. */
export const MODELS_WITHOUT_WEIGHTS: ModelsResponse = {
  ...MODELS,
  models: [],
  skipped: MODELS.models.map((model) => ({
    name: model.name,
    label: model.label,
    reason: `the weights are not downloaded (no snapshot for '${model.name}')`,
    remedy: `python scripts/fetch_models.py --only ${model.name}`,
  })),
};

export const EXAMPLES: ExamplesResponse = {
  total: 2000,
  source: "data/imdb_eval_sample.jsonl",
  examples: [
    {
      id: 44944,
      text: '"The Cure" is a very touching and poignant drama about two neighbours who become friends.',
      label: 1,
      n_chars: 575,
      truncated: false,
    },
    {
      id: 34486,
      text: "The really sad thing is that this was supposedly the highest budget film of its year.",
      label: 0,
      n_chars: 655,
      truncated: false,
    },
    {
      id: 8761,
      text: "If it is true that sadomasochism is a two-sided coin, this film spends four thousand characters proving it.",
      label: 1,
      n_chars: 4689,
      truncated: true,
    },
  ],
};

export const COMPARE: CompareResponse = {
  baseline: "bert-imdb",
  baseline_scored: true,
  runtime: MODELS.runtime,
  disagree: ["adabert"],
  results: [
    {
      name: "bert-imdb",
      label: "BERT-base",
      verdict: "negative",
      positive_probability: 0.005_674_272_775_65,
      probabilities: { negative: 0.994_325_697_422_03, positive: 0.005_674_272_775_65 },
      n_tokens: 32,
      params: 109_483_778,
      n_layers: 12,
      elapsed_ms: 20.355,
      latency: {
        batch_size: 1,
        warmup: 3,
        repeats: 5,
        median_ms: 19.954,
        p95_ms: 25.084,
        n_tokens: 32,
      },
      throughput: null,
      agrees_with_baseline: null,
    },
    {
      name: "adabert",
      label: "AdaBERT",
      verdict: "positive",
      positive_probability: 0.784_371_912_479_4,
      probabilities: { negative: 0.215_628_162_026_4, positive: 0.784_371_912_479_4 },
      // 128 against everyone else's 32: the model reads a padded window, and the card shows it.
      n_tokens: 128,
      params: 7_814_146,
      n_layers: null,
      elapsed_ms: 1.033,
      latency: {
        batch_size: 1,
        warmup: 3,
        repeats: 5,
        median_ms: 0.889,
        p95_ms: 0.958,
        n_tokens: 128,
      },
      throughput: null,
      agrees_with_baseline: false,
    },
    {
      name: "alphanas",
      label: "AlphaNAS",
      verdict: "negative",
      positive_probability: 0.319_147_318_601_6,
      probabilities: { negative: 0.680_852_711_200_7, positive: 0.319_147_318_601_6 },
      n_tokens: 32,
      params: 52_780_802,
      n_layers: 4,
      elapsed_ms: 9.157,
      latency: {
        batch_size: 1,
        warmup: 3,
        repeats: 5,
        median_ms: 6.715,
        p95_ms: 7.044,
        n_tokens: 32,
      },
      throughput: null,
      agrees_with_baseline: true,
    },
    {
      name: "bananas",
      label: "BANANAS",
      verdict: "negative",
      positive_probability: 0.061_227_552_592_75,
      probabilities: { negative: 0.938_772_499_561_31, positive: 0.061_227_552_592_75 },
      n_tokens: 32,
      params: 52_780_802,
      n_layers: 4,
      elapsed_ms: 8.379,
      latency: {
        batch_size: 1,
        warmup: 3,
        repeats: 5,
        median_ms: 6.811,
        p95_ms: 7.305,
        n_tokens: 32,
      },
      throughput: null,
      agrees_with_baseline: true,
    },
    {
      name: "random-search",
      label: "Random Search",
      verdict: "negative",
      positive_probability: 0.141_892_001_032_83,
      probabilities: { negative: 0.858_107_984_066_01, positive: 0.141_892_001_032_83 },
      n_tokens: 32,
      params: 59_868_674,
      n_layers: 5,
      elapsed_ms: 10.133,
      latency: {
        batch_size: 1,
        warmup: 3,
        repeats: 5,
        median_ms: 8.46,
        p95_ms: 9.212,
        n_tokens: 32,
      },
      throughput: null,
      agrees_with_baseline: true,
    },
  ],
};

/**
 * The same call on a machine where the baseline is not being served - one checkpoint fetched, or
 * the baseline dropped by the polarity probe. `disagree` is empty because nothing was compared,
 * and the page has to say that rather than read the empty list as agreement.
 */
export const COMPARE_WITHOUT_BASELINE: CompareResponse = {
  ...COMPARE,
  baseline_scored: false,
  disagree: [],
  results: COMPARE.results
    .filter((entry) => entry.name !== "bert-imdb")
    .map((entry) => ({ ...entry, agrees_with_baseline: null })),
};

/**
 * A disagreement scan, captured from a real 100-row run: the same models, the same shape, and a
 * genuine split. The verdicts in the rows below are what the checkpoints actually said.
 */
export const SCAN_RESULT: ScanResult = {
  baseline: "bert-imdb",
  rows_scanned: 2000,
  threads_used: 11,
  machine: "Darwin arm64",
  models: [
    {
      name: "bert-imdb",
      label: "Bert-Imdb",
      params: 109_483_778,
      n_layers: 12,
      correct: 1920,
      accuracy: 0.96,
      agree_with_baseline: null,
      agreement: null,
    },
    {
      name: "adabert",
      label: "Adabert",
      params: 7_814_146,
      n_layers: null,
      correct: 1820,
      accuracy: 0.91,
      agree_with_baseline: 1740,
      agreement: 0.87,
    },
    {
      name: "bananas",
      label: "Bananas",
      params: 52_780_802,
      n_layers: 4,
      correct: 1860,
      accuracy: 0.93,
      agree_with_baseline: 1860,
      agreement: 0.93,
    },
  ],
  disagreements_found: 412,
  disagreements_shown: 2,
  disagreements: [
    {
      id: 8761,
      label: "positive",
      n_chars: 4689,
      excerpt: "If it is true that sadomasochism is a two-sided coin which cuts both ways",
      truncated: true,
      verdicts: { "bert-imdb": "positive", adabert: "negative", bananas: "positive" },
    },
    {
      id: 14881,
      label: "negative",
      n_chars: 4528,
      excerpt: "In short, this movie is a declaration of artistic bankruptcy.",
      truncated: true,
      verdicts: { "bert-imdb": "negative", adabert: "positive", bananas: "negative" },
    },
  ],
};

const SCAN_STAGES = ["scoring with Bert-Imdb", "scoring with Adabert", "comparing"];

export const SCAN_STARTED: JobSnapshot = {
  id: "job000000001",
  kind: "disagreement",
  status: "running",
  stages: SCAN_STAGES,
  stage: null,
  stage_index: 0,
  processed: 0,
  total: 0,
  messages: [],
  elapsed_s: 0,
  result: null,
  error: null,
};

/** The frames a real run emits, trimmed to the three that matter: start, progress, terminal. */
export const SCAN_FRAMES: JobSnapshot[] = [
  { ...SCAN_STARTED, stage: SCAN_STAGES[0]!, total: 2000, processed: 320, elapsed_s: 12 },
  {
    ...SCAN_STARTED,
    stage: SCAN_STAGES[1]!,
    stage_index: 1,
    total: 2000,
    processed: 1440,
    elapsed_s: 96,
  },
  {
    ...SCAN_STARTED,
    status: "done",
    stage: SCAN_STAGES[2]!,
    stage_index: 3,
    total: 2000,
    processed: 2000,
    elapsed_s: 402,
    messages: ["412 of 2000 reviews split the models"],
    result: SCAN_RESULT as unknown as Record<string, unknown>,
  },
];

// --- what this project measured itself ----------------------------------------------------------

/**
 * Scores for a fixture row, from its accuracy alone.
 *
 * The accuracies are the real measured ones, because the assertions are about them: the page has to
 * separate AlphaNAS at 0.9027 from BANANAS at 0.9031, and a fixture that rounded them together
 * would let that bug through. The derived metrics are not independently real - macro F1 tracks
 * accuracy here, which on the balanced IMDB split it very nearly does anyway - and nothing asserts
 * a relationship between them.
 */
function scored(accuracy: number, rows = 15_000): Scored {
  const correct = Math.round(accuracy * rows);
  const half = Math.round(correct / 2);
  const wrong = rows - correct;
  return {
    correct,
    accuracy,
    macro_precision: accuracy,
    macro_recall: accuracy,
    macro_f1: accuracy,
    positive_precision: accuracy,
    positive_recall: accuracy,
    positive_f1: accuracy,
    confusion: {
      true_negative: correct - half,
      false_positive: wrong - Math.round(wrong / 2),
      false_negative: Math.round(wrong / 2),
      true_positive: half,
    },
  };
}

export const BENCHMARK: Benchmark = {
  schema: 1,
  generated_by: "training/benchmark.py",
  note: "Measured here, not reported by the notebooks.",
  protocol: {
    test_rows: 15_000,
    test_index_sha256: "33d7fee1070469ac628d568dfa95fc960d8d76e8a548acdb553fbb15ce1796c9",
    corpus_sha256: "dfc447764f82be365fa9c2beef4e8df89d3919e3da95f5088004797d79695aa2",
    accuracy_device: "mps",
    latency_device: "cpu",
    latency_threads: 1,
    latency_warmup: 5,
    latency_repeats: 25,
    throughput_batch: 16,
    score_batch: 32,
    machine: "Darwin arm64",
    python: "3.12.13",
    torch: "2.13.0",
    transformers: "5.14.1",
    memory_measured: "resident set size of a process holding exactly one loaded and warmed model",
  },
  models: [
    {
      name: "bert-imdb",
      label: "BERT-base",
      method: null,
      n_layers: 12,
      max_length: 512,
      hf_repo: "AndreySerdyukov/bert-imdb",
      hf_revision: "79678c3",
      params: 109_483_778,
      params_formula: 109_483_778,
      params_agree: true,
      flops_per_example: 96_637_946_880,
      bytes_on_disk: 438_192_240,
      bytes_fp32: 437_935_112,
      score_seconds: 452.31,
      latency: {
        batch_size: 1,
        warmup: 5,
        repeats: 25,
        median_ms: 20.7,
        p95_ms: 21.0,
        n_tokens: 33,
      },
      throughput: {
        batch_size: 16,
        warmup: 1,
        repeats: 5,
        median_batch_ms: 125.3,
        per_example_ms: 7.8,
        examples_per_second: 128,
        n_tokens: 33,
      },
      resident_bytes: 797_884_416,
      runtime_baseline_bytes: 411_566_080,
      ...scored(0.933),
    },
    {
      name: "random-search",
      label: "Random Search",
      method: "random-search",
      n_layers: 5,
      max_length: 512,
      hf_repo: "AndreySerdyukov/random-search",
      hf_revision: "aa11bb2",
      params: 59_868_674,
      params_formula: 59_868_674,
      params_agree: true,
      flops_per_example: 40_265_811_200,
      bytes_on_disk: 239_600_000,
      bytes_fp32: 239_474_696,
      score_seconds: 240.1,
      latency: { batch_size: 1, warmup: 5, repeats: 25, median_ms: 8.9, p95_ms: 9.2, n_tokens: 33 },
      throughput: {
        batch_size: 16,
        warmup: 1,
        repeats: 5,
        median_batch_ms: 53.5,
        per_example_ms: 3.3,
        examples_per_second: 299,
        n_tokens: 33,
      },
      resident_bytes: 594_000_000,
      runtime_baseline_bytes: 411_566_080,
      ...scored(0.9077),
    },
    {
      name: "alphanas",
      label: "AlphaNAS",
      method: "alphanas",
      n_layers: 4,
      max_length: 512,
      hf_repo: "AndreySerdyukov/alphanas",
      hf_revision: "cc33dd4",
      params: 52_780_802,
      params_formula: 52_780_802,
      params_agree: true,
      flops_per_example: 32_212_254_720,
      bytes_on_disk: 211_200_000,
      bytes_fp32: 211_123_208,
      score_seconds: 205.4,
      latency: { batch_size: 1, warmup: 5, repeats: 25, median_ms: 7.2, p95_ms: 7.5, n_tokens: 33 },
      throughput: {
        batch_size: 16,
        warmup: 1,
        repeats: 5,
        median_batch_ms: 44.2,
        per_example_ms: 2.8,
        examples_per_second: 362,
        n_tokens: 33,
      },
      resident_bytes: 558_000_000,
      runtime_baseline_bytes: 411_566_080,
      ...scored(0.9027),
    },
    {
      name: "bananas",
      label: "BANANAS",
      method: "bananas",
      n_layers: 4,
      max_length: 512,
      hf_repo: "alinaselivanets/bananas-bert",
      hf_revision: "ee55ff6",
      params: 52_780_802,
      params_formula: 52_780_802,
      params_agree: true,
      flops_per_example: 32_212_254_720,
      bytes_on_disk: 211_200_000,
      bytes_fp32: 211_123_208,
      score_seconds: 206.0,
      latency: { batch_size: 1, warmup: 5, repeats: 25, median_ms: 7.3, p95_ms: 7.6, n_tokens: 33 },
      throughput: {
        batch_size: 16,
        warmup: 1,
        repeats: 5,
        median_batch_ms: 44.6,
        per_example_ms: 2.8,
        examples_per_second: 359,
        n_tokens: 33,
      },
      resident_bytes: 557_000_000,
      runtime_baseline_bytes: 411_566_080,
      ...scored(0.9031),
    },
  ],
};

/**
 * A subset of the eighteen, chosen so the page's arithmetic is checkable by hand.
 *
 * Three random masks at depth four, not five: the median of 0.8940, 0.9039 and 0.9065 is 0.9039,
 * which AlphaNAS at 0.9027 sits below, and one of the three is below AlphaNAS - so the percentile
 * the page prints has to come out 33, and a helper that counted "at or below" would print 67.
 */
export const CONTROLS: Controls = {
  schema: 1,
  generated_by: "training/train_reference.py",
  note: "Controls, so the searched architectures have something to be compared against.",
  protocol: {
    epochs: 1,
    train_rows: 28_000,
    test_rows: 15_000,
    test_index_sha256: "33d7fee1070469ac628d568dfa95fc960d8d76e8a548acdb553fbb15ce1796c9",
    max_length: 512,
    batch_size: 16,
    learning_rate: 3e-5,
    weight_decay: 0.01,
    warmup_ratio: 0.1,
    base_model: "bert-base-uncased",
    device: "mps",
    cooldown_seconds: 120,
    torch: "2.13.0",
    source: "notebooks/*/*_Eval.ipynb, which agree on every value.",
  },
  planned: [
    "tfidf",
    "uniform-4",
    "uniform-5",
    "first-4",
    "first-5",
    "last-4",
    "distilbert",
    "random-4-seed0",
    "random-4-seed1",
    "random-4-seed2",
  ],
  not_run: [],
  weights_saved: false,
  weights_note: "The control's deliverable is a number.",
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
      train: { seconds: 9.78, steps: null, features: 200_000, note: "not a transformer" },
      ...scored(0.9141),
    },
    {
      key: "uniform-4",
      label: "Evenly spaced, 4 layers",
      question: "Did searching beat spreading the layers evenly?",
      priority: 2,
      kind: "mask",
      layers: [0, 4, 7, 11],
      n_layers: 4,
      seed: null,
      train: { seconds: 1385.55, steps: 1750, params: 52_780_802 },
      ...scored(0.8959),
    },
    {
      key: "uniform-5",
      label: "Evenly spaced, 5 layers",
      question: "Did searching beat spreading the layers evenly?",
      priority: 2,
      kind: "mask",
      layers: [0, 3, 6, 8, 11],
      n_layers: 5,
      seed: null,
      train: { seconds: 1690.2, steps: 1750, params: 59_868_674 },
      ...scored(0.908),
    },
    {
      key: "first-4",
      label: "First 4 layers",
      question: "Does it matter which end of the stack the layers come from?",
      priority: 3,
      kind: "mask",
      layers: [0, 1, 2, 3],
      n_layers: 4,
      seed: null,
      train: { seconds: 1370.1, steps: 1750, params: 52_780_802 },
      ...scored(0.9062),
    },
    {
      key: "first-5",
      label: "First 5 layers",
      question: "Does it matter which end of the stack the layers come from?",
      priority: 3,
      kind: "mask",
      layers: [0, 1, 2, 3, 4],
      n_layers: 5,
      seed: null,
      train: { seconds: 1685.4, steps: 1750, params: 59_868_674 },
      ...scored(0.9114),
    },
    {
      key: "last-4",
      label: "Last 4 layers",
      question: "Does it matter which end of the stack the layers come from?",
      priority: 3,
      kind: "mask",
      layers: [8, 9, 10, 11],
      n_layers: 4,
      seed: null,
      train: { seconds: 1372.9, steps: 1750, params: 52_780_802 },
      ...scored(0.8799),
    },
    {
      key: "distilbert",
      label: "DistilBERT",
      question: "How does distillation compare with search at a similar size?",
      priority: 4,
      kind: "distilbert",
      layers: null,
      n_layers: null,
      seed: null,
      train: { seconds: 1994.81, steps: 1750, params: 66_955_010 },
      ...scored(0.9295),
    },
    {
      key: "random-4-seed0",
      label: "Random 4 layers, seed 0",
      question: "Is the found mask in the tail of the distribution, or the middle?",
      priority: 5,
      kind: "mask",
      layers: [0, 4, 6, 11],
      n_layers: 4,
      seed: 0,
      train: { seconds: 1379.4, steps: 1750, params: 52_780_802 },
      ...scored(0.894),
    },
    {
      key: "random-4-seed1",
      label: "Random 4 layers, seed 1",
      question: "Is the found mask in the tail of the distribution, or the middle?",
      priority: 5,
      kind: "mask",
      layers: [1, 2, 4, 9],
      n_layers: 4,
      seed: 1,
      train: { seconds: 1381.0, steps: 1750, params: 52_780_802 },
      ...scored(0.9065),
    },
    {
      key: "random-4-seed2",
      label: "Random 4 layers, seed 2",
      question: "Is the found mask in the tail of the distribution, or the middle?",
      priority: 5,
      kind: "mask",
      layers: [0, 1, 5, 10],
      n_layers: 4,
      seed: 2,
      train: { seconds: 1380.2, steps: 1750, params: 52_780_802 },
      ...scored(0.9039),
    },
  ],
};

// --- the explorer's ablation --------------------------------------------------------------------

/**
 * An amputation of the bottom four layers: what `mask(finetuned)` does to a model that reaches
 * 93% at full depth.
 *
 * The collapse is the point of the page, so the fixture carries a real collapse rather than a
 * gentle dip - a stub scoring 0.90 would let a page through that had quietly swapped the two
 * columns and still looked plausible.
 */
export const ABLATION_RESULT = {
  mask: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  layers: [0, 1, 2, 3],
  n_layers: 4,
  params: 52_780_802,
  matches_known_architecture: null,
  baseline: "bert-imdb",
  baseline_label: "BERT-base",
  rows_scanned: 2000,
  ablated: { correct: 1243, accuracy: 0.6215, wilson_low: 0.6, wilson_high: 0.6427 },
  full: { correct: 1865, accuracy: 0.9325, wilson_low: 0.9206, wilson_high: 0.9428 },
  agreement_with_full: 0.6335,
  protocol:
    "Both figures are the same 2000 reviews from the evaluation sample, scored in one pass. This is not the benchmark, which is the full 15 000-row test split.",
  note: "The layers were removed from an already fine-tuned model and nothing was retrained. The searched architectures were trained after their layers were chosen, which is the expensive half of NAS and the half this measurement leaves out.",
};

const ABLATION_STAGES = ["scoring the amputated model", "scoring the full model"];

export const ABLATION_STARTED: JobSnapshot = {
  id: "ablation-1",
  kind: "ablation",
  status: "running",
  stages: ABLATION_STAGES,
  stage: ABLATION_STAGES[0]!,
  stage_index: 0,
  processed: 0,
  total: 2000,
  messages: [],
  elapsed_s: 0,
  result: null,
  error: null,
};

export const ABLATION_FRAMES: JobSnapshot[] = [
  { ...ABLATION_STARTED, processed: 900, elapsed_s: 3.1 },
  {
    ...ABLATION_STARTED,
    stage: ABLATION_STAGES[1]!,
    stage_index: 1,
    processed: 1200,
    elapsed_s: 9.4,
    messages: ["4 of 12 layers, no retraining: accuracy 0.6215"],
  },
  {
    ...ABLATION_STARTED,
    status: "done",
    stage: null,
    stage_index: 2,
    processed: 2000,
    elapsed_s: 15.2,
    messages: [
      "4 of 12 layers, no retraining: accuracy 0.6215",
      "the same model at full depth: accuracy 0.9325",
    ],
    result: ABLATION_RESULT as unknown as Record<string, unknown>,
  },
];
