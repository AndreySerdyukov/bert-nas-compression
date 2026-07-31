import type {
  Architectures,
  CompareResponse,
  ExamplesResponse,
  ModelsResponse,
  ReportedResults,
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
