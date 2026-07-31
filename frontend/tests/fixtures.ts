import type { Architectures, ReportedResults, Trajectories } from "../src/api";

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
