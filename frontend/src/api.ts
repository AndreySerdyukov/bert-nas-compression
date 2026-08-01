/**
 * Typed client for the backend.
 *
 * The types mirror what `backend/training/extract_notebook_data.py` writes and what
 * `app/api/content.py` serves. They are hand-written rather than generated: there are four
 * documents, they change when the notebooks do (which is to say almost never), and a generator
 * would be a build step earning less than it costs.
 */

const BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* not a JSON error body; the status line is what we have */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

// --- architectures ---------------------------------------------------------------------------

export interface ShippedArchitecture {
  /** null for AdaBERT, which is not a layer mask at all. */
  mask: number[] | null;
  layers: number[] | null;
  n_layers: number | null;
  params: number;
  finetune_epochs: number | null;
}

export interface MethodArchitecture {
  label: string;
  hf_repo: string;
  eval_notebook: string;
  selection_notebook: string;
  shipped: ShippedArchitecture;
  search_best: { mask: number[]; layers: number[] } | null;
  /** False for Random Search alone: the published model is not the mask the search reported. */
  shipped_matches_search: boolean | null;
  note?: string;
}

export interface Architectures {
  schema: number;
  baseline: {
    label: string;
    hf_repo: string;
    n_layers: number;
    params: number;
    eval_notebook: string;
  };
  params_formula: { base: number; per_layer: number };
  methods: Record<string, MethodArchitecture>;
}

export const fetchArchitectures = () => getJson<Architectures>("/api/architectures");

// --- search trajectories ---------------------------------------------------------------------

export interface Candidate {
  mask: number[];
  layers: number[];
  n_layers: number;
  accuracy: number;
  params: number;
  /** null when the candidate was rejected outright, so it has no comparable score. */
  fitness: number | null;
  trial?: number;
  generation?: number;
  phase?: "initial" | "bayesian";
  index?: number;
  rejected?: boolean;
  rejection_reason?: string | null;
  final_reevaluation?: boolean;
  train_seconds?: number;
}

export interface SurrogateProposal {
  round: number;
  mask: number[];
  n_layers: number;
  predicted_fitness: number;
}

export interface MethodTrajectory {
  notebook: string;
  budget: {
    train_rows: number;
    val_rows: number;
    max_length: number;
    epochs_per_candidate: number;
    fitness: string;
    fitness_params: Record<string, number>;
    surrogate?: string;
  };
  candidates: Candidate[];
  proposals?: SurrogateProposal[];
  stages: { name: string; index: number }[];
}

export interface Trajectories {
  schema: number;
  params_formula: { base: number; per_layer: number };
  methods: Record<string, MethodTrajectory>;
}

export const fetchTrajectories = () => getJson<Trajectories>("/api/trajectories");

// --- the four disagreeing source tables --------------------------------------------------------

export interface ReportedModel {
  key: string;
  label: string;
  hf_repo: string;
  n_layers: number | null;
  params: number;
  accuracy: Record<string, number>;
  ms_per_example?: Record<string, number>;
  notes?: Record<string, string>;
}

export interface ReportedResults {
  schema: number;
  note: string;
  sources: Record<
    string,
    { label: string; kind: string; path?: string; canonical?: boolean; protocol: string }
  >;
  models: ReportedModel[];
  latency_caveat: string;
}

export const fetchReportedResults = () => getJson<ReportedResults>("/api/reported-results");

// --- what this project measured itself ----------------------------------------------------------

/** Counts, not rates: every derived figure on the benchmark page is computed from these four. */
export interface Confusion {
  true_negative: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
}

/** The scores every model and every control carries, so both tables read the same shape. */
export interface Scored {
  correct: number;
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  positive_precision: number;
  positive_recall: number;
  positive_f1: number;
  confusion: Confusion;
}

export interface BenchmarkModel extends Scored {
  name: string;
  label: string;
  /** null on the baseline, which was not produced by a search. */
  method: string | null;
  n_layers: number | null;
  max_length: number;
  hf_repo: string;
  hf_revision: string;
  params: number;
  params_formula: number | null;
  params_agree: boolean | null;
  flops_per_example: number | null;
  bytes_on_disk: number | null;
  bytes_fp32: number | null;
  score_seconds: number;
  latency: LatencyStats | null;
  throughput: ThroughputStats | null;
  /** Weighed one model per process; the baseline below is the floor none of them owns. */
  resident_bytes: number | null;
  runtime_baseline_bytes: number | null;
}

export interface BenchmarkProtocol {
  test_rows: number;
  test_index_sha256: string;
  corpus_sha256: string;
  accuracy_device: string;
  latency_device: string;
  latency_threads: number;
  latency_warmup: number;
  latency_repeats: number;
  throughput_batch: number;
  score_batch: number;
  machine: string;
  python: string;
  torch: string;
  transformers: string;
  memory_measured: string;
}

export interface Benchmark {
  schema: number;
  generated_by: string;
  note: string;
  protocol: BenchmarkProtocol;
  models: BenchmarkModel[];
}

export const fetchBenchmark = () => getJson<Benchmark>("/api/benchmark");

// --- the controls the searched architectures are compared against -------------------------------

export interface ControlProtocol {
  epochs: number;
  train_rows: number;
  test_rows: number;
  /** The same split the benchmark scored on; a control on another split compares to nothing. */
  test_index_sha256: string;
  max_length: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  warmup_ratio: number;
  base_model: string;
  device: string;
  cooldown_seconds: number;
  torch: string;
  source: string;
}

/** What one control cost. `steps` is null for TF-IDF, which is not trained in steps at all. */
export interface ControlCost {
  seconds: number;
  steps: number | null;
  steps_per_epoch?: number;
  seconds_per_step?: number;
  train_loss?: number;
  params?: number;
  features?: number;
  note?: string;
}

export interface ControlResult extends Scored {
  key: string;
  label: string;
  /** Every control carries the question it exists to answer; a bare number is unusable. */
  question: string;
  priority: number;
  kind: "tfidf" | "mask" | "distilbert";
  layers: number[] | null;
  n_layers: number | null;
  seed: number | null;
  train: ControlCost;
}

export interface Controls {
  schema: number;
  generated_by: string;
  note: string;
  protocol: ControlProtocol;
  /** Printed beside the results: six of eighteen reads as "the controls" unless the plan is shown. */
  planned: string[];
  not_run: string[];
  weights_saved: boolean;
  weights_note: string;
  controls: ControlResult[];
}

export const fetchControls = () => getJson<Controls>("/api/controls");

// --- serving: the catalog, the reviews, and scoring ---------------------------------------------

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      /* not a JSON error body; the status line is what we have */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

/**
 * The machine behind a timing. Never rendered apart from the number it belongs to: a millisecond
 * figure without its thread count and device is not comparable to anything.
 */
export interface RuntimeInfo {
  device: string;
  threads: number;
  interop_threads: number;
  machine: string;
  torch_version: string;
}

export interface ServedModel {
  name: string;
  label: string;
  description: string;
  method: string | null;
  params: number | null;
  /** null for AdaBERT, which has no encoder layers at all. */
  n_layers: number | null;
  /** 128 for AdaBERT, 512 for the rest. Part of the model, not a serving preference. */
  max_length: number;
  hf_repo: string;
  hf_revision: string;
  positive_index: number;
  is_baseline: boolean;
}

/** A model with a manifest that is not being served, and what to do about it. */
export interface SkippedModel {
  name: string;
  label: string;
  reason: string;
  remedy: string | null;
}

export interface ModelsResponse {
  models: ServedModel[];
  skipped: SkippedModel[];
  baseline: string;
  runtime: RuntimeInfo;
}

/** Single-example latency: what one reader waits for one review. */
export interface LatencyStats {
  batch_size: number;
  warmup: number;
  repeats: number;
  median_ms: number;
  p95_ms: number;
  n_tokens: number;
}

/** Batched throughput. `per_example_ms` is not a latency and is never labelled as one. */
export interface ThroughputStats {
  batch_size: number;
  warmup: number;
  repeats: number;
  median_batch_ms: number;
  per_example_ms: number;
  examples_per_second: number;
  n_tokens: number;
}

export interface ModelPrediction {
  name: string;
  label: string;
  verdict: string;
  positive_probability: number;
  probabilities: Record<string, number>;
  n_tokens: number;
  params: number | null;
  n_layers: number | null;
  /** One scoring, warm-up included. Honest about being a single sample. */
  elapsed_ms: number;
  latency: LatencyStats | null;
  throughput: ThroughputStats | null;
  /** null on the baseline itself, and when the baseline was not among the compared models. */
  agrees_with_baseline: boolean | null;
}

export interface CompareResponse {
  baseline: string;
  results: ModelPrediction[];
  disagree: string[];
  runtime: RuntimeInfo;
}

export interface Example {
  id: number;
  text: string;
  /** The corpus label: 1 positive, 0 negative. */
  label: number;
  n_chars: number;
  /** True when the review runs past the 512-token window. */
  truncated: boolean;
}

export interface ExamplesResponse {
  examples: Example[];
  total: number;
  source: string;
}

export const fetchModels = () => getJson<ModelsResponse>("/api/models");
export const fetchExamples = () => getJson<ExamplesResponse>("/api/examples");

export const compare = (body: {
  text: string;
  models?: string[];
  measure_latency?: boolean;
  measure_throughput?: boolean;
}) => postJson<CompareResponse>("/api/compare", body);

// --- long-running work ---------------------------------------------------------------------------

export type JobStatus = "running" | "done" | "failed";

export interface JobSnapshot {
  id: string;
  kind: string;
  status: JobStatus;
  /** Named stages, fixed before the job starts, so the whole bar can be drawn up front. */
  stages: string[];
  stage: string | null;
  stage_index: number;
  processed: number;
  total: number;
  messages: string[];
  elapsed_s: number;
  /** Shape depends on `kind`; use `scanResult` to narrow it. Null until the job finishes. */
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface ScanModel {
  name: string;
  label: string;
  params: number | null;
  n_layers: number | null;
  /** Against the corpus label, over the scanned rows. Not the benchmark. */
  correct: number;
  accuracy: number;
  /** Against the baseline's verdicts. Null on the baseline itself. */
  agree_with_baseline: number | null;
  agreement: number | null;
}

export interface ScanRow {
  id: number;
  /** The corpus label as a verdict word, so it compares directly with what the models said. */
  label: string;
  n_chars: number;
  excerpt: string;
  truncated: boolean;
  verdicts: Record<string, string>;
}

export interface ScanResult {
  baseline: string | null;
  rows_scanned: number;
  models: ScanModel[];
  disagreements_found: number;
  /** Fewer than `disagreements_found` when the cap bit. Both are shown, never just this one. */
  disagreements_shown: number;
  disagreements: ScanRow[];
  /** The scan drops the thread pin, because it counts disagreements rather than timing anything. */
  threads_used: number;
  machine: string;
}

/**
 * Narrow a finished job's result. The cast is real and is why this lives in one place: the server
 * types `result` by `kind`, and TypeScript cannot see that relationship across the wire.
 */
export function scanResult(job: JobSnapshot | null): ScanResult | null {
  if (job === null || job.kind !== "disagreement" || job.result === null) return null;
  return job.result as unknown as ScanResult;
}

/** One accuracy with the interval that says how much of it is evidence. */
export interface ScoredRun {
  correct: number;
  accuracy: number;
  wilson_low: number;
  wilson_high: number;
}

export interface AblationResult {
  mask: number[];
  layers: number[];
  n_layers: number;
  params: number;
  /** The method that shipped this exact mask, if one did. Null for anything the user invents. */
  matches_known_architecture: string | null;
  baseline: string;
  baseline_label: string;
  rows_scanned: number;
  /** The mask applied to a fine-tuned model, with no retraining. */
  ablated: ScoredRun;
  /** The same model at full depth, over the same rows in the same pass. */
  full: ScoredRun;
  agreement_with_full: number;
  protocol: string;
  note: string;
}

/** Narrow a finished ablation, the same way `scanResult` narrows a scan. */
export function ablationResult(job: JobSnapshot | null): AblationResult | null {
  if (job === null || job.kind !== "ablation" || job.result === null) return null;
  return job.result as unknown as AblationResult;
}

export const startJob = (body: {
  kind: "disagreement" | "selftest" | "ablation";
  limit?: number;
  /** `ablation` only: which encoder layers survive. */
  layers?: number[];
}) => postJson<JobSnapshot>("/api/jobs", body);

export const fetchJob = (id: string) => getJson<JobSnapshot>(`/api/jobs/${id}`);

export const jobEventsUrl = (id: string) => `${BASE}/api/jobs/${id}/events`;
