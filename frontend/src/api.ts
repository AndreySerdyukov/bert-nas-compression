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
