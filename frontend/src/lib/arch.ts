/**
 * The arithmetic of a layer mask, in the browser.
 *
 * This is the TypeScript twin of `backend/app/services/architecture.py`, and it exists so the
 * explorer answers instantly: toggling a layer changes the parameter count, the FLOPs and the
 * footprint with no request in flight. That is possible because the cost of an architecture depends
 * only on which encoder layers it keeps - never on its weights - which is exactly the asymmetry the
 * methodology section is about. Quality is the other half, and that one costs a GPU.
 *
 * The two implementations are kept honest by pinning the same seven-value parameter ladder: in
 * `backend/tests/test_architecture.py` for Python, and through the explorer's browser test for
 * this one. If they ever disagree, a shipped mask stops matching its published parameter count.
 */

/** BERT-base geometry. None of it was ever searched - only which layers to keep. */
export const N_LAYERS = 12;
export const HIDDEN_SIZE = 768;
export const INTERMEDIATE_SIZE = 3072;
export const MAX_SEQ_LEN = 512;

/**
 * Derived from a real checkpoint and pinned. A full model reports 109 483 778 parameters; each
 * encoder layer is 7 087 872 of them, and what is left - embeddings, pooler, classifier - does not
 * move when the mask does.
 */
export const PARAMS_PER_LAYER = 7_087_872;
export const NON_LAYER_PARAMS = 24_429_314;

const BYTES_PER_FP32 = 4;

export type LayerMask = readonly number[];

export interface ArchitectureCost {
  layers: number[];
  nLayers: number;
  params: number;
  /** Fraction of the 12-layer baseline, 0 to 1. */
  paramsPctOfBase: number;
  flopsPerExample: number;
  bytesFp32: number;
}

export class InvalidMaskError extends Error {}

export function validateMask(mask: LayerMask): number[] {
  if (mask.length !== N_LAYERS) {
    throw new InvalidMaskError(`mask must have ${N_LAYERS} entries, got ${mask.length}`);
  }
  return mask.map((bit) => {
    if (bit !== 0 && bit !== 1)
      throw new InvalidMaskError(`mask entries must be 0 or 1, got ${bit}`);
    return bit;
  });
}

export function layersFromMask(mask: LayerMask): number[] {
  const layers: number[] = [];
  mask.forEach((bit, index) => {
    if (bit) layers.push(index);
  });
  return layers;
}

export function maskFromLayers(layers: readonly number[]): number[] {
  const mask = new Array<number>(N_LAYERS).fill(0);
  for (const index of layers) {
    if (!Number.isInteger(index) || index < 0 || index >= N_LAYERS) {
      throw new InvalidMaskError(`layer index out of range: ${index}`);
    }
    mask[index] = 1;
  }
  return mask;
}

/** Parameter count of a classifier keeping `nLayers` encoder layers. */
export function paramsFor(nLayers: number): number {
  return NON_LAYER_PARAMS + PARAMS_PER_LAYER * nLayers;
}

/**
 * Multiply-accumulate FLOPs for one encoder layer at `seqLen` tokens.
 *
 * Matrix multiplications only - four attention projections, two attention batched matmuls, two
 * feed-forward projections - at two FLOPs per multiply-accumulate. Softmax, LayerNorm and GELU are
 * elementwise and add a percent or so; omitting them is the usual convention.
 */
export function flopsPerLayer(seqLen: number): number {
  const attentionProjections = 4 * seqLen * HIDDEN_SIZE * HIDDEN_SIZE;
  const attentionScores = 2 * seqLen * seqLen * HIDDEN_SIZE;
  const feedForward = 2 * seqLen * HIDDEN_SIZE * INTERMEDIATE_SIZE;
  return 2 * (attentionProjections + attentionScores + feedForward);
}

export function flopsFor(nLayers: number, seqLen: number): number {
  if (seqLen < 1 || seqLen > MAX_SEQ_LEN) {
    throw new InvalidMaskError(`seqLen must be between 1 and ${MAX_SEQ_LEN}`);
  }
  const head = 2 * (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * 2);
  return nLayers * flopsPerLayer(seqLen) + head;
}

/** Full cost breakdown. Pure arithmetic: no fetch, no weights. */
export function describe(mask: LayerMask, seqLen = 256): ArchitectureCost {
  const validated = validateMask(mask);
  const layers = layersFromMask(validated);
  const params = paramsFor(layers.length);
  return {
    layers,
    nLayers: layers.length,
    params,
    paramsPctOfBase: params / paramsFor(N_LAYERS),
    flopsPerExample: flopsFor(layers.length, seqLen),
    bytesFp32: params * BYTES_PER_FP32,
  };
}

/**
 * The name of the method that shipped this exact mask, or null.
 *
 * `known` comes from `/api/architectures`, fetched once when the page loads - so this stays a pure
 * function and toggling a layer still costs no request. Note what it will not match: the mask
 * Random Search's write-up names. The model on the Hub is a different one, and this answers for
 * the model, because that is what an ablation gets compared against.
 */
export function matchesKnown(
  mask: LayerMask,
  known: Record<string, LayerMask | null>,
): string | null {
  const key = validateMask(mask).join("");
  for (const [name, candidate] of Object.entries(known)) {
    if (candidate && validateMask(candidate).join("") === key) return name;
  }
  return null;
}

/** How many masks keep between `minLayers` and `maxLayers` layers. For 4-12 that is 3 797. */
export function searchSpaceSize(minLayers: number, maxLayers: number): number {
  const choose = (n: number, k: number): number => {
    if (k < 0 || k > n) return 0;
    let result = 1;
    for (let i = 0; i < k; i += 1) result = (result * (n - i)) / (i + 1);
    return Math.round(result);
  };
  let total = 0;
  for (let k = minLayers; k <= maxLayers; k += 1) total += choose(N_LAYERS, k);
  return total;
}

/** Presets the explorer offers, so "what a simple rule would pick" is one click away. */
export const PRESETS: ReadonlyArray<{ label: string; layers: number[] }> = [
  { label: "Full (12)", layers: [...Array(12).keys()] },
  { label: "Random Search", layers: [0, 1, 5, 7, 9] },
  { label: "AlphaNAS", layers: [1, 2, 6, 11] },
  { label: "BANANAS", layers: [0, 1, 6, 9] },
  { label: "First 4", layers: [0, 1, 2, 3] },
  { label: "Last 4", layers: [8, 9, 10, 11] },
  { label: "Every third", layers: [0, 3, 6, 9] },
];
