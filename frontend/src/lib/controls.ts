import type { BenchmarkModel, ControlResult } from "../api";

/**
 * The comparison the controls were run to make, as data rather than as prose on a page.
 *
 * The question this project is asked - did searching buy anything - has no single answer, and the
 * shape here reflects that. It has an answer *per depth*, because a four-layer mask can only be
 * judged against four-layer controls, and the two depths disagree: at four layers the searched
 * architectures fall below the median of five random masks, at five layers one clears it. A helper
 * that returned one verdict would have to pick which depth to believe.
 */

/** Median of a list. Returns null for an empty one rather than NaN, which renders as "NaN%". */
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[middle]!;
  return (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/**
 * Share of `values` strictly below `value`, as a percentage.
 *
 * Strictly below, so a model that ties every random mask reads as the 0th percentile rather than
 * the 100th. With five seeds the resolution is 20 points and the figure should be read as
 * "roughly where in the cloud", which is why the page prints the five accuracies beside it.
 */
export function percentileAmong(value: number, values: number[]): number | null {
  if (values.length === 0) return null;
  return (values.filter((other) => other < value).length / values.length) * 100;
}

export interface DepthVerdict {
  nLayers: number;
  /** Searched architectures at this depth, best first. */
  searched: { label: string; accuracy: number }[];
  /** The naive masks, by the rule that produced them. Null when that control was not run. */
  uniform: ControlResult | null;
  first: ControlResult | null;
  last: ControlResult | null;
  randomAccuracies: number[];
  randomMedian: number | null;
  /** Positive when the search beat the evenly spaced mask, in accuracy points. */
  vsUniformPp: number | null;
  /** Positive when the search beat "the bottom k layers", in accuracy points. */
  vsFirstPp: number | null;
  /** Where the best searched model sits inside the random masks of its own depth. */
  percentile: number | null;
}

const depthOf = (control: ControlResult): number | null => control.n_layers;

/**
 * Group everything by encoder depth and work out, per depth, what the search is worth.
 *
 * AdaBERT is excluded on purpose: it has no encoder layers to compare against a mask, and putting
 * it at a depth would invent one. It appears in the model table, where it belongs.
 */
export function verdictByDepth(
  models: BenchmarkModel[],
  controls: ControlResult[],
): DepthVerdict[] {
  const depths = [...new Set(controls.map(depthOf).filter((n): n is number => n !== null))].sort(
    (a, b) => a - b,
  );

  return depths.map((nLayers) => {
    const at = (prefix: string) =>
      controls.find((control) => control.key === `${prefix}-${nLayers}`) ?? null;

    const randomAccuracies = controls
      .filter((control) => control.key.startsWith(`random-${nLayers}-`))
      .map((control) => control.accuracy);

    const searched = models
      .filter((model) => model.method !== null && model.n_layers === nLayers)
      .map((model) => ({ label: model.label, accuracy: model.accuracy }))
      .sort((a, b) => b.accuracy - a.accuracy);

    const uniform = at("uniform");
    const first = at("first");
    const best = searched[0];
    const points = (a: number, b: number) => (a - b) * 100;

    return {
      nLayers,
      searched,
      uniform,
      first,
      last: at("last"),
      randomAccuracies,
      randomMedian: median(randomAccuracies),
      vsUniformPp: best && uniform ? points(best.accuracy, uniform.accuracy) : null,
      vsFirstPp: best && first ? points(best.accuracy, first.accuracy) : null,
      percentile: best ? percentileAmong(best.accuracy, randomAccuracies) : null,
    };
  });
}

/**
 * The controls that answer a question no mask can: whether this task needed a transformer, and
 * what distillation gets at the same size. Neither has an encoder depth, so neither is in
 * `verdictByDepth`.
 */
export function nonMaskControls(controls: ControlResult[]): ControlResult[] {
  return controls.filter((control) => control.kind !== "mask");
}
