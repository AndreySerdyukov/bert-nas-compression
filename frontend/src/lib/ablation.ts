import type { Architectures, ControlResult } from "../api";

/**
 * Finding the trained counterpart of a mask, so an ablation is never shown on its own.
 *
 * An amputated model scoring 0.62 means nothing by itself - it looks like a broken model rather
 * than like a measurement. It means something beside two other numbers: the same mask trained
 * properly, and the same weights at full depth. The third column comes from the job; this is what
 * finds the second.
 *
 * Not every mask has one. There are 4 096 masks and eighteen controls plus four searched
 * architectures, so most of what a reader clicks has no trained twin at all - and saying so is
 * better than quietly comparing against the nearest thing of a different depth.
 */

export interface TrainedTwin {
  label: string;
  accuracy: number;
  /** Where the figure came from, since it is not from the same rows as the ablation. */
  source: "control" | "searched";
  rows: number;
}

const sameLayers = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((value, index) => value === b[index]);

export function trainedTwin(
  layers: readonly number[],
  controls: readonly ControlResult[],
  architectures: Architectures | null,
): TrainedTwin | null {
  const control = controls.find(
    (candidate) => candidate.layers !== null && sameLayers(candidate.layers, layers),
  );
  if (control) {
    return { label: control.label, accuracy: control.accuracy, source: "control", rows: 15_000 };
  }

  for (const method of Object.values(architectures?.methods ?? {})) {
    const shipped = method.shipped.layers;
    if (shipped !== null && sameLayers(shipped, layers)) {
      // The accuracy lives in benchmark.json rather than here, so the caller fills it in; this
      // only establishes that the mask is one somebody shipped.
      return { label: method.label, accuracy: Number.NaN, source: "searched", rows: 15_000 };
    }
  }
  return null;
}

/** Presets worth one click, chosen because each one answers a different question. */
export const PRESETS: { label: string; layers: number[]; why: string }[] = [
  { label: "First 4", layers: [0, 1, 2, 3], why: "the bottom of the stack, the best naive rule" },
  { label: "Last 4", layers: [8, 9, 10, 11], why: "the top, which trained is 2.6 points worse" },
  // Named for what it is rather than for "every third": the rule reaches both ends of the stack,
  // so at four layers the gaps are 4, 3 and 4 rather than a uniform 3. This is the mask the
  // `uniform-4` control was trained on.
  { label: "Evenly spaced", layers: [0, 4, 7, 11], why: "the naive rule, and a trained control" },
  { label: "AlphaNAS", layers: [1, 2, 6, 11], why: "what one search chose" },
  { label: "BANANAS", layers: [0, 1, 6, 9], why: "what another chose" },
  { label: "Random Search", layers: [0, 1, 5, 7, 9], why: "the mask actually shipped" },
];
