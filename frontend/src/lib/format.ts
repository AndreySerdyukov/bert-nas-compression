/**
 * Number formatting shared by the serving pages.
 *
 * Small on purpose. Every function here exists because the same figure is rendered in several
 * places and had to read identically in all of them - a parameter count that is "109.5M" on a card
 * and "109,483,778" in a table two lines below invites the reader to wonder whether they are the
 * same number.
 */

/** 109_483_778 -> "109.5M". Parameter counts, where the leading digits are what matter. */
export function formatParams(params: number | null): string {
  if (params === null) return "unknown";
  if (params >= 1_000_000) return `${(params / 1_000_000).toFixed(1)}M`;
  if (params >= 1_000) return `${(params / 1_000).toFixed(1)}k`;
  return String(params);
}

/**
 * Milliseconds, with the precision the magnitude deserves. AdaBERT answers in under a millisecond
 * and BERT-base in tens, so one fixed number of decimals is wrong for one of them.
 */
export function formatMs(ms: number): string {
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

/** 0.9973 -> "99.7%". */
export function formatPercent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/**
 * The machine behind a timing, as one line.
 *
 * Rendered wherever a latency is, and that is a rule rather than a nicety: these figures are the
 * product, and a figure whose thread count and device are missing cannot be compared with anything.
 */
export function describeRuntime(runtime: {
  device: string;
  threads: number;
  machine: string;
  torch_version: string;
}): string {
  const threads = runtime.threads === 1 ? "1 thread" : `${runtime.threads} threads`;
  return `${threads}, ${runtime.device}, ${runtime.machine}, torch ${runtime.torch_version}`;
}
