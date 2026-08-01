/**
 * Scales and ticks for the hand-rolled charts.
 *
 * There is no chart library anywhere in this portfolio, which is a deliberate choice and also the
 * reason this file exists: nine charts each inventing their own padding and tick spacing is exactly
 * what makes hand-rolled charts look hand-rolled. One grammar, shared.
 */

export interface Scale {
  /** Data value to pixel position inside the plot area. */
  (value: number): number;
  domain: readonly [number, number];
  range: readonly [number, number];
  ticks(count?: number): number[];
}

function makeScale(
  domain: readonly [number, number],
  range: readonly [number, number],
  forward: (value: number) => number,
  tickFn: (count: number) => number[],
): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const f0 = forward(d0);
  const span = forward(d1) - f0 || 1;
  const scale = ((value: number) => r0 + ((forward(value) - f0) / span) * (r1 - r0)) as Scale;
  scale.domain = domain;
  scale.range = range;
  scale.ticks = (count = 5) => tickFn(count);
  return scale;
}

export function linear(domain: readonly [number, number], range: readonly [number, number]): Scale {
  return makeScale(
    domain,
    range,
    (value) => value,
    (count) => niceTicks(domain[0], domain[1], count),
  );
}

/**
 * Log scale. Used for parameter counts, where the models span 7.8M to 109.5M - a factor of 14 that
 * a linear axis crushes into the left third of the plot.
 */
export function log(domain: readonly [number, number], range: readonly [number, number]): Scale {
  if (domain[0] <= 0) throw new Error("a log scale needs a positive domain");
  return makeScale(domain, range, Math.log, (count) => logTicks(domain[0], domain[1], count));
}

/** Round tick values covering [min, max] - the 1/2/5 sequence, so labels stay readable. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const rawStep = (max - min) / Math.max(count, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  // Rounded because floating-point accumulation turns 0.30000000000000004 into a visible label.
  for (let value = first; value <= max + step / 1e6; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

/** Powers of ten inside the domain, halved if that would give fewer than two ticks. */
export function logTicks(min: number, max: number, count = 4): number[] {
  const ticks: number[] = [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  for (let exponent = lo; exponent <= hi; exponent += 1) {
    for (const multiple of [1, 2, 5]) {
      const value = multiple * 10 ** exponent;
      if (value >= min && value <= max) ticks.push(value);
    }
  }
  if (ticks.length <= count) return ticks;
  const stride = Math.ceil(ticks.length / count);
  return ticks.filter((_, index) => index % stride === 0);
}

/** Pad a domain by a fraction of its span, so points never sit on the frame. */
export function padDomain(min: number, max: number, fraction = 0.06): readonly [number, number] {
  const span = max - min || Math.abs(max) || 1;
  return [min - span * fraction, max + span * fraction];
}

// --- formatting ------------------------------------------------------------------------------

/** 109 483 778 -> "109.5M". Parameter counts are compared, not read digit by digit. */
export function formatParams(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}k`;
  return String(value);
}

/** 0.9027 -> "90.3%". One decimal is right for an axis tick, where a label has to stay readable. */
export function formatAccuracy(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * 0.9027 -> "90.27%". For tables and any figure that gets compared to another one.
 *
 * The controls made this necessary: AlphaNAS scores 0.9027 and BANANAS 0.9031, and at one decimal
 * both render "90.3%" - a table that cannot separate two models it is putting on adjacent rows.
 * The extra digit is not false precision either way, since the same 15 000 rows produced both.
 */
export function formatAccuracyPrecise(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatFlops(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)} TFLOP`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GFLOP`;
  return `${(value / 1e6).toFixed(0)} MFLOP`;
}

export function formatBytes(value: number): string {
  if (value >= 1 << 30) return `${(value / (1 << 30)).toFixed(2)} GB`;
  return `${(value / (1 << 20)).toFixed(0)} MB`;
}
