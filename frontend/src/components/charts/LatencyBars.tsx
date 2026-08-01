import Frame from "./Frame";
import { linear } from "./scale";

/**
 * Single-example latency per model: the median as a bar, the p95 as a cap above it.
 *
 * Two marks rather than one because a bare median hides the thing a reader actually feels. The cap
 * is drawn as a line at p95 with a stem down to the median, so the distance between them is the
 * spread and is legible without a legend.
 *
 * What this chart deliberately cannot show is throughput. Dividing a batch time by its batch size
 * produces a smaller number with the same unit, and putting it on these axes would invite exactly
 * the comparison this project exists to correct. Throughput lives in the table, under its own
 * heading, in its own column.
 */

export interface LatencyBar {
  label: string;
  medianMs: number;
  p95Ms: number;
  /** Tokens the model actually saw. Two models timed on one review are not doing equal work. */
  nTokens: number;
}

export interface LatencyBarsProps {
  bars: LatencyBar[];
  device: string;
  threads: number;
  ariaLabel: string;
  caption?: string;
}

const WIDTH = 640;
const HEIGHT = 260;
const PADDING = { top: 16, right: 24, bottom: 44, left: 54 };

export default function LatencyBars({
  bars,
  device,
  threads,
  ariaLabel,
  caption,
}: LatencyBarsProps) {
  if (bars.length === 0) return null;

  const top = Math.max(...bars.map((bar) => bar.p95Ms));
  // From zero, always. A bar chart with a clipped baseline exaggerates every ratio on it, and the
  // ratios are the whole point when one model is meant to be three times faster than another.
  const y = linear([0, top * 1.12], [HEIGHT - PADDING.bottom, PADDING.top]);
  const x = linear([-0.5, bars.length - 0.5], [PADDING.left, WIDTH - PADDING.right]);

  const slot = (WIDTH - PADDING.left - PADDING.right) / bars.length;
  const barWidth = Math.min(46, slot * 0.55);

  return (
    <figure className="m-0">
      <Frame
        width={WIDTH}
        height={HEIGHT}
        padding={PADDING}
        x={x}
        y={y}
        xLabel={`one review, batch 1, on ${device} at ${threads} thread${threads === 1 ? "" : "s"}`}
        yLabel="milliseconds"
        xTicks={bars.map((_, index) => index)}
        formatX={(tick) => bars[tick]?.label ?? ""}
        formatY={(value) => `${value.toFixed(0)}`}
        yTicks={y.ticks(4)}
        ariaLabel={ariaLabel}
        data={{
          columns: ["model", "median ms", "p95 ms", "tokens"],
          rows: bars.map((bar) => [
            bar.label,
            bar.medianMs.toFixed(1),
            bar.p95Ms.toFixed(1),
            bar.nTokens,
          ]),
        }}
      >
        {bars.map((bar, index) => {
          const centre = x(index);
          return (
            <g key={bar.label}>
              <rect
                x={centre - barWidth / 2}
                y={y(bar.medianMs)}
                width={barWidth}
                height={Math.max(0, y(0) - y(bar.medianMs))}
                fill="rgb(var(--accent) / 0.75)"
              />
              {/* The stem first, so the p95 cap sits on top of it rather than beside it. */}
              <line
                x1={centre}
                x2={centre}
                y1={y(bar.medianMs)}
                y2={y(bar.p95Ms)}
                stroke="rgb(var(--slate))"
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={centre - barWidth / 4}
                x2={centre + barWidth / 4}
                y1={y(bar.p95Ms)}
                y2={y(bar.p95Ms)}
                stroke="rgb(var(--slate))"
                strokeWidth="1.4"
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={centre}
                y={y(bar.medianMs) - 8}
                textAnchor="middle"
                fill="rgb(var(--slate))"
                fontSize="10"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {bar.medianMs.toFixed(1)}
              </text>
            </g>
          );
        })}
      </Frame>
      {caption && <figcaption className="mt-2 text-[12px] text-slate">{caption}</figcaption>}
    </figure>
  );
}
