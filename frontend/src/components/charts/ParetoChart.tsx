import Frame from "./Frame";
import { formatAccuracy, formatParams, linear, log, padDomain } from "./scale";

/**
 * Accuracy against parameter count - the trade the whole project is about, on one pair of axes.
 *
 * The x axis is logarithmic because the models span 7.8M to 109.5M, a factor of fourteen that a
 * linear axis squashes into the left third of the plot and makes the four NAS models look
 * identical. They nearly are, which is itself a finding, but it should be legible rather than an
 * artefact of the scale.
 *
 * Two mark styles carry the distinction that matters most here: **filled** points are models that
 * were trained on their architecture, **hollow** ones are ablations - a mask applied to a model
 * fine-tuned with all twelve layers, never retrained. They belong on the same axes precisely
 * because the gap between them is what a training budget buys, but they must never be mistaken for
 * each other.
 */

export interface ParetoPoint {
  label: string;
  params: number;
  accuracy: number;
  /** Hollow: an ablation, not a trained model. */
  ablation?: boolean;
  /** Drawn larger and labelled: the reference the budget band is measured from. */
  baseline?: boolean;
}

export interface ParetoChartProps {
  points: ParetoPoint[];
  /** Accuracy drop, in points, that the project set as its budget. Drawn as a band. */
  budgetPp?: number;
  ariaLabel: string;
  /** Says where the accuracies came from, since this chart is used for both sources. */
  caption?: string;
}

const WIDTH = 640;
const HEIGHT = 300;
const PADDING = { top: 16, right: 24, bottom: 34, left: 54 };

export default function ParetoChart({ points, budgetPp, ariaLabel, caption }: ParetoChartProps) {
  if (points.length === 0) return null;

  const paramValues = points.map((point) => point.params);
  const accuracies = points.map((point) => point.accuracy);
  const x = log(
    [Math.min(...paramValues) * 0.8, Math.max(...paramValues) * 1.25],
    [PADDING.left, WIDTH - PADDING.right],
  );
  const y = linear(padDomain(Math.min(...accuracies), Math.max(...accuracies), 0.1), [
    HEIGHT - PADDING.bottom,
    PADDING.top,
  ]);

  const baseline = points.find((point) => point.baseline);

  // The frontier over trained models only: an ablation is not something anyone would ship.
  const trained = points.filter((point) => !point.ablation).sort((a, b) => a.params - b.params);
  const frontier: ParetoPoint[] = [];
  let bestSoFar = -Infinity;
  for (const point of trained) {
    if (point.accuracy > bestSoFar) {
      frontier.push(point);
      bestSoFar = point.accuracy;
    }
  }
  const frontierPath = frontier
    .flatMap((point, index) => {
      const previous = frontier[index - 1];
      const step = previous ? `L${x(point.params)},${y(previous.accuracy)}` : "";
      return `${index === 0 ? "M" : step + "L"}${x(point.params)},${y(point.accuracy)}`;
    })
    .join(" ");

  return (
    <figure className="m-0">
      <Frame
        width={WIDTH}
        height={HEIGHT}
        padding={PADDING}
        x={x}
        y={y}
        xLabel="parameters (log scale)"
        yLabel="accuracy"
        formatX={formatParams}
        formatY={formatAccuracy}
        yTicks={y.ticks(4)}
        ariaLabel={ariaLabel}
        data={{
          columns: ["model", "params", "accuracy", "kind"],
          rows: points.map((point) => [
            point.label,
            formatParams(point.params),
            formatAccuracy(point.accuracy),
            point.ablation ? "ablation, not retrained" : point.baseline ? "baseline" : "trained",
          ]),
        }}
      >
        {baseline && budgetPp !== undefined && (
          <>
            {/* Everything inside the band is within the accuracy budget the project set itself. */}
            <rect
              x={PADDING.left}
              y={y(baseline.accuracy)}
              width={WIDTH - PADDING.left - PADDING.right}
              height={Math.max(0, y(baseline.accuracy - budgetPp / 100) - y(baseline.accuracy))}
              fill="rgb(var(--accent) / 0.07)"
            />
            <line
              x1={PADDING.left}
              x2={WIDTH - PADDING.right}
              y1={y(baseline.accuracy - budgetPp / 100)}
              y2={y(baseline.accuracy - budgetPp / 100)}
              stroke="rgb(var(--accent) / 0.5)"
              strokeWidth="1"
              strokeDasharray="4 3"
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}

        {frontier.length > 1 && (
          <path
            d={frontierPath}
            fill="none"
            stroke="rgb(var(--slate))"
            strokeWidth="1"
            strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {points.map((point) => (
          <g key={`${point.label}-${point.params}-${point.accuracy}`}>
            <circle
              cx={x(point.params)}
              cy={y(point.accuracy)}
              r={point.baseline ? 5.5 : 4.5}
              fill={point.ablation ? "rgb(var(--canvas))" : "rgb(var(--accent))"}
              stroke="rgb(var(--accent))"
              strokeWidth="1.4"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={x(point.params)}
              y={y(point.accuracy) - 10}
              textAnchor="middle"
              fill="rgb(var(--slate))"
              fontSize="10"
            >
              {point.label}
            </text>
          </g>
        ))}
      </Frame>
      {caption && <figcaption className="mt-2 text-[12px] text-slate">{caption}</figcaption>}
    </figure>
  );
}
