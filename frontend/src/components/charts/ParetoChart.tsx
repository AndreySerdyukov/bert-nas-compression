import Frame from "./Frame";
import {
  formatAccuracy,
  formatAccuracyPrecise,
  formatParams,
  linear,
  log,
  padDomain,
} from "./scale";

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
  /**
   * A control rather than a searched architecture: drawn as a small square in the muted colour and
   * left unlabelled.
   *
   * Unlabelled because the controls share parameter counts exactly - every four-layer mask is
   * 52 780 802 parameters, so eight of them stack into one vertical line and eight captions would
   * be a smear. The reader is meant to see the cloud and where the searched models sit inside it;
   * the exact figures are in the table below and in the screen-reader table Frame renders.
   */
  control?: boolean;
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

  /**
   * Where each label goes, resolved before the render so no two of them sit on top of each other.
   *
   * AlphaNAS and BANANAS forced this: byte-identical architectures at 52 780 802 parameters, four
   * hundredths of a point apart, so their captions land within a couple of pixels and overprint
   * into an unreadable smudge. Anything still colliding gets pushed down a line at a time. The
   * order is by accuracy descending, so the model on top keeps the natural position above its
   * point and the one underneath moves.
   */
  const taken: { left: number; right: number; ly: number }[] = [];
  const labelPositions = points
    .filter((point) => !point.control)
    .sort((a, b) => b.accuracy - a.accuracy)
    .map((point) => {
      const px = x(point.params);
      const py = y(point.accuracy);
      // Close enough at fontSize 10 to keep two captions from touching.
      const width = point.label.length * 5.2;

      // Above the point, then out to the side, then further above. Sideways before upwards
      // because a caption that has drifted two lines up stops reading as belonging to its point,
      // and at the 52.8M column there is empty plot to the right.
      const candidates: { lx: number; ly: number; anchor: "middle" | "start" | "end" }[] = [
        { lx: px, ly: py - 10, anchor: "middle" },
        { lx: px + 9, ly: py + 3.5, anchor: "start" },
        { lx: px - 9, ly: py + 3.5, anchor: "end" },
        { lx: px, ly: py - 22, anchor: "middle" },
        { lx: px, ly: py - 34, anchor: "middle" },
      ];

      const extent = (lx: number, anchor: "middle" | "start" | "end") =>
        anchor === "middle"
          ? { left: lx - width / 2, right: lx + width / 2 }
          : anchor === "start"
            ? { left: lx, right: lx + width }
            : { left: lx - width, right: lx };

      const free = candidates.find(({ lx, ly, anchor }) => {
        const { left, right } = extent(lx, anchor);
        return !taken.some(
          (other) => left < other.right && right > other.left && Math.abs(other.ly - ly) < 11,
        );
      });
      const chosen = free ?? candidates[candidates.length - 1]!;
      taken.push({ ...extent(chosen.lx, chosen.anchor), ly: chosen.ly });
      return { point, px, py, ...chosen };
    });

  // The frontier over trained models only: an ablation is not something anyone would ship.
  // Controls do belong on it - they were trained under the same protocol, and a frontier that
  // excluded them would draw the searched models as the best available at their size when the
  // measurement says otherwise.
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
            // Two decimals here, not one: this table is what the browser tests read, and at one
            // decimal AlphaNAS and BANANAS become the same string.
            formatAccuracyPrecise(point.accuracy),
            point.ablation
              ? "ablation, not retrained"
              : point.baseline
                ? "baseline"
                : point.control
                  ? "control"
                  : "trained",
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

        {/* Controls first, so a searched model never disappears under the cloud it is being
            compared against. */}
        {points
          .filter((point) => point.control)
          .map((point) => (
            <rect
              key={`c-${point.label}`}
              x={x(point.params) - 3}
              y={y(point.accuracy) - 3}
              width={6}
              height={6}
              fill="rgb(var(--slate) / 0.55)"
              stroke="rgb(var(--slate))"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
          ))}

        {labelPositions.map(({ point, px, py, lx, ly, anchor }) => (
          <g key={`${point.label}-${point.params}-${point.accuracy}`}>
            <circle
              cx={px}
              cy={py}
              r={point.baseline ? 5.5 : 4.5}
              fill={point.ablation ? "rgb(var(--canvas))" : "rgb(var(--accent))"}
              stroke="rgb(var(--accent))"
              strokeWidth="1.4"
              vectorEffect="non-scaling-stroke"
            />
            {/* Painted with a halo of the panel colour: at 50M the controls stack into a dense
                column and a plain label lands with squares struck through the letters. Stroke
                first, fill over it, so the outline never eats into the glyphs. */}
            <text
              x={lx}
              y={ly}
              textAnchor={anchor}
              fill="rgb(var(--slate))"
              fontSize="10"
              stroke="rgb(var(--surface))"
              strokeWidth="3.5"
              paintOrder="stroke"
              strokeLinejoin="round"
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
