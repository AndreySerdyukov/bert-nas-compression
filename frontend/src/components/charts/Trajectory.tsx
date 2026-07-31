import type { Candidate } from "../../api";
import Frame from "./Frame";
import { formatAccuracy, formatParams, linear, padDomain } from "./scale";

/**
 * What one search actually did: every candidate it evaluated, in the order it evaluated them.
 *
 * One component renders all three searches. That is the point rather than an economy - the three
 * printed different formats and ran wildly different budgets, and drawing them identically is what
 * makes the difference in *shape* visible. Random Search is three dots. AlphaNAS is three
 * generations that converge. BANANAS is ten dots scattered between 0.58 and 0.69, which is what a
 * search on 560 training rows looks like when it is reading noise.
 *
 * Marks carry three channels: position is (order, accuracy), radius is parameter count, and a
 * hollow ring means the candidate was rejected outright rather than scored - AlphaNAS's all-ones
 * seed, every generation, by a parameter cap it could never satisfy.
 */

export interface TrajectoryProps {
  candidates: Candidate[];
  /** Drawn as a step line: the best fitness seen so far. Off when there are too few points. */
  showRunningBest?: boolean;
  /** Draws a rule between the initial sample and the model-guided phase (BANANAS). */
  phaseBoundaryAfter?: number;
  ariaLabel: string;
}

const WIDTH = 640;
const HEIGHT = 220;
const PADDING = { top: 12, right: 16, bottom: 32, left: 52 };

export default function Trajectory({
  candidates,
  showRunningBest = true,
  phaseBoundaryAfter,
  ariaLabel,
}: TrajectoryProps) {
  if (candidates.length === 0) return null;

  const accuracies = candidates.map((candidate) => candidate.accuracy);
  const x = linear([0.5, candidates.length + 0.5], [PADDING.left, WIDTH - PADDING.right]);
  const y = linear(padDomain(Math.min(...accuracies), Math.max(...accuracies), 0.12), [
    HEIGHT - PADDING.bottom,
    PADDING.top,
  ]);

  const paramValues = candidates.map((candidate) => candidate.params);
  const minParams = Math.min(...paramValues);
  const maxParams = Math.max(...paramValues);
  const radius = (params: number) =>
    maxParams === minParams ? 4.5 : 3 + ((params - minParams) / (maxParams - minParams)) * 4;

  // Running best over the candidates the search would actually have kept. A rejected candidate
  // never entered the ranking, so it cannot move this line.
  const runningBest = candidates.reduce<number[]>((seen, candidate) => {
    const previous = seen.at(-1) ?? -Infinity;
    const eligible = !candidate.rejected && candidate.fitness !== null;
    return [...seen, eligible ? Math.max(previous, candidate.accuracy) : previous];
  }, []);

  const bestPath = runningBest
    .map((value, index) => `${index === 0 ? "M" : "L"}${x(index + 1)},${y(value)}`)
    .join(" ");

  return (
    <Frame
      width={WIDTH}
      height={HEIGHT}
      padding={PADDING}
      x={x}
      y={y}
      xLabel="candidate, in the order it was evaluated"
      yLabel="accuracy on the search's own validation split"
      xTicks={candidates.map((_, index) => index + 1)}
      yTicks={y.ticks(4)}
      formatX={(value) => String(value)}
      formatY={formatAccuracy}
      ariaLabel={ariaLabel}
      data={{
        columns: ["order", "layers", "accuracy", "params", "fitness", "status"],
        rows: candidates.map((candidate, index) => [
          index + 1,
          candidate.layers.join(" "),
          formatAccuracy(candidate.accuracy),
          formatParams(candidate.params),
          candidate.fitness === null ? "rejected" : candidate.fitness.toFixed(6),
          candidate.rejected
            ? (candidate.rejection_reason ?? "rejected")
            : candidate.final_reevaluation
              ? "final re-evaluation"
              : "scored",
        ]),
      }}
    >
      {showRunningBest && candidates.length > 2 && Number.isFinite(runningBest[0] ?? NaN) && (
        <path
          d={bestPath}
          fill="none"
          stroke="rgb(var(--slate))"
          strokeWidth="1"
          strokeDasharray="3 3"
          vectorEffect="non-scaling-stroke"
        />
      )}

      {phaseBoundaryAfter !== undefined && phaseBoundaryAfter < candidates.length && (
        <line
          x1={x(phaseBoundaryAfter + 0.5)}
          x2={x(phaseBoundaryAfter + 0.5)}
          y1={PADDING.top}
          y2={HEIGHT - PADDING.bottom}
          stroke="rgb(var(--slate))"
          strokeWidth="1"
          strokeDasharray="2 4"
          vectorEffect="non-scaling-stroke"
        />
      )}

      {candidates.map((candidate, index) => {
        const cx = x(index + 1);
        const cy = y(candidate.accuracy);
        const r = radius(candidate.params);
        if (candidate.rejected) {
          // Hollow with a slash: evaluated, then thrown away without a score.
          return (
            <g key={index}>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="rgb(var(--slate))"
                strokeWidth="1.2"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={cx - r}
                y1={cy + r}
                x2={cx + r}
                y2={cy - r}
                stroke="rgb(var(--slate))"
                strokeWidth="1.2"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          );
        }
        return (
          <circle
            key={index}
            cx={cx}
            cy={cy}
            r={r}
            fill={candidate.final_reevaluation ? "rgb(var(--canvas))" : "rgb(var(--accent))"}
            stroke="rgb(var(--accent))"
            strokeWidth="1.2"
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </Frame>
  );
}
