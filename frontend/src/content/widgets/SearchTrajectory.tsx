import Trajectory from "../../components/charts/Trajectory";
import LayerMask from "../../components/LayerMask";
import { useTrajectories } from "../../lib/useContent";
import Figure from "./Figure";

/**
 * One search's trajectory, straight from the committed extraction.
 *
 * The budget line under the chart is not decoration. The three searches ran on 2 800, 5 600 and
 * 560 training rows, and no comparison between them survives that difference - so the number is
 * stated wherever a trajectory is shown, rather than in a caveat further down the page.
 */
export default function SearchTrajectory({
  method,
  phaseBoundaryAfter,
}: {
  method: "random-search" | "alphanas" | "bananas";
  phaseBoundaryAfter?: number;
}) {
  const { data, error, loading } = useTrajectories();

  if (loading) return <Figure wide>Loading the search log…</Figure>;
  if (error || !data) return <Figure wide>Could not load the search log: {error}</Figure>;

  const trajectory = data.methods[method];
  if (!trajectory) return null;

  const { budget, candidates } = trajectory;
  const scored = candidates.filter((candidate) => !candidate.rejected);
  const best = scored.reduce(
    (a, b) => ((b.fitness ?? -Infinity) > (a.fitness ?? -Infinity) ? b : a),
    scored[0]!,
  );

  return (
    <Figure
      wide
      caption={
        `${candidates.length} candidates, each trained for ${budget.epochs_per_candidate} epoch on ` +
        `${budget.train_rows.toLocaleString("en-US")} rows at ${budget.max_length} tokens. ` +
        `Radius is parameter count; a crossed circle is a candidate rejected before it was scored. ` +
        `Fitness: ${budget.fitness}`
      }
    >
      <Trajectory
        candidates={candidates}
        phaseBoundaryAfter={phaseBoundaryAfter}
        ariaLabel={`${method} search log: ${candidates.length} candidates evaluated on ${budget.train_rows} training rows`}
      />
      <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-hair pt-3 font-sans text-[13px]">
        <span className="text-slate">Best by fitness:</span>
        <LayerMask mask={best.mask} size={18} />
        <span className="tnum text-ink">
          layers [{best.layers.join(", ")}] · {(best.accuracy * 100).toFixed(2)}%
        </span>
      </div>
    </Figure>
  );
}
