import ParetoChart from "../../components/charts/ParetoChart";
import { useReportedResults } from "../../lib/useContent";
import Figure from "./Figure";

/**
 * The original project's own head-to-head, on the trade-off axes.
 *
 * Explicitly labelled as *reported*, not measured. Chapter 9 is about how far these figures can be
 * trusted, and a chart that quietly presented them as this project's answer would undercut it
 * before the reader got there. Once benchmark.json exists this same component takes the measured
 * points, and the two can be shown side by side.
 */
export default function ReportedPareto({ budgetPp = 7 }: { budgetPp?: number }) {
  const { data, error, loading } = useReportedResults();

  if (loading) return <Figure wide>Loading the reported results…</Figure>;
  if (error || !data) return <Figure wide>Could not load the reported results: {error}</Figure>;

  // One baseline, not three. Which fine-tune to treat as the reference is exactly the question the
  // rebuild had to settle, and the answer is the run with a real held-out split.
  const shown = data.models.filter(
    (model) => model.key !== "bert-ilya" && model.key !== "bert-alina",
  );

  const points = shown.map((model) => ({
    label: model.label.replace(" (Andrey)", ""),
    params: model.params,
    accuracy: model.accuracy.nas_results!,
    baseline: model.key === "bert-andrey",
  }));

  return (
    <Figure wide>
      <ParetoChart
        points={points}
        budgetPp={budgetPp}
        ariaLabel={`Reported accuracy against parameter count for ${points.length} models, from the original head-to-head notebook`}
        caption={`Reported by ${data.sources.nas_results!.label}, not measured here. The band is the ${budgetPp} point accuracy budget the project set itself; everything in it met the goal. The other two full-BERT runs are left out - one baseline is the point.`}
      />
    </Figure>
  );
}
