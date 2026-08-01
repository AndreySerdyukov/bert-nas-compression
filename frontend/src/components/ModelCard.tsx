import type { ModelPrediction } from "../api";
import { formatMs, formatParams, formatPercent } from "../lib/format";

/**
 * One model's verdict on one review.
 *
 * The card is arranged around the comparison rather than around the model: the accent appears only
 * where a model parted from the baseline, and the timing line names what kind of number it is
 * showing. A card that said "6.7 ms" with nothing beside it would be the defect this project was
 * rebuilt to remove.
 */
interface Props {
  prediction: ModelPrediction;
  /** The corpus label, when the scored text is an unedited example. Null once it is edited. */
  truth: number | null;
  /** Name of the baseline model, so the badge is decided by identity rather than by inference. */
  baseline: string;
}

/**
 * P(positive) on a 0-1 track with the decision boundary marked.
 *
 * Purely decorative: the same number is written out beside it, so the bar carries no information a
 * screen reader would miss and is hidden from the accessibility tree rather than mislabelled.
 */
function ProbabilityBar({ positive }: { positive: number }) {
  return (
    <div aria-hidden="true" className="relative mt-3 h-1.5 rounded-full bg-raised">
      <div
        className="absolute inset-y-0 left-0 rounded-full bg-ink"
        style={{ width: `${(positive * 100).toFixed(1)}%` }}
      />
      {/* The whole question is which side of this line the probability fell on. */}
      <div className="absolute inset-y-[-3px] left-1/2 w-px bg-hair" />
    </div>
  );
}

function Badge({ children, tone }: { children: string; tone: "accent" | "muted" }) {
  const classes =
    tone === "accent"
      ? "border-accent/50 bg-accent/10 text-accent"
      : "border-hair bg-raised text-slate";
  return (
    <span className={`rounded-row border px-1.5 py-0.5 text-[11px] font-medium ${classes}`}>
      {children}
    </span>
  );
}

export default function ModelCard({ prediction, truth, baseline }: Props) {
  const positive = prediction.verdict === "positive";
  const confidence = positive
    ? prediction.positive_probability
    : 1 - prediction.positive_probability;
  const disagrees = prediction.agrees_with_baseline === false;
  // By name, not by `agrees_with_baseline === null`. That field is also null on every card when
  // the baseline is not being served, so the inferred version badged all five as the baseline.
  const isBaseline = prediction.name === baseline;
  const correct = truth === null ? null : (positive ? 1 : 0) === truth;
  const { latency } = prediction;

  return (
    <article
      data-model={prediction.name}
      className={`panel p-4 transition-colors ${disagrees ? "border-accent/60" : ""}`}
    >
      <header className="flex items-baseline gap-2">
        <h3 className="text-[15px] font-semibold tracking-tight">{prediction.label}</h3>
        <div className="ml-auto flex items-center gap-1.5">
          {isBaseline && <Badge tone="muted">baseline</Badge>}
          {disagrees && <Badge tone="accent">disagrees</Badge>}
        </div>
      </header>

      <p className="mt-3 flex items-baseline gap-2">
        <span className={`text-[22px] font-semibold ${disagrees ? "text-accent" : "text-ink"}`}>
          {prediction.verdict}
        </span>
        <span className="tnum text-[14px] text-slate">{formatPercent(confidence)}</span>
        {correct !== null && (
          <span className="tnum ml-auto text-[12px] text-slate">
            {correct ? "matches the corpus label" : "wrong: the corpus says otherwise"}
          </span>
        )}
      </p>

      <ProbabilityBar positive={prediction.positive_probability} />

      <dl className="mt-4 space-y-1 text-[13px] text-slate">
        <div className="flex justify-between gap-3">
          <dt>Size</dt>
          <dd className="tnum text-ink">
            {prediction.n_layers === null ? "no encoder layers" : `${prediction.n_layers} layers`}
            {" · "}
            {formatParams(prediction.params)} params
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          {/* Two different quantities never share a label: this row says which one it is. */}
          <dt>{latency ? "Latency" : "This scoring"}</dt>
          <dd className="tnum text-ink">
            {latency
              ? `${formatMs(latency.median_ms)} median · ${formatMs(latency.p95_ms)} p95`
              : formatMs(prediction.elapsed_ms)}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>Read</dt>
          {/* The token count belongs beside the timing: these models truncate at different
              lengths, so the same review is not the same amount of work for all of them. */}
          <dd className="tnum text-ink">{prediction.n_tokens} tokens</dd>
        </div>
        {latency && (
          <p className="pt-1 text-[12px] text-slate">
            {latency.repeats} repeats, {latency.warmup} warm-up passes discarded
          </p>
        )}
      </dl>
    </article>
  );
}
