import { Link } from "react-router-dom";

import { formatAccuracyPrecise } from "../components/charts/scale";
import { useBenchmark, useControls } from "../lib/useContent";

/**
 * The front page, which leads with a control rather than with the compression table.
 *
 * That ordering is the whole argument of the rebuild. "Four NAS methods compressed BERT to under
 * half its size for two and a half points" is true, and on its own it is the sentence that made
 * this project worth redoing: measured on the same split, a bag of bigrams with a logistic
 * regression on top scores higher than all four, and a distilled model of the same size gives up
 * seven times less accuracy. A reader who leaves with only the first sentence has been misled by
 * an arrangement of true facts.
 *
 * Everything here is read from the two measurement documents. Nothing is a literal in this file,
 * because a headline figure typed into prose is one that can drift from the JSON it came from -
 * which is the defect chapter 9 is about.
 */

interface HeadlineProps {
  label: string;
  value: string;
  note: string;
  accent?: boolean;
}

function Headline({ label, value, note, accent }: HeadlineProps) {
  return (
    <div className="panel p-4">
      <div className="section-label">{label}</div>
      <div
        className={`mt-2 text-[28px] font-semibold tabular-nums tracking-tight ${accent ? "text-accent" : ""}`}
      >
        {value}
      </div>
      <p className="mt-1 text-[13px] leading-snug text-slate">{note}</p>
    </div>
  );
}

function Card({ to, title, body }: { to: string; title: string; body: string }) {
  return (
    <Link to={to} className="panel block p-4 transition-colors hover:bg-raised">
      <div className="text-[15px] font-semibold">{title}</div>
      <p className="mt-1 text-[14px] leading-snug text-slate">{body}</p>
    </Link>
  );
}

export default function Overview() {
  const benchmark = useBenchmark();
  const controls = useControls();

  const models = benchmark.data?.models ?? [];
  const searched = models.filter((model) => model.method !== null);
  const bestSearched = searched.reduce<(typeof searched)[number] | null>(
    (best, model) => (best === null || model.accuracy > best.accuracy ? model : best),
    null,
  );
  const baseline = models.find((model) => model.method === null) ?? null;

  const results = controls.data?.controls ?? [];
  const tfidf = results.find((control) => control.kind === "tfidf") ?? null;
  const distil = results.find((control) => control.kind === "distilbert") ?? null;

  return (
    <main className="mx-auto max-w-[900px] px-6 py-12">
      <h1 className="max-w-[20ch] text-[40px] font-semibold leading-[1.05] tracking-tight">
        What Neural Architecture Search actually bought
      </h1>
      <p className="mt-4 max-w-[68ch] text-[16px] leading-relaxed text-slate">
        Four NAS methods compressed a BERT sentiment classifier to under half its size, and the
        original project measured them against each other. This rebuild measures them against
        something else: controls, on the same 15 000 rows, under the same protocol.
      </p>

      {bestSearched && tfidf && (
        <>
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <Headline
              label="Best searched model"
              value={formatAccuracyPrecise(bestSearched.accuracy)}
              note={`${bestSearched.label}, ${bestSearched.n_layers ?? "?"} of 12 encoder layers, found by a search that ran for days`}
            />
            <Headline
              label="TF-IDF + logistic regression"
              value={formatAccuracyPrecise(tfidf.accuracy)}
              note={`No transformer at all, ${Math.round(tfidf.train.seconds)} seconds of CPU, and it scores higher`}
              accent
            />
            {distil && (
              <Headline
                label="DistilBERT, same size"
                value={formatAccuracyPrecise(distil.accuracy)}
                note={
                  baseline
                    ? `Gives up ${((baseline.accuracy - distil.accuracy) * 100).toFixed(2)} points to full BERT, where the best searched model gives up ${((baseline.accuracy - bestSearched.accuracy) * 100).toFixed(2)}`
                    : "A distilled model of comparable size"
                }
              />
            )}
          </div>

          <p className="mt-6 max-w-[68ch] text-[15px] leading-relaxed">
            None of this says the searches were built wrong - they were implemented from scratch and
            they work. It says the comparison that would have told anyone what they were worth was
            never run. Run now, it puts the searched masks inside the spread of masks chosen at
            random, and below two rules simple enough to write on one line.
          </p>
        </>
      )}

      <div className="mt-10 grid gap-4 sm:grid-cols-3">
        <Card
          to="/benchmark"
          title="Benchmark"
          body="Every number, the controls beside them, and the verdict per depth."
        />
        <Card
          to="/playground"
          title="Playground"
          body="One review, every model at once, and the reviews where they disagree."
        />
        <Card
          to="/methodology"
          title="Methodology"
          body="Ten chapters on how the searches worked and what the audit found."
        />
      </div>
    </main>
  );
}
