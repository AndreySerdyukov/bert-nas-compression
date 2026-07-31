import { useState } from "react";

import { ApiError, compare, scanResult, type CompareResponse, type Example } from "../api";
import JobProgress from "../components/JobProgress";
import ModelCard from "../components/ModelCard";
import ScanReport from "../components/ScanReport";
import { describeRuntime } from "../lib/format";
import { useJob } from "../lib/useJob";
import { useExamples, useModels } from "../lib/useServing";

/** Rows of the evaluation sample the scan can cover. The full sample is the number that counts. */
const FULL_SAMPLE = 2000;
const SCAN_SIZES = [
  { rows: FULL_SAMPLE, label: "Scan all 2,000" },
  { rows: 500, label: "First 500" },
  { rows: 200, label: "First 200" },
] as const;

/**
 * One review, every model at once.
 *
 * The page is arranged so the comparison is the subject and the individual verdicts are the
 * evidence: the models are scored in one request, measured round-robin, and the only thing drawn
 * in the accent colour is a model parting from the baseline. Where a model disagrees is the
 * interesting part - the headline accuracy gap of a couple of points is exactly this, a few dozen
 * reviews out of two thousand, and this is where a visitor gets to see one.
 *
 * Every state a fresh clone can be in is a state this page renders: no weights downloaded, serving
 * switched off, some models skipped. None of them is an error screen.
 */
export default function Playground() {
  const catalog = useModels();
  const examples = useExamples();

  const [text, setText] = useState("");
  // The corpus label travels with the text only while the text is exactly the example. Editing a
  // review invalidates the label, and claiming otherwise would be the page lying about its data.
  const [chosen, setChosen] = useState<Example | null>(null);
  const [measureLatency, setMeasureLatency] = useState(true);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The text the shown result belongs to, so an edited box cannot look like a fresh answer.
  const [scoredText, setScoredText] = useState("");

  const scan = useJob();

  const models = catalog.data?.models ?? [];
  const canScore = text.trim().length > 0 && models.length > 0 && !scoring;
  const truth = chosen !== null && chosen.text === scoredText ? chosen.label : null;
  const scanned = scanResult(scan.job);

  function pick(example: Example) {
    setText(example.text);
    setChosen(example);
  }

  function edit(value: string) {
    setText(value);
    if (chosen !== null && value !== chosen.text) setChosen(null);
  }

  async function score() {
    setScoring(true);
    setError(null);
    try {
      const response = await compare({ text, measure_latency: measureLatency });
      setResult(response);
      setScoredText(text);
    } catch (caught: unknown) {
      setResult(null);
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setScoring(false);
    }
  }

  return (
    <div className="mx-auto max-w-content px-6 py-14">
      <header className="max-w-prose">
        <p className="section-label">Playground</p>
        <h1 className="mt-2 text-[34px] font-semibold leading-tight tracking-tight">
          One review, every model
        </h1>
        <p className="mt-4 text-[16px] leading-relaxed text-slate">
          The uncompressed reference and its four compressed descendants score the same review in
          one request, measured round-robin so none of them pays for going last. A model is marked
          only where it parts from the baseline.
        </p>
      </header>

      {catalog.loading && <p className="mt-10 text-[14px] text-slate">Loading the catalog…</p>}

      {catalog.status === 503 && (
        <p className="panel mt-10 p-4 text-[14px] text-slate">
          Serving is switched off in this deployment. The methodology section needs no models and
          works completely; this page needs them.
        </p>
      )}

      {catalog.error !== null && catalog.status !== 503 && (
        <p className="panel mt-10 p-4 text-[14px] text-slate">
          The backend did not answer: {catalog.error}
        </p>
      )}

      {catalog.data !== null && (
        <>
          {models.length === 0 && (
            <div className="panel mt-10 p-4">
              <p className="text-[15px] font-medium text-ink">
                No checkpoints have been downloaded
              </p>
              <p className="mt-2 max-w-prose text-[14px] leading-relaxed text-slate">
                The weights are about 1.1 GB and are not committed, so a fresh clone has none of
                them. Everything else in this project works without them.
              </p>
              <pre className="mt-3 overflow-x-auto rounded-row border border-hair bg-canvas p-3 font-mono text-[13px]">
                python scripts/fetch_models.py
              </pre>
            </div>
          )}

          <section className="mt-10">
            <label htmlFor="review" className="section-label">
              The review
            </label>
            <textarea
              id="review"
              rows={7}
              value={text}
              onChange={(event) => edit(event.target.value)}
              placeholder="Paste a film review, or pick one below."
              className="field mt-2 resize-y font-serif leading-relaxed"
            />

            {/* The corpus carries HTML line breaks and not one of the eleven notebooks stripped
                them, so this is the text the checkpoints were fine-tuned on. Cleaning it here
                would score them on something they never saw, which is a worse defect than an
                ugly textarea - so the markup stays and the page says why. */}
            {text.includes("<br") && (
              <p className="mt-2 max-w-prose text-[13px] leading-relaxed text-slate">
                The <code className="font-mono">&lt;br /&gt;</code> markup is part of the corpus and
                is left in on purpose: no notebook stripped it, so it is what these models were
                fine-tuned on. Removing it here would score them on text they never saw.
              </p>
            )}

            {examples.data !== null && examples.data.examples.length > 0 && (
              <div className="mt-3">
                <p className="text-[13px] text-slate">
                  Or start from one of {examples.data.total.toLocaleString("en-US")} reviews in the
                  evaluation sample every model is scored on:
                </p>
                <ul className="mt-2 flex flex-wrap gap-2">
                  {examples.data.examples.map((example) => (
                    <li key={example.id}>
                      <button
                        type="button"
                        onClick={() => pick(example)}
                        aria-pressed={chosen?.id === example.id}
                        className={`tnum rounded-row border px-2 py-1 text-[12px] transition-colors ${
                          chosen?.id === example.id
                            ? "border-slate text-ink"
                            : "border-hair text-slate hover:bg-raised hover:text-ink"
                        }`}
                      >
                        #{example.id} · {example.label === 1 ? "positive" : "negative"}
                        {example.truncated && " · long"}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-4">
              <button type="button" onClick={score} disabled={!canScore} className="btn-primary">
                {scoring ? "Scoring…" : "Score with every model"}
              </button>
              <label className="flex items-center gap-2 text-[13px] text-slate">
                <input
                  type="checkbox"
                  checked={measureLatency}
                  onChange={(event) => setMeasureLatency(event.target.checked)}
                  className="accent-accent"
                />
                Measure latency (several more forward passes per model)
              </label>
            </div>
          </section>

          {error !== null && <p className="panel mt-6 p-4 text-[14px] text-slate">{error}</p>}

          {result !== null && (
            <section className="mt-10">
              <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-hair pb-3">
                <h2 className="text-[18px] font-semibold tracking-tight">
                  {result.disagree.length === 0
                    ? "Every model agrees with the baseline"
                    : `${result.disagree.length} of ${result.results.length - 1} part from the baseline`}
                </h2>
                {/* The rule of this project: no timing is rendered without the machine and thread
                    count it came from. */}
                <p className="tnum text-[12px] text-slate">{describeRuntime(result.runtime)}</p>
              </div>

              {text !== scoredText && (
                <p className="mt-3 text-[13px] text-slate">
                  The review has been edited since these were scored.
                </p>
              )}

              <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {result.results.map((prediction) => (
                  <ModelCard key={prediction.name} prediction={prediction} truth={truth} />
                ))}
              </div>
            </section>
          )}

          {models.length > 0 && (
            <section className="mt-14 border-t border-hair pt-8">
              <h2 className="text-[22px] font-semibold tracking-tight">
                Find every review they disagree on
              </h2>
              <p className="mt-2 max-w-prose text-[15px] leading-relaxed text-slate">
                One review at a time is an anecdote. This scores the whole evaluation sample with
                every model and keeps the rows where they part company, which is what a two-point
                accuracy gap actually looks like: a few dozen reviews, and you can read them.
              </p>

              <div className="mt-5 flex flex-wrap items-center gap-3">
                {SCAN_SIZES.map((size) => (
                  <button
                    key={size.rows}
                    type="button"
                    disabled={scan.busy}
                    onClick={() => void scan.start({ limit: size.rows })}
                    className={size.rows === FULL_SAMPLE ? "btn-primary" : "btn-secondary"}
                  >
                    {size.label}
                  </button>
                ))}
                {/* Stated up front rather than discovered halfway through: the full sample takes
                    minutes, and a reader deserves to choose knowingly. */}
                <p className="text-[13px] text-slate">
                  The full sample takes several minutes on a laptop.
                </p>
              </div>

              {scan.error !== null && (
                <p className="panel mt-6 p-4 text-[14px] text-slate">{scan.error}</p>
              )}

              {scan.job !== null && scan.job.status === "running" && <JobProgress job={scan.job} />}

              {scanned !== null && <ScanReport result={scanned} />}
            </section>
          )}

          {catalog.data.skipped.length > 0 && (
            <section className="mt-12">
              <h2 className="section-label">Not being served</h2>
              <ul className="mt-2 border-t border-hair">
                {catalog.data.skipped.map((skipped) => (
                  <li key={skipped.name} className="border-b border-hair py-3">
                    <p className="text-[14px] text-ink">{skipped.label}</p>
                    <p className="mt-1 text-[13px] text-slate">{skipped.reason}</p>
                    {skipped.remedy !== null && (
                      <code className="mt-1 block font-mono text-[12px] text-slate">
                        {skipped.remedy}
                      </code>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
