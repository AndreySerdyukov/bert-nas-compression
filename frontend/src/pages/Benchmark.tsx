import type { BenchmarkModel, ControlResult } from "../api";
import LatencyBars from "../components/charts/LatencyBars";
import ParetoChart, { type ParetoPoint } from "../components/charts/ParetoChart";
import { formatAccuracyPrecise, formatBytes, formatParams } from "../components/charts/scale";
import { nonMaskControls, verdictByDepth } from "../lib/controls";
import { useBenchmark, useControls } from "../lib/useContent";

/**
 * Every number this project publishes, and the comparison they were produced to make.
 *
 * The page leads with the controls rather than with the NAS table, and that ordering is the
 * finding. Measured on the same 15 000 rows, TF-IDF with logistic regression beats all four
 * searched architectures, and DistilBERT at the same parameter count beats the best of them by
 * more than two points. A page that opened with "search compressed BERT to 48% of its size at a
 * 2.5 point cost" would be true and would leave the reader with the wrong conclusion.
 *
 * The two halves degrade independently. `benchmark.json` and `controls.json` are separate
 * documents produced by separate scripts, either can be absent in a fresh clone, and the endpoint
 * answers 503 naming the script that makes it. So a missing document renders as an explanation
 * with a command in it, never as an error.
 */

/** The accuracy budget the project set itself, in points below the baseline. */
const BUDGET_PP = 7;

function ownMemory(model: BenchmarkModel): number | null {
  const { resident_bytes: total, runtime_baseline_bytes: floor } = model;
  if (total === null || floor === null) return null;
  return total - floor;
}

function Missing({ what, command }: { what: string; command: string }) {
  return (
    <div className="panel p-4 text-[14px] text-slate">
      <p>
        No {what} on this machine yet. It is not committed to a fresh clone until someone has spent
        the hours producing it.
      </p>
      <p className="mt-2">
        Run <code className="kbd font-mono">{command}</code> from{" "}
        <code className="kbd">backend/</code>.
      </p>
    </div>
  );
}

function ControlRow({ control }: { control: ControlResult }) {
  const layers = control.layers === null ? "-" : control.layers.join(",");
  return (
    <tr className="border-t border-hair">
      <td className="py-2 pr-4">{control.label}</td>
      <td className="py-2 pr-4 font-mono text-[12px] text-slate">{layers}</td>
      <td className="py-2 pr-4 text-right tabular-nums">
        {formatAccuracyPrecise(control.accuracy)}
      </td>
      <td className="py-2 pr-4 text-right tabular-nums text-slate">
        {formatAccuracyPrecise(control.macro_f1)}
      </td>
      <td className="py-2 text-[13px] text-slate">{control.question}</td>
    </tr>
  );
}

export default function Benchmark() {
  const benchmark = useBenchmark();
  const controls = useControls();

  if (benchmark.loading || controls.loading) {
    return (
      <main className="mx-auto max-w-[900px] px-6 py-10">
        <p className="text-slate">Loading the measurements…</p>
      </main>
    );
  }

  const models = benchmark.data?.models ?? [];
  const protocol = benchmark.data?.protocol ?? null;
  const controlResults = controls.data?.controls ?? [];
  const verdicts = verdictByDepth(models, controlResults);
  const extras = nonMaskControls(controlResults);

  const points: ParetoPoint[] = [
    ...models.map((model) => ({
      label: model.label,
      params: model.params,
      accuracy: model.accuracy,
      baseline: model.method === null,
    })),
    // A control with no parameter count of its own cannot go on these axes at all: TF-IDF is a
    // sparse linear model, and placing it at "zero parameters" would be a claim, not a measurement.
    ...controlResults
      .filter(
        (control): control is ControlResult & { train: { params: number } } =>
          typeof control.train.params === "number",
      )
      .map((control) => ({
        label: control.label,
        params: control.train.params,
        accuracy: control.accuracy,
        control: true,
      })),
  ];

  const bars = models
    .filter((model) => model.latency !== null)
    .map((model) => ({
      label: model.label,
      medianMs: model.latency!.median_ms,
      p95Ms: model.latency!.p95_ms,
      nTokens: model.latency!.n_tokens,
    }));

  return (
    <main className="mx-auto max-w-[900px] px-6 py-10">
      <header>
        <h1 className="text-[32px] font-semibold leading-tight tracking-tight">Benchmark</h1>
        <p className="mt-2 max-w-[68ch] text-[15px] text-slate">
          Every model re-measured here, on the full test split, through the same registry the app
          serves from - and beside them the controls that say what the search was actually worth.
        </p>
      </header>

      {benchmark.error !== null && (
        <div className="mt-8">
          <Missing what="benchmark" command="python -m training.benchmark" />
        </div>
      )}

      {protocol !== null && (
        <>
          <section className="mt-10">
            <h2 className="section-label">Accuracy against size</h2>
            <div className="panel mt-3 p-4">
              <ParetoChart
                points={points}
                budgetPp={BUDGET_PP}
                ariaLabel={`Accuracy against parameter count for ${models.length} models and ${points.length - models.length} controls, all measured on ${protocol.test_rows.toLocaleString("en-US")} rows`}
                caption={
                  controlResults.length > 0
                    ? `Circles are models someone shipped, squares are controls trained here under the same protocol. The band is the ${BUDGET_PP} point budget the project set itself. The frontier steps through controls, not through the searched architectures.`
                    : `The band is the ${BUDGET_PP} point accuracy budget the project set itself.`
                }
              />
            </div>
          </section>

          <section className="mt-10">
            <h2 className="section-label">The models</h2>
            <div className="panel mt-3 overflow-x-auto p-4">
              {/* Named, because three tables on this page carry a row per model - this one and the
                  screen-reader tables under the two charts - and landing on one of them without a
                  name says nothing about which. */}
              <table
                aria-label="Models measured on the full test split"
                className="w-full text-[14px]"
              >
                <thead className="text-left text-[12px] uppercase tracking-wider text-slate">
                  <tr>
                    <th className="pb-2 pr-4 font-medium">Model</th>
                    <th className="pb-2 pr-4 font-medium">Method</th>
                    <th className="pb-2 pr-4 text-right font-medium">Accuracy</th>
                    <th className="pb-2 pr-4 text-right font-medium">Macro F1</th>
                    <th className="pb-2 pr-4 text-right font-medium">Params</th>
                    <th className="pb-2 pr-4 text-right font-medium">Latency</th>
                    <th className="pb-2 text-right font-medium">Memory</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => (
                    <tr key={model.name} className="border-t border-hair">
                      <td className="py-2 pr-4">{model.label}</td>
                      <td className="py-2 pr-4 text-[13px] text-slate">
                        {model.method ?? "fine-tune (baseline)"}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums">
                        {formatAccuracyPrecise(model.accuracy)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums text-slate">
                        {formatAccuracyPrecise(model.macro_f1)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums text-slate">
                        {formatParams(model.params)}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums text-slate">
                        {model.latency === null
                          ? "not measured"
                          : `${model.latency.median_ms.toFixed(1)} ms`}
                      </td>
                      <td className="py-2 text-right tabular-nums text-slate">
                        {(() => {
                          const own = ownMemory(model);
                          return own === null ? "not measured" : formatBytes(own);
                        })()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 max-w-[68ch] text-[12px] text-slate">
                Accuracy on <code className="font-mono">{protocol.accuracy_device}</code> over the
                full {protocol.test_rows.toLocaleString("en-US")} row test split. Latency is
                single-example, batch 1, on{" "}
                <code className="font-mono">{protocol.latency_device}</code> at{" "}
                {protocol.latency_threads} thread
                {protocol.latency_threads === 1 ? "" : "s"}: {protocol.latency_warmup} warm-up
                passes discarded, median of {protocol.latency_repeats} repeats, models interleaved
                round-robin. Memory is the resident set of a process holding one model, less the{" "}
                {formatBytes(
                  Math.min(
                    ...models
                      .map((model) => model.runtime_baseline_bytes)
                      .filter((value): value is number => value !== null),
                  ),
                )}{" "}
                floor every model pays and none of them owns.
              </p>
            </div>
          </section>

          {bars.length > 0 && (
            <section className="mt-10">
              <h2 className="section-label">Latency, median and p95</h2>
              <div className="panel mt-3 p-4">
                <LatencyBars
                  bars={bars}
                  device={protocol.latency_device}
                  threads={protocol.latency_threads}
                  ariaLabel={`Single-example latency for ${bars.length} models, median with p95 above it`}
                  caption="The cap is p95, the bar is the median. All of them timed the same review, but not on the same amount of work: AdaBERT has no attention mask and is served at a fixed 128 tokens, so its bar is a shorter time over more tokens rather than a shorter time over the same ones."
                />
              </div>
            </section>
          )}
        </>
      )}

      {controls.error !== null && (
        <section className="mt-10">
          <h2 className="section-label">The controls</h2>
          <div className="mt-3">
            <Missing what="controls" command="python -m training.train_reference" />
          </div>
        </section>
      )}

      {controlResults.length > 0 && controls.data !== null && (
        <>
          <section className="mt-12">
            <h2 className="text-[22px] font-semibold tracking-tight">
              Did searching beat a simple rule?
            </h2>
            <p className="mt-2 max-w-[68ch] text-[15px] text-slate">
              A mask can only be judged against masks of its own depth, and the two depths do not
              agree. Every figure below is the same 15 000 rows and the same one-epoch protocol the
              shipped checkpoints were trained under.
            </p>

            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              {verdicts.map((verdict) => (
                <div key={verdict.nLayers} className="panel p-4">
                  <h3 className="text-[15px] font-semibold">{verdict.nLayers} layers</h3>
                  <dl className="mt-3 space-y-1.5 text-[14px]">
                    {verdict.searched.map((model) => (
                      <div key={model.label} className="flex justify-between gap-4">
                        <dt className="text-accent">{model.label}</dt>
                        <dd className="tabular-nums">{formatAccuracyPrecise(model.accuracy)}</dd>
                      </div>
                    ))}
                    {verdict.first && (
                      <div className="flex justify-between gap-4">
                        <dt className="text-slate">first-{verdict.nLayers}, the bottom layers</dt>
                        <dd className="tabular-nums">
                          {formatAccuracyPrecise(verdict.first.accuracy)}
                        </dd>
                      </div>
                    )}
                    {verdict.uniform && (
                      <div className="flex justify-between gap-4">
                        <dt className="text-slate">uniform-{verdict.nLayers}, evenly spaced</dt>
                        <dd className="tabular-nums">
                          {formatAccuracyPrecise(verdict.uniform.accuracy)}
                        </dd>
                      </div>
                    )}
                    {verdict.last && (
                      <div className="flex justify-between gap-4">
                        <dt className="text-slate">last-{verdict.nLayers}, the top layers</dt>
                        <dd className="tabular-nums">
                          {formatAccuracyPrecise(verdict.last.accuracy)}
                        </dd>
                      </div>
                    )}
                    {verdict.randomMedian !== null && (
                      <div className="flex justify-between gap-4">
                        <dt className="text-slate">
                          median of {verdict.randomAccuracies.length} random masks
                        </dt>
                        <dd className="tabular-nums">
                          {formatAccuracyPrecise(verdict.randomMedian)}
                        </dd>
                      </div>
                    )}
                  </dl>

                  <p className="mt-3 border-t border-hair pt-3 text-[13px] text-slate">
                    {verdict.vsUniformPp !== null && (
                      <>
                        Against the evenly spaced mask the search is{" "}
                        <strong className="text-ink">
                          {verdict.vsUniformPp >= 0 ? "+" : ""}
                          {verdict.vsUniformPp.toFixed(2)} pp
                        </strong>
                        .{" "}
                      </>
                    )}
                    {verdict.vsFirstPp !== null && (
                      <>
                        Against the bottom {verdict.nLayers} layers it is{" "}
                        <strong className="text-ink">
                          {verdict.vsFirstPp >= 0 ? "+" : ""}
                          {verdict.vsFirstPp.toFixed(2)} pp
                        </strong>
                        .{" "}
                      </>
                    )}
                    {verdict.percentile !== null && (
                      <>
                        Inside the random masks of this depth it sits at the{" "}
                        <strong className="text-ink">{verdict.percentile.toFixed(0)}th</strong>{" "}
                        percentile.
                      </>
                    )}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {extras.length > 0 && (
            <section className="mt-10">
              <h2 className="section-label">And the two controls that are not masks at all</h2>
              <div className="panel mt-3 p-4">
                <dl className="space-y-3 text-[14px]">
                  {extras.map((control) => (
                    <div key={control.key} className="flex items-baseline justify-between gap-4">
                      <dt>
                        {control.label}
                        <span className="ml-2 text-[13px] text-slate">{control.question}</span>
                      </dt>
                      <dd className="shrink-0 tabular-nums">
                        {formatAccuracyPrecise(control.accuracy)}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            </section>
          )}

          <section className="mt-10">
            <h2 className="section-label">All {controlResults.length} controls</h2>
            <div className="panel mt-3 overflow-x-auto p-4">
              <table
                aria-label="Every control that was run, best first"
                className="w-full text-[14px]"
              >
                <thead className="text-left text-[12px] uppercase tracking-wider text-slate">
                  <tr>
                    <th className="pb-2 pr-4 font-medium">Control</th>
                    <th className="pb-2 pr-4 font-medium">Layers</th>
                    <th className="pb-2 pr-4 text-right font-medium">Accuracy</th>
                    <th className="pb-2 pr-4 text-right font-medium">Macro F1</th>
                    <th className="pb-2 font-medium">The question it answers</th>
                  </tr>
                </thead>
                <tbody>
                  {[...controlResults]
                    .sort((a, b) => b.accuracy - a.accuracy)
                    .map((control) => (
                      <ControlRow key={control.key} control={control} />
                    ))}
                </tbody>
              </table>
              <p className="mt-3 max-w-[68ch] text-[12px] text-slate">
                Trained under the shipped checkpoints&apos; own protocol, read out of the eval
                notebooks: {controls.data.protocol.epochs} epoch over all{" "}
                {controls.data.protocol.train_rows.toLocaleString("en-US")} training rows,{" "}
                {controls.data.protocol.max_length} tokens, batch{" "}
                {controls.data.protocol.batch_size}, AdamW at {controls.data.protocol.learning_rate}{" "}
                with weight decay {controls.data.protocol.weight_decay} and a{" "}
                {Math.round(controls.data.protocol.warmup_ratio * 100)}% warm-up. Nothing here was
                tuned - shortening the training would bias every comparison on this page in this
                project&apos;s favour. Weights are not saved: eighteen fine-tuned BERTs are seven
                gigabytes, and a control&apos;s deliverable is a number.
              </p>
              {controls.data.not_run.length > 0 && (
                <p className="mt-2 max-w-[68ch] text-[12px] text-accent">
                  Not run: {controls.data.not_run.join(", ")}. {controlResults.length} of{" "}
                  {controls.data.planned.length} planned controls completed within the time budget.
                </p>
              )}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
