import { useState } from "react";

import { ablationResult, type ScoredRun } from "../api";
import JobProgress from "../components/JobProgress";
import LayerMask from "../components/LayerMask";
import MetricBlock from "../components/MetricBlock";
import { formatAccuracyPrecise, formatParams } from "../components/charts/scale";
import { PRESETS, trainedTwin } from "../lib/ablation";
import { N_LAYERS, layersFromMask, maskFromLayers, paramsFor } from "../lib/arch";
import { FULL_SAMPLE, JOB_SIZES } from "../lib/jobSizes";
import { useArchitectures, useBenchmark, useControls } from "../lib/useContent";
import { useJob } from "../lib/useJob";

/**
 * Pick any subset of the twelve encoder layers and see what it is worth without retraining.
 *
 * The framing is the page, and losing it would make this a toy. NAS does
 * `finetune(mask(pretrained))`; this does `mask(finetuned)` and stops. Accuracy collapses, and the
 * collapse is the finding: it is the size of what training each candidate buys, and therefore the
 * reason a search cannot skip that cost. A visitor who reads the number as "four layers are
 * hopeless" has been given the opposite of the lesson - four trained layers reach 90%.
 *
 * So the warning comes **before** the run rather than as a footnote under the result, and the
 * result always carries three columns: this amputation, the same mask trained properly where such
 * a model exists, and the same weights at full depth.
 */

// The bottom four: the mask that trained is the best of the naive rules, so the page opens on the
// comparison that has an answer rather than on an empty stack.
const DEFAULT_MASK = maskFromLayers([0, 1, 2, 3]);

/** The Wilson bounds as one string, for the slot MetricBlock keeps under the number. */
function interval(run: ScoredRun): string {
  return `${formatAccuracyPrecise(run.wilson_low)} – ${formatAccuracyPrecise(run.wilson_high)}`;
}

export default function Explorer() {
  const [mask, setMask] = useState<number[]>(DEFAULT_MASK);
  // Chosen rather than defaulted. Without a limit the job takes the server's cap, and an ablation
  // scores every row twice - once amputated, once at full depth - so the quiet default was the
  // longest piece of work in the application behind a button that mentioned no duration at all.
  const [rows, setRows] = useState<number>(FULL_SAMPLE);
  const ablation = useJob();
  const controls = useControls();
  const architectures = useArchitectures();
  const benchmark = useBenchmark();

  const layers = layersFromMask(mask);
  const kept = layers.length;
  const result = ablationResult(ablation.job);

  const twinOf = (of: readonly number[]) =>
    trainedTwin(of, controls.data?.controls ?? [], architectures.data);
  // Two lookups on purpose. The label under the strip follows what the reader is currently
  // clicking; the result column follows the mask that was actually scored. One lookup would let
  // an edit made after a run relabel a result it does not belong to.
  const selected = twinOf(layers);
  const twin = result ? twinOf(result.layers) : null;

  // A searched mask carries its accuracy in the benchmark rather than in the controls file.
  const twinAccuracy =
    twin?.source === "searched"
      ? (benchmark.data?.models.find((model) => model.label === twin.label)?.accuracy ?? Number.NaN)
      : (twin?.accuracy ?? Number.NaN);

  // Counted from the two documents rather than typed into the sentence below. It was "twenty-two
  // trained models" in prose, on a page whose every other figure is read from data - so adding a
  // control would have left the number quietly wrong.
  const trainedMasks =
    (controls.data?.controls.filter((control) => control.layers !== null).length ?? 0) +
    (benchmark.data?.models.filter((model) => model.method !== null && model.n_layers !== null)
      .length ?? 0);
  const allMasks = 2 ** N_LAYERS;

  function toggle(index: number) {
    setMask(mask.map((bit, position) => (position === index ? (bit === 1 ? 0 : 1) : bit)));
  }

  return (
    <main className="mx-auto max-w-[900px] px-6 py-10">
      <header>
        <h1 className="text-[32px] font-semibold leading-tight tracking-tight">Explorer</h1>
        <p className="mt-2 max-w-[68ch] text-[15px] text-slate">
          Remove encoder layers from the fine-tuned baseline and score what is left. Nothing is
          retrained, which is exactly what makes the result worth looking at.
        </p>
      </header>

      <section className="mt-8">
        <h2 className="section-label">The mask</h2>
        <div className="panel mt-3 p-4">
          <LayerMask mask={mask} onToggle={toggle} size={34} showIndices />
          <p className="mt-3 text-[14px] text-slate">
            {kept} of {N_LAYERS} layers kept
            {kept > 0 && <> · {formatParams(paramsFor(kept))} parameters</>}
            {selected && <> · this is the mask of {selected.label}</>}
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="btn-secondary text-[13px]"
                title={preset.why}
                onClick={() => setMask(maskFromLayers(preset.layers))}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Before the button, not under the result. A reader who runs this and sees 0.62 without
          having been told why has learned the wrong thing, and no footnote undoes that. */}
      <section className="mt-6">
        <div className="panel border-accent/40 p-4">
          <div className="section-label text-accent">Read this before you run it</div>
          <p className="mt-2 max-w-[68ch] text-[14px] leading-relaxed">
            The layers are cut out of a model that was fine-tuned with all twelve, and nothing is
            trained afterwards. The searched architectures were trained <em>after</em> their layers
            were chosen, and that training is the expensive half of NAS. So the number below is not
            what this mask is worth - it is what this mask is worth <strong>without</strong> the
            half that costs GPU-days, and the gap between the two columns is the measurement.
          </p>
        </div>
      </section>

      <section className="mt-6">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="btn-primary"
            disabled={kept === 0 || ablation.busy}
            onClick={() => void ablation.start({ kind: "ablation", layers, limit: rows })}
          >
            {ablation.busy ? "Scoring…" : `Score ${kept} layers`}
          </button>
          <label className="flex items-center gap-2 text-[13px] text-slate">
            over
            <select
              aria-label="How many reviews to score"
              className="field w-auto py-1 text-[13px]"
              value={rows}
              disabled={ablation.busy}
              onChange={(event) => setRows(Number(event.target.value))}
            >
              {JOB_SIZES.map((size) => (
                <option key={size.rows} value={size.rows}>
                  {size.label}
                </option>
              ))}
            </select>
            reviews
          </label>
        </div>
        {/* Said before the wait rather than discovered during it: each review goes through the
            model twice here, which is why this is slower than it looks for the same row count. */}
        <p className="mt-2 text-[13px] text-slate">
          Every review is scored twice - once by the mask, once by the full model on the same rows -
          so {rows.toLocaleString("en-US")} reviews is {(rows * 2).toLocaleString("en-US")} forward
          passes. The full sample takes several minutes on a laptop.
        </p>
        {kept === 0 && (
          <span className="mt-2 block text-[13px] text-slate">
            Keep at least one layer: an encoder with none is not a model.
          </span>
        )}
        {ablation.error !== null && (
          <p className="mt-3 text-[14px] text-accent">{ablation.error}</p>
        )}
      </section>

      {ablation.job !== null && (
        <section className="mt-6">
          <JobProgress job={ablation.job} />
        </section>
      )}

      {result && (
        <>
          <section className="mt-8">
            <h2 className="section-label">
              {result.n_layers} layers, over {result.rows_scanned.toLocaleString("en-US")} reviews
            </h2>
            <div className="mt-3 grid gap-4 sm:grid-cols-3">
              <MetricBlock
                label="Amputated, not retrained"
                value={formatAccuracyPrecise(result.ablated.accuracy)}
                interval={interval(result.ablated)}
                caption="This mask applied to the fine-tuned baseline, scored as-is."
                accent
              />
              <MetricBlock
                label="The same mask, trained"
                value={
                  Number.isNaN(twinAccuracy) ? "no such model" : formatAccuracyPrecise(twinAccuracy)
                }
                caption={
                  twin
                    ? `${twin.label}, trained on this mask and scored on the full ${twin.rows.toLocaleString("en-US")}-row split - a different set of reviews from the two columns beside it.`
                    : `Nobody has trained this mask. There are ${allMasks.toLocaleString("en-US")} of them and ${trainedMasks} trained ones, so most masks have no counterpart.`
                }
              />
              <MetricBlock
                label={`${result.baseline_label}, all 12 layers`}
                value={formatAccuracyPrecise(result.full.accuracy)}
                interval={interval(result.full)}
                caption="The same weights at full depth, scored in the same pass on the same reviews."
              />
            </div>

            <p className="mt-4 max-w-[68ch] text-[13px] leading-relaxed text-slate">
              {result.protocol} The amputated model still agrees with the full one on{" "}
              {formatAccuracyPrecise(result.agreement_with_full)} of these reviews, which is a
              different question from how often it is right.
            </p>
          </section>

          <section className="mt-6">
            <div className="panel p-4 text-[14px] leading-relaxed text-slate">{result.note}</div>
          </section>
        </>
      )}
    </main>
  );
}
