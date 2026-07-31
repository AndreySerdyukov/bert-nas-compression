import { useMemo, useState } from "react";

import LayerMask from "../../components/LayerMask";
import { formatBytes, formatFlops, formatParams } from "../../components/charts/scale";
import {
  describe,
  maskFromLayers,
  matchesKnown,
  PRESETS,
  type LayerMask as Mask,
} from "../../lib/arch";
import { useArchitectures } from "../../lib/useContent";
import Figure from "./Figure";

/**
 * Toggle layers, watch the cost move. The chapter-3 widget, and the explorer's core in miniature.
 *
 * Every number here is arithmetic - `describe()` runs locally and nothing is fetched when a layer
 * is clicked. That is not a performance trick, it is the chapter's argument made tangible: the cost
 * of an architecture is a closed form in the mask, while its quality needs a training run. Cost
 * commutes with masking; quality does not.
 *
 * The one thing that *is* fetched, once, is the list of shipped masks - so the widget can say "this
 * is what AlphaNAS shipped" when you happen to land on it.
 */
export default function MaskCost({
  initialLayers = [0, 1, 5, 7, 9],
}: {
  initialLayers?: number[];
}) {
  const [mask, setMask] = useState<Mask>(() => maskFromLayers(initialLayers));
  const [seqLen, setSeqLen] = useState(256);
  const { data: architectures } = useArchitectures();

  const cost = useMemo(() => describe(mask, seqLen), [mask, seqLen]);

  const known = useMemo(() => {
    const entries: Record<string, Mask | null> = {};
    for (const [name, method] of Object.entries(architectures?.methods ?? {})) {
      entries[name] = method.shipped.mask;
    }
    return entries;
  }, [architectures]);

  const match = useMemo(() => matchesKnown(mask, known), [mask, known]);
  const label = match ? (architectures?.methods[match]?.label ?? match) : null;

  const toggle = (index: number) =>
    setMask((current) => current.map((bit, i) => (i === index ? (bit ? 0 : 1) : bit)));

  return (
    <Figure
      wide
      caption="Nothing here is measured and nothing is fetched when you click. Parameters, FLOPs and footprint follow from the mask alone, which is why they can be instant - and why the other half of the trade cannot."
    >
      <div className="flex flex-col gap-4 font-sans">
        <div className="flex flex-wrap items-center gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => setMask(maskFromLayers(preset.layers))}
              className="rounded-row border border-hair px-2 py-1 text-[12px] text-slate transition-colors hover:border-accent hover:text-ink"
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div>
          <LayerMask mask={mask} onToggle={toggle} size={34} showIndices />
          <p className="mt-2 text-[12px] text-slate">
            Click a layer, or use the arrow keys and space. Layer 0 is closest to the embeddings.
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
          <Stat label="layers kept" value={`${cost.nLayers} of 12`} />
          <Stat
            label="parameters"
            value={formatParams(cost.params)}
            hint={`${(cost.paramsPctOfBase * 100).toFixed(1)}% of baseline`}
            accent
          />
          <Stat label="fp32 on disk" value={formatBytes(cost.bytesFp32)} />
          <Stat
            label={`FLOPs at ${seqLen} tokens`}
            value={formatFlops(cost.flopsPerExample)}
            hint="forward pass, one example"
          />
        </dl>

        <div className="flex flex-wrap items-center gap-3 border-t border-hair pt-3 text-[13px]">
          <span className="text-slate">Sequence length</span>
          {[128, 256, 512].map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setSeqLen(value)}
              aria-pressed={seqLen === value}
              className={`rounded-row px-2 py-1 text-[12px] transition-colors ${
                seqLen === value
                  ? "bg-ink text-canvas"
                  : "text-slate hover:bg-raised hover:text-ink"
              }`}
            >
              {value}
            </button>
          ))}
          <span className="text-slate">
            {label ? (
              <>
                This is the architecture <strong className="text-ink">{label}</strong> shipped
              </>
            ) : (
              "Not one of the four shipped architectures"
            )}
          </span>
        </div>
      </div>
    </Figure>
  );
}

function Stat({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div>
      <dt className="section-label">{label}</dt>
      <dd className={`tnum mt-1 text-[20px] font-semibold ${accent ? "text-accent" : "text-ink"}`}>
        {value}
      </dd>
      {hint && <dd className="tnum text-[12px] text-slate">{hint}</dd>}
    </div>
  );
}
