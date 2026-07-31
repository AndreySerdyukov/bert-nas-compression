import { useState } from "react";

import { searchSpaceSize } from "../../lib/arch";
import Figure from "./Figure";

/**
 * How big the search space is, and how long exhausting it would take.
 *
 * The point of the widget is the gap between two numbers that both come out of it: 3 797
 * architectures in the range the methods searched, and 3 candidates that Random Search actually
 * evaluated. Stating that in prose invites skimming. Making someone move the bounds and watch the
 * count move does not.
 */
export default function SearchSpaceCounter() {
  const [minLayers, setMinLayers] = useState(4);
  const [maxLayers, setMaxLayers] = useState(12);

  const low = Math.min(minLayers, maxLayers);
  const high = Math.max(minLayers, maxLayers);
  const size = searchSpaceSize(low, high);

  // Two minutes is roughly what one candidate cost in these searches: the Random Search log records
  // 123 seconds for a five-layer model on 2 800 rows.
  const minutesEach = 2;
  const days = (size * minutesEach) / 60 / 24;

  return (
    <Figure
      wide
      caption="Evaluating a candidate means training it, which is why exhaustive search is not on the table and why every one of these methods is a way of not looking at most of the space."
    >
      <div className="flex flex-col gap-4 font-sans">
        <div className="flex flex-wrap gap-6">
          <Bound label="keep at least" value={minLayers} onChange={setMinLayers} />
          <Bound label="keep at most" value={maxLayers} onChange={setMaxLayers} />
        </div>

        <div className="grid grid-cols-1 gap-4 border-t border-hair pt-4 sm:grid-cols-3">
          <div>
            <div className="section-label">architectures</div>
            <div className="tnum mt-1 text-[28px] font-semibold text-accent">
              {size.toLocaleString("en-US")}
            </div>
          </div>
          <div>
            <div className="section-label">exhaustive search</div>
            <div className="tnum mt-1 text-[28px] font-semibold text-ink">
              {days < 1 ? `${(days * 24).toFixed(1)} h` : `${days.toFixed(1)} days`}
            </div>
            <div className="text-[12px] text-slate">at {minutesEach} minutes per candidate</div>
          </div>
          <div>
            <div className="section-label">what was actually tried</div>
            <div className="tnum mt-1 text-[28px] font-semibold text-ink">3 to 13</div>
            <div className="text-[12px] text-slate">candidates, depending on the method</div>
          </div>
        </div>
      </div>
    </Figure>
  );
}

function Bound({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex items-center gap-3 text-[13px] text-slate">
      {label}
      <input
        type="range"
        min={0}
        max={12}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="accent-accent"
      />
      <span className="tnum w-6 text-right font-semibold text-ink">{value}</span>
    </label>
  );
}
