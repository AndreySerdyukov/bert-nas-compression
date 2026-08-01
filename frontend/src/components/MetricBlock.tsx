import type { ReactNode } from "react";

/**
 * A measured figure: a label, the number, optionally the interval it sits inside, and a caption.
 *
 * The most repeated element in the app, and it was written twice - once on the overview, once per
 * column in the explorer - before it was extracted. The two copies had already drifted apart in
 * type size by the time this file existed, which is the usual way a design system stops being one.
 *
 * The interval is a separate slot rather than part of the caption on purpose. On 2 000 reviews a
 * Wilson interval is over a point wide, which is the same order as the gaps a reader will try to
 * read off the page, so it has to sit against the number rather than in prose underneath it.
 */
export interface MetricBlockProps {
  label: string;
  value: string;
  /** Sits directly under the number: a Wilson interval, a range, a delta. */
  interval?: ReactNode;
  caption?: ReactNode;
  /** Reserved for the figure the section is about, never for every figure in it. */
  accent?: boolean;
}

export default function MetricBlock({ label, value, interval, caption, accent }: MetricBlockProps) {
  return (
    <div className="panel p-4">
      <div className="section-label">{label}</div>
      <div
        className={`mt-2 text-[26px] font-semibold tabular-nums tracking-tight ${accent ? "text-accent" : ""}`}
      >
        {value}
      </div>
      {interval !== undefined && <div className="mt-0.5 text-[12px] text-slate">{interval}</div>}
      {caption !== undefined && (
        <p className="mt-2 text-[13px] leading-snug text-slate">{caption}</p>
      )}
    </div>
  );
}
