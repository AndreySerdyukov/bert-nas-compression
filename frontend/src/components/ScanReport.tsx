import type { ScanResult } from "../api";
import { formatParams, formatPercent } from "../lib/format";

/**
 * What the disagreement scan found.
 *
 * The table is the summary and the list underneath is the argument. A reader who only reads the
 * table learns that two models differ by a couple of points; a reader who scrolls sees the actual
 * reviews they differ on, which is the thing an accuracy figure cannot convey.
 *
 * Nothing here is a timing. The scan runs unpinned and batched precisely because it is not one, and
 * a millisecond figure taken under those conditions would not be comparable to anything else the
 * application publishes.
 */
export default function ScanReport({ result }: { result: ScanResult }) {
  const capped = result.disagreements_found > result.disagreements_shown;

  return (
    <div className="mt-6">
      <h3 className="text-[18px] font-semibold tracking-tight">
        {result.disagreements_found === 0
          ? `All ${result.rows_scanned.toLocaleString("en-US")} reviews got the same verdict from every model`
          : `${result.disagreements_found.toLocaleString("en-US")} of ${result.rows_scanned.toLocaleString("en-US")} reviews split the models`}
      </h3>
      <p className="mt-1 text-[13px] text-slate">
        Scored on {result.threads_used} threads on {result.machine}. This is a count of
        disagreements, not a measurement of speed, so the thread pin the latency figures rely on was
        off for it.
      </p>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[560px] text-[14px]">
          <thead>
            <tr className="border-b border-hair text-left text-slate">
              <th className="py-2 pr-3 font-medium">Model</th>
              <th className="py-2 pr-3 text-right font-medium">Params</th>
              <th className="py-2 pr-3 text-right font-medium">
                Accuracy on these {result.rows_scanned.toLocaleString("en-US")}
              </th>
              <th className="py-2 text-right font-medium">Agrees with the baseline</th>
            </tr>
          </thead>
          <tbody className="tnum">
            {result.models.map((model) => (
              <tr key={model.name} className="border-b border-hair" data-scan-model={model.name}>
                <td className="py-2 pr-3">
                  {model.label}
                  {model.name === result.baseline && (
                    <span className="ml-2 text-[12px] text-slate">baseline</span>
                  )}
                </td>
                <td className="py-2 pr-3 text-right text-slate">{formatParams(model.params)}</td>
                <td className="py-2 pr-3 text-right">{formatPercent(model.accuracy, 2)}</td>
                <td className="py-2 text-right">
                  {model.agreement === null ? (
                    <span className="text-slate">–</span>
                  ) : (
                    formatPercent(model.agreement, 2)
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 max-w-prose text-[13px] leading-relaxed text-slate">
        Accuracy here is over the {result.rows_scanned.toLocaleString("en-US")} rows just scanned,
        not the full 15 000-row test split. The published benchmark is the one from
        <code className="mx-1 font-mono">training/benchmark.py</code>, and this is not it.
      </p>

      {result.disagreements.length > 0 && (
        <section className="mt-8">
          <h4 className="section-label">
            {capped
              ? `${result.disagreements_shown} of the ${result.disagreements_found} disagreeing reviews`
              : "The disagreeing reviews"}
          </h4>
          {capped && (
            <p className="mt-1 text-[13px] text-slate">
              The rest are not shown; a cap that went unmentioned would read as though these were
              all of them.
            </p>
          )}
          {/* The accent means something different here than on the cards above, so it is spelled
              out. Up there it marks a model parting from the baseline; here the corpus label is
              known for every row, so what is worth marking is who got it wrong - including the
              baseline, which is a reference and not ground truth. */}
          <p className="mt-1 text-[13px] text-slate">
            A marked verdict is one the corpus disagrees with. The baseline can be marked too: it is
            the reference these models were compressed from, not the truth.
          </p>
          <ul className="mt-3 space-y-3">
            {result.disagreements.map((row) => (
              <li key={row.id} className="panel p-4" data-disagreement={row.id}>
                <div className="flex flex-wrap items-baseline gap-3 text-[13px]">
                  <span className="tnum font-mono text-slate">#{row.id}</span>
                  <span className="text-slate">
                    corpus label: <span className="text-ink">{row.label}</span>
                  </span>
                  <span className="ml-auto flex flex-wrap gap-2">
                    {Object.entries(row.verdicts).map(([name, verdict]) => (
                      <span
                        key={name}
                        className={`rounded-row border px-1.5 py-0.5 text-[12px] ${
                          verdict === row.label
                            ? "border-hair text-slate"
                            : "border-accent/50 bg-accent/10 text-accent"
                        }`}
                      >
                        {name}: {verdict}
                      </span>
                    ))}
                  </span>
                </div>
                <p className="mt-2 font-serif text-[15px] leading-relaxed text-ink">
                  {row.excerpt}
                  {row.truncated && <span className="text-slate"> … </span>}
                </p>
                {row.truncated && (
                  <p className="mt-1 text-[12px] text-slate">
                    Excerpt of {row.n_chars.toLocaleString("en-US")} characters.
                  </p>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
