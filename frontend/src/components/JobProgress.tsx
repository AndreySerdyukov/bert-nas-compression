import type { JobSnapshot } from "../api";

/**
 * A running job, drawn from one snapshot.
 *
 * The stage name is the part that matters. A bar reading "1400 / 2000" tells a reader how long they
 * have left; one reading "scoring with BANANAS - 1400 / 2000" tells them what the machine is doing,
 * and on a run of several minutes that is the difference between waiting and wondering whether it
 * has hung.
 */
export default function JobProgress({ job }: { job: JobSnapshot }) {
  const fraction = job.total > 0 ? Math.min(job.processed / job.total, 1) : 0;
  const last = job.messages.at(-1);

  return (
    <div className="panel mt-6 p-4" role="status" aria-live="polite">
      <div className="flex flex-wrap items-baseline gap-3">
        <p className="text-[15px] font-medium text-ink">{job.stage ?? "starting"}</p>
        <p className="tnum text-[13px] text-slate">
          stage {Math.min(job.stage_index + 1, job.stages.length)} of {job.stages.length}
        </p>
        <p className="tnum ml-auto text-[13px] text-slate">{job.elapsed_s.toFixed(0)} s</p>
      </div>

      <div aria-hidden="true" className="mt-3 h-1.5 overflow-hidden rounded-full bg-raised">
        <div
          className="h-full rounded-full bg-ink transition-[width] duration-200"
          style={{ width: `${(fraction * 100).toFixed(1)}%` }}
        />
      </div>

      <p className="tnum mt-2 text-[13px] text-slate">
        {job.total > 0
          ? `${job.processed.toLocaleString("en-US")} / ${job.total.toLocaleString("en-US")} reviews`
          : "working"}
      </p>

      {last !== undefined && <p className="mt-2 text-[13px] text-slate">{last}</p>}
    </div>
  );
}
