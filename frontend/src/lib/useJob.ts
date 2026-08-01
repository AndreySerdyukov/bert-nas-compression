import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchJob, jobEventsUrl, startJob, type JobSnapshot } from "../api";

/**
 * Start a job and follow it.
 *
 * The stream is the fast path and polling is the fallback, and neither is the source of truth: both
 * carry the same whole snapshot, so switching between them mid-job changes nothing about what is
 * rendered. That is the reason the server sends snapshots rather than deltas - a dropped frame
 * would otherwise be a permanent hole in the progress bar.
 *
 * `EventSource` reconnects on its own when a connection closes, which is unhelpful once the job has
 * finished, so the terminal snapshot closes it from this side.
 */
export interface JobState {
  job: JobSnapshot | null;
  error: string | null;
  /** True from the moment the request to start is sent until a terminal snapshot arrives. */
  busy: boolean;
}

const POLL_MS = 1000;

/** What a caller asks for. `kind` defaults to the scan, which is what the playground wants. */
export interface StartOptions {
  kind?: "disagreement" | "ablation";
  limit?: number;
  layers?: number[];
}

export function useJob(): JobState & { start: (body: StartOptions) => Promise<void> } {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const source = useRef<EventSource | null>(null);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    source.current?.close();
    source.current = null;
    if (poll.current !== null) clearInterval(poll.current);
    poll.current = null;
  }, []);

  // A job outlives the component only if nobody hangs up, so leaving the page hangs up.
  useEffect(() => stop, [stop]);

  const accept = useCallback(
    (snapshot: JobSnapshot) => {
      setJob(snapshot);
      if (snapshot.status !== "running") {
        stop();
        setBusy(false);
        if (snapshot.status === "failed") setError(snapshot.error ?? "the job failed");
      }
    },
    [stop],
  );

  const follow = useCallback(
    (id: string) => {
      const events = new EventSource(jobEventsUrl(id));
      source.current = events;
      events.addEventListener("update", (event) => {
        accept(JSON.parse((event as MessageEvent<string>).data) as JobSnapshot);
      });
      events.onerror = () => {
        // The stream is gone. Rather than guess whether it will come back, hang up and ask.
        events.close();
        source.current = null;
        if (poll.current === null) {
          poll.current = setInterval(() => {
            fetchJob(id)
              .then(accept)
              .catch(() => {
                /* the next tick tries again; a terminal snapshot clears the interval */
              });
          }, POLL_MS);
        }
      };
    },
    [accept],
  );

  const start = useCallback(
    async ({ kind = "disagreement", ...body }: StartOptions) => {
      stop();
      setError(null);
      setJob(null);
      setBusy(true);
      try {
        const started = await startJob({ kind, ...body });
        setJob(started);
        follow(started.id);
      } catch (caught: unknown) {
        setBusy(false);
        setError(caught instanceof ApiError ? caught.message : String(caught));
      }
    },
    [follow, stop],
  );

  return { job, error, busy, start };
}
