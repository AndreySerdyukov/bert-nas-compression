/**
 * How many rows of the evaluation sample a long-running job covers.
 *
 * Shared by the two pages that start one, because the choice is the same choice and the numbers
 * have to agree with `APP_MAX_EVAL_SAMPLE` on the server - a button offering more rows than the
 * backend will accept is a 422 the reader cannot act on.
 *
 * Offering the choice at all is the point. Both jobs run for minutes on a laptop, and a page with
 * a single button that says nothing about the wait leaves the reader unable to tell a long job
 * from a hung one. The explorer had exactly that: one button, no size, and a default of the full
 * sample scored twice.
 */

/** The whole committed sample. The server's own cap, so nothing above this is offered. */
export const FULL_SAMPLE = 2000;

export interface JobSize {
  rows: number;
  label: string;
}

/** Largest first: the full sample is the number that counts, and the rest are for a quick look. */
export const JOB_SIZES: readonly JobSize[] = [
  { rows: FULL_SAMPLE, label: "All 2,000" },
  { rows: 500, label: "First 500" },
  { rows: 200, label: "First 200" },
];
