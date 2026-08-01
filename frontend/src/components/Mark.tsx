/**
 * The brand mark: an encoder stack with two of its four layers kept.
 *
 * Four bars rather than twelve. At the 18px this renders at in the header, twelve cells are a
 * smudge, and the mark has to survive being small far more than it has to be literal. The filled
 * pair sits at the top because the controls found that the bottom of a BERT stack is where the
 * useful layers are - `first-k` beat the search at both depths - so the mark carries the finding
 * rather than an arbitrary pattern.
 *
 * Stroked with `currentColor`, so the header, the light theme and the dark one are one file.
 * The source lives at `nas-design/mark.svg`.
 */
export default function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="shrink-0"
    >
      <rect
        x="4.75"
        y="3.75"
        width="14.5"
        height="3.2"
        rx="1.1"
        fill="currentColor"
        stroke="none"
      />
      <rect x="4.75" y="8.6" width="14.5" height="3.2" rx="1.1" fill="currentColor" stroke="none" />
      <rect x="4.75" y="13.45" width="14.5" height="3.2" rx="1.1" />
      <rect x="4.75" y="18.3" width="14.5" height="3.2" rx="1.1" />
    </svg>
  );
}
