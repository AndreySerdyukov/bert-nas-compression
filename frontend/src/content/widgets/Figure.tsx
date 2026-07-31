import type { ReactNode } from "react";

/**
 * A figure inside a chapter.
 *
 * Prose is set to a 68-character measure, which is right for reading and far too narrow for a
 * chart. `wide` breaks the figure out of that measure while leaving the text where it belongs -
 * the one layout affordance the methodology section actually needs.
 */
export default function Figure({
  children,
  caption,
  wide = false,
}: {
  children: ReactNode;
  caption?: string;
  wide?: boolean;
}) {
  return (
    <figure className={`my-8 ${wide ? "max-w-none" : "max-w-prose"}`}>
      <div className="rounded-panel border border-hair bg-surface p-4">{children}</div>
      {caption && (
        <figcaption className="mt-2 max-w-prose font-sans text-[13px] leading-normal text-slate">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
