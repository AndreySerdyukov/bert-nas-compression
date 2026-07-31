import { Link } from "react-router-dom";

import { CHAPTERS, TOTAL_MINUTES } from "../content";

/** The table of contents. Ten chapters, and none of them needs a model to be downloaded. */
export default function Methodology() {
  return (
    <div className="mx-auto max-w-content px-6 py-14">
      <header className="max-w-prose">
        <p className="section-label">Methodology</p>
        <h1 className="mt-2 text-[34px] font-semibold leading-tight tracking-tight">
          Neural Architecture Search, and what it bought here
        </h1>
        <p className="mt-4 font-serif text-[18px] leading-[1.7] text-slate">
          Ten chapters on how a 110-million-parameter classifier was cut to four encoder layers,
          what each of the four methods does, and how far the results hold up. Roughly{" "}
          {TOTAL_MINUTES} minutes end to end, and the interactive parts are the argument rather than
          an illustration of it.
        </p>
      </header>

      <ol className="mt-12 border-t border-hair">
        {CHAPTERS.map((chapter) => (
          <li key={chapter.slug} className="border-b border-hair">
            <Link
              to={`/methodology/${chapter.slug}`}
              className="group flex gap-5 py-5 transition-colors hover:bg-raised"
            >
              <span className="tnum w-8 shrink-0 pl-1 text-[14px] text-slate">
                {String(chapter.number).padStart(2, "0")}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-[17px] font-medium text-ink">{chapter.title}</span>
                <span className="mt-1 block max-w-prose text-[14px] leading-relaxed text-slate">
                  {chapter.summary}
                </span>
                {chapter.widget && (
                  <span className="mt-2 inline-block rounded-row border border-hair px-2 py-0.5 text-[11px] text-slate">
                    interactive: {chapter.widget}
                  </span>
                )}
              </span>
              <span className="tnum shrink-0 pr-1 text-[13px] text-slate">
                {chapter.minutes} min
              </span>
            </Link>
          </li>
        ))}
      </ol>
    </div>
  );
}
