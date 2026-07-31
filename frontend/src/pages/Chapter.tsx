import { useEffect, useRef, useState } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";

import { chapterBySlug, neighbours } from "../content";

interface Heading {
  id: string;
  text: string;
}

/**
 * The chapter's own section list, read off the rendered article.
 *
 * Derived from the DOM rather than declared in the MDX, because the alternative is a list of
 * headings maintained next to the headings themselves - two places to edit and one of them silently
 * wrong. The ids are assigned here too, so an anchor cannot drift from its heading.
 */
function useHeadings(container: React.RefObject<HTMLElement | null>, slug: string): Heading[] {
  const [headings, setHeadings] = useState<Heading[]>([]);

  useEffect(() => {
    const article = container.current;
    if (!article) return;
    const found = Array.from(article.querySelectorAll("h2")).map((element, index) => {
      const text = element.textContent ?? `Section ${index + 1}`;
      const id =
        element.id ||
        text
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-|-$/g, "");
      element.id = id;
      // Sticky header height, so an anchored heading is not hidden under the nav.
      element.style.scrollMarginTop = "5rem";
      return { id, text };
    });
    // Setting state from an effect is normally a smell, and here it is the only option: the
    // headings do not exist until the compiled MDX has rendered, so they cannot be derived during
    // render. The effect runs once per chapter and writes a list whose contents are a pure function
    // of the DOM it just read, so there is no loop to fall into.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHeadings(found);
  }, [container, slug]);

  return headings;
}

/**
 * One chapter: the compiled MDX, framed, with the way out at both ends.
 *
 * `j` and `k` move between chapters. That is not a flourish - the section is meant to be read
 * straight through, and reaching for the mouse at every chapter boundary is what stops people.
 */
export default function Chapter() {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const chapter = chapterBySlug(slug);
  const { previous, next } = neighbours(slug ?? "");
  const articleRef = useRef<HTMLElement>(null);
  const headings = useHeadings(articleRef, slug ?? "");

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // Ignore the shortcut while someone is typing, and while a modifier is held.
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "j" && next) navigate(`/methodology/${next.slug}`);
      if (event.key === "k" && previous) navigate(`/methodology/${previous.slug}`);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [navigate, next, previous]);

  if (!chapter) return <Navigate to="/methodology" replace />;

  const { Component } = chapter;

  return (
    <div className="mx-auto max-w-content px-6 py-14">
      <Link to="/methodology" className="text-[13px] text-slate hover:text-ink">
        ← All chapters
      </Link>

      <header className="mt-6 max-w-prose">
        <p className="section-label">
          Chapter {String(chapter.number).padStart(2, "0")} · {chapter.minutes} min
        </p>
        <h1 className="mt-2 text-[32px] font-semibold leading-tight tracking-tight">
          {chapter.title}
        </h1>
      </header>

      <div className="mt-10 gap-12 lg:flex">
        <article ref={articleRef} className="prose-nas min-w-0 flex-1">
          <Component />
        </article>

        {headings.length > 1 && (
          <nav aria-label="Sections in this chapter" className="hidden w-52 shrink-0 lg:block">
            <div className="sticky top-20">
              <p className="section-label">In this chapter</p>
              <ul className="mt-3 space-y-2 border-l border-hair">
                {headings.map((heading) => (
                  <li key={heading.id}>
                    <a
                      href={`#${heading.id}`}
                      className="-ml-px block border-l border-transparent pl-3 text-[13px] leading-snug text-slate transition-colors hover:border-accent hover:text-ink"
                    >
                      {heading.text}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </nav>
        )}
      </div>

      <nav className="mt-16 flex justify-between gap-6 border-t border-hair pt-6 text-[14px]">
        {previous ? (
          <Link to={`/methodology/${previous.slug}`} className="group max-w-[45%]">
            <span className="block text-[12px] text-slate">
              Previous <kbd className="kbd ml-1">k</kbd>
            </span>
            <span className="mt-1 block text-ink group-hover:text-accent">{previous.title}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link to={`/methodology/${next.slug}`} className="group max-w-[45%] text-right">
            <span className="block text-[12px] text-slate">
              Next <kbd className="kbd ml-1">j</kbd>
            </span>
            <span className="mt-1 block text-ink group-hover:text-accent">{next.title}</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </div>
  );
}
