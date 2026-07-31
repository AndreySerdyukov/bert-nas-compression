import type { ComponentType } from "react";

import Ch01 from "./01-the-problem.mdx";
import Ch02 from "./02-what-nas-is.mdx";
import Ch03 from "./03-the-search-space.mdx";
import Ch04 from "./04-random-search.mdx";
import Ch05 from "./05-alphanas.mdx";
import Ch06 from "./06-bananas.mdx";
import Ch07 from "./07-adabert.mdx";
import Ch08 from "./08-measuring-honestly.mdx";
import Ch09 from "./09-what-did-not-reproduce.mdx";
import Ch10 from "./10-what-i-would-do-differently.mdx";
import { CHAPTER_META, type ChapterMeta } from "./chapters";

/**
 * The chapter list: metadata from `chapters.ts` zipped with the compiled MDX.
 *
 * The two are separate so the browser tests can read the metadata without needing the MDX build
 * plugin. The order here is the reading order and is checked against the metadata at module load -
 * a chapter added to one list and not the other fails loudly rather than rendering blank.
 */

export type { ChapterMeta };
export { CHAPTER_META, TOTAL_MINUTES } from "./chapters";

export interface Chapter extends ChapterMeta {
  Component: ComponentType;
}

const COMPONENTS: readonly ComponentType[] = [
  Ch01,
  Ch02,
  Ch03,
  Ch04,
  Ch05,
  Ch06,
  Ch07,
  Ch08,
  Ch09,
  Ch10,
];

if (COMPONENTS.length !== CHAPTER_META.length) {
  throw new Error(
    `chapter metadata lists ${CHAPTER_META.length} entries but ${COMPONENTS.length} MDX modules are imported`,
  );
}

export const CHAPTERS: readonly Chapter[] = CHAPTER_META.map((meta, index) => ({
  ...meta,
  Component: COMPONENTS[index]!,
}));

export function chapterBySlug(slug: string | undefined): Chapter | undefined {
  return CHAPTERS.find((chapter) => chapter.slug === slug);
}

export function neighbours(slug: string): { previous?: Chapter; next?: Chapter } {
  const index = CHAPTERS.findIndex((chapter) => chapter.slug === slug);
  if (index < 0) return {};
  return { previous: CHAPTERS[index - 1], next: CHAPTERS[index + 1] };
}
