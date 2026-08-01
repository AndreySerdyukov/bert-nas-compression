/**
 * Chapter metadata: everything about a chapter except the chapter.
 *
 * Split out from `index.ts` because that file imports ten .mdx modules, and anything importing it
 * needs the MDX build plugin. The Playwright suite does not have one - it reads this list to know
 * what to visit - and a test suite that had to duplicate the chapter list would be a test suite
 * that could not notice a chapter going missing.
 */

export interface ChapterMeta {
  slug: string;
  number: number;
  title: string;
  /** One line, shown in the table of contents. Says what the chapter argues, not what it covers. */
  summary: string;
  minutes: number;
  /** Named so the contents page can promise the interaction rather than surprise you with it. */
  widget?: string;
}

export const CHAPTER_META: readonly ChapterMeta[] = [
  {
    slug: "the-problem",
    number: 1,
    title: "110 million parameters for a yes or no",
    summary:
      "What the model costs, what the task actually is, and why the gap between them is worth closing.",
    minutes: 4,
    widget: "the trade-off, as reported",
  },
  {
    slug: "what-nas-is",
    number: 2,
    title: "What NAS is",
    summary:
      "Search space, search strategy, performance estimation - and why the third one is where the money goes.",
    minutes: 5,
    widget: "search-space counter",
  },
  {
    slug: "the-search-space",
    number: 3,
    title: "The search space we actually used",
    summary: "Twelve bits, 3 797 architectures, and a cost function you can evaluate in your head.",
    minutes: 5,
    widget: "interactive layer mask",
  },
  {
    slug: "random-search",
    number: 4,
    title: "Random Search",
    summary: "The baseline that keeps winning, and the three draws it got here.",
    minutes: 5,
    widget: "search log",
  },
  {
    slug: "alphanas",
    number: 5,
    title: "AlphaNAS",
    summary: "Evolution over the mask, and a seed that could never survive its own constraint.",
    minutes: 5,
    widget: "search log",
  },
  {
    slug: "bananas",
    number: 6,
    title: "BANANAS",
    summary:
      "A surrogate model, Thompson sampling, and what happens when the signal is smaller than the noise.",
    minutes: 6,
    widget: "search log",
  },
  {
    slug: "adabert",
    number: 7,
    title: "AdaBERT and differentiable NAS",
    summary: "Learning the architecture by gradient descent, and the degenerate optimum it found.",
    minutes: 6,
  },
  {
    slug: "measuring-honestly",
    number: 8,
    title: "Measuring honestly",
    summary:
      "Warm-up, synchronisation, medians, and the difference between latency and throughput.",
    minutes: 5,
  },
  {
    slug: "what-did-not-reproduce",
    number: 9,
    title: "What did not reproduce",
    summary: "Four tables, one test set, and ten specific disagreements between them.",
    minutes: 7,
  },
  {
    slug: "what-i-would-do-differently",
    number: 10,
    title: "What I would do differently",
    summary: "The controls that were missing, and what they said once they were run.",
    minutes: 6,
  },
] as const;

export const TOTAL_MINUTES = CHAPTER_META.reduce((total, chapter) => total + chapter.minutes, 0);
