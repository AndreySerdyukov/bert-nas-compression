# nas-design

Source SVGs for the brand mark and the icon set, mirroring `ml-design/` and `dl-design/` in the
sibling projects.

- `mark.svg` - the brand mark: an encoder stack with two of its four layers kept. Monochrome,
  `currentColor`, so one file serves both themes. Rendered in the app by
  `frontend/src/components/Mark.tsx`, and reduced for the tab at
  `frontend/public/favicon.svg`.

The icon set is not drawn yet. `docs/DESIGN_PROMPT.md` is the prompt that produces it, keyed to the
names the components will ask for.
