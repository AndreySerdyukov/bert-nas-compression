/**
 * Assert the two claims the frontend makes about its own weight.
 *
 * 1. The runtime dependency list is exactly three packages.
 * 2. No MDX runtime reached the bundle.
 *
 * The second is the one worth automating. MDX is configured to compile at build time with
 * `jsxImportSource: "react"` and no `providerImportSource`, so the emitted modules import nothing
 * but `react/jsx-runtime`. Adding `providerImportSource` - the line every MDX tutorial starts with -
 * would silently add a runtime package and nothing else would complain.
 *
 *     node scripts/check-runtime-deps.mjs
 */
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const EXPECTED = ["react", "react-dom", "react-router-dom"];
const FORBIDDEN_IN_BUNDLE = ["@mdx-js", "MDXProvider"];

const failures = [];

const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const actual = Object.keys(pkg.dependencies ?? {}).sort();
if (actual.join(",") !== EXPECTED.join(",")) {
  failures.push(`runtime dependencies are [${actual}], expected [${EXPECTED}]`);
}

const assets = join(root, "dist", "assets");
if (!existsSync(assets)) {
  console.log("dependencies: ok (3 runtime packages); dist/ absent, skipping the bundle scan");
  if (failures.length) {
    console.error(failures.join("\n"));
    process.exit(1);
  }
  process.exit(0);
}

for (const file of readdirSync(assets).filter((name) => name.endsWith(".js"))) {
  const source = readFileSync(join(assets, file), "utf8");
  for (const marker of FORBIDDEN_IN_BUNDLE) {
    if (source.includes(marker)) failures.push(`${file} contains ${marker}`);
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`dependencies: ok (${actual.length} runtime packages, no MDX runtime in the bundle)`);
