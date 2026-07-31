/// <reference types="vite/client" />

// MDX modules compile to React components at build time; without this they are untyped imports
// and the chapter registry loses its type checking.
declare module "*.mdx" {
  import type { ComponentType } from "react";
  const MDXComponent: ComponentType<Record<string, unknown>>;
  export default MDXComponent;
}
