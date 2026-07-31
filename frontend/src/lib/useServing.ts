import { useCallback, useEffect, useState } from "react";

import { fetchExamples, fetchModels, type ExamplesResponse, type ModelsResponse } from "../api";

/**
 * Loaders for the serving side.
 *
 * Deliberately not cached at module scope the way the methodology documents are. Those are static
 * files; this is the state of a registry that depends on whether someone has run the fetch script,
 * and a catalog cached across a reload would be the wrong kind of stale. The cost is one request
 * per visit to the page, which is a few hundred bytes.
 */

export interface Async<T> {
  data: T | null;
  /** The status code, so a page can tell "serving is off" (503) from "the backend is down". */
  status: number | null;
  error: string | null;
  loading: boolean;
}

function useAsync<T>(loader: () => Promise<T>): Async<T> {
  const [state, setState] = useState<Async<T>>({
    data: null,
    status: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let alive = true;
    loader()
      .then((data) => {
        if (alive) setState({ data, status: 200, error: null, loading: false });
      })
      .catch((error: unknown) => {
        if (!alive) return;
        const status = error instanceof Error && "status" in error ? Number(error.status) : null;
        setState({
          data: null,
          status,
          error: error instanceof Error ? error.message : String(error),
          loading: false,
        });
      });
    return () => {
      alive = false;
    };
  }, [loader]);

  return state;
}

export function useModels(): Async<ModelsResponse> {
  return useAsync(useCallback(() => fetchModels(), []));
}

export function useExamples(): Async<ExamplesResponse> {
  return useAsync(useCallback(() => fetchExamples(), []));
}
