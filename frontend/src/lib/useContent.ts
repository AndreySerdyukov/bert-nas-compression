import { useEffect, useState } from "react";

import {
  fetchArchitectures,
  fetchReportedResults,
  fetchTrajectories,
  type Architectures,
  type ReportedResults,
  type Trajectories,
} from "../api";

/**
 * Loaders for the committed documents, cached at module scope.
 *
 * A methodology chapter can hold several widgets and they all want the same trajectories file.
 * Caching the *promise* rather than the result means concurrent mounts share one request instead of
 * racing, and a second visit to the chapter costs nothing. The documents are static - they change
 * only when the notebooks do - so there is no invalidation to get wrong.
 */

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

function cached<T>(loader: () => Promise<T>): () => Promise<T> {
  let promise: Promise<T> | null = null;
  return () => {
    // A failed request is not cached: a backend that was not running yet should be retried.
    promise ??= loader().catch((error: unknown) => {
      promise = null;
      throw error;
    });
    return promise;
  };
}

const loadArchitectures = cached(fetchArchitectures);
const loadTrajectories = cached(fetchTrajectories);
const loadReportedResults = cached(fetchReportedResults);

function useAsync<T>(loader: () => Promise<T>): Async<T> {
  const [state, setState] = useState<Async<T>>({ data: null, error: null, loading: true });

  useEffect(() => {
    let alive = true;
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error: unknown) =>
        alive
          ? setState({
              data: null,
              error: error instanceof Error ? error.message : String(error),
              loading: false,
            })
          : undefined,
      );
    return () => {
      alive = false;
    };
  }, [loader]);

  return state;
}

export const useArchitectures = (): Async<Architectures> => useAsync(loadArchitectures);
export const useTrajectories = (): Async<Trajectories> => useAsync(loadTrajectories);
export const useReportedResults = (): Async<ReportedResults> => useAsync(loadReportedResults);
