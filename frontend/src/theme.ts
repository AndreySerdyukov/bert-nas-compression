import { useCallback, useState } from "react";

export type Theme = "light" | "dark";

/**
 * Theme state, kept in step with the anti-flicker script in index.html.
 *
 * The initial value is read off the <html> class rather than from storage, so this hook and that
 * script can never disagree about what is currently on screen.
 */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  );

  const toggle = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === "dark" ? "light" : "dark";
      document.documentElement.classList.toggle("dark", next === "dark");
      try {
        localStorage.setItem("theme", next);
      } catch {
        /* private mode: the toggle still works, it just does not persist */
      }
      return next;
    });
  }, []);

  return [theme, toggle];
}
