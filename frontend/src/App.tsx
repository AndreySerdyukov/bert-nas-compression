import { NavLink, Route, Routes } from "react-router-dom";

import Chapter from "./pages/Chapter";
import Methodology from "./pages/Methodology";
import Playground from "./pages/Playground";
import { useTheme } from "./theme";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/playground", label: "Playground" },
  { to: "/explorer", label: "Explorer" },
  { to: "/benchmark", label: "Benchmark" },
  { to: "/methodology", label: "Methodology" },
] as const;

function TopNav() {
  const [theme, toggleTheme] = useTheme();
  return (
    <header className="sticky top-0 z-20 border-b border-hair bg-canvas/85 backdrop-blur">
      <nav className="mx-auto flex h-14 max-w-content items-center gap-6 px-6">
        <span className="font-mono text-[13px] font-semibold tracking-tight">bert-nas</span>
        <ul className="flex flex-1 items-center gap-5">
          {NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={"end" in item ? item.end : undefined}
                className={({ isActive }) =>
                  `text-[14px] transition-colors ${isActive ? "text-ink" : "text-slate hover:text-ink"}`
                }
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={toggleTheme}
          className="rounded-row px-2 py-1 text-[13px] text-slate hover:bg-raised hover:text-ink"
          aria-label={`Switch to the ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </nav>
    </header>
  );
}

/** Placeholder for a page that later phases fill in. Named so the route list stays honest. */
function Stub({ title, note }: { title: string; note: string }) {
  return (
    <section className="mx-auto max-w-content px-6 py-16">
      <h1 className="text-[32px] font-semibold tracking-tight">{title}</h1>
      <p className="mt-3 max-w-prose text-[16px] text-slate">{note}</p>
    </section>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <TopNav />
      <main>
        <Routes>
          <Route
            path="/"
            element={
              <Stub
                title="BERT NAS Compression"
                note="What Neural Architecture Search actually buys when compressing a BERT sentiment classifier - measured, with the controls it needs."
              />
            }
          />
          <Route path="/playground" element={<Playground />} />
          <Route
            path="/explorer"
            element={<Stub title="Explorer" note="Pick any subset of the twelve encoder layers." />}
          />
          <Route
            path="/benchmark"
            element={<Stub title="Benchmark" note="Every number this project publishes." />}
          />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/methodology/:slug" element={<Chapter />} />
          <Route path="*" element={<Stub title="Not found" note="No such page." />} />
        </Routes>
      </main>
    </div>
  );
}
