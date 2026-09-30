import { useState } from "react";
import { useTheme } from "./ui";
import UsaPage from "./pages/UsaPage";
import IndiaPage from "./pages/IndiaPage";
import BenchmarkPage from "./pages/BenchmarkPage";

type Tab = "usa" | "india" | "benchmark";
const TABS: { key: Tab; label: string }[] = [
  { key: "usa", label: "USA · Real observations" },
  { key: "benchmark", label: "Detection benchmark" },
  { key: "india", label: "India · Synthetic scenarios" },
];

export default function App() {
  const [mode, palette, toggle] = useTheme();
  const [tab, setTab] = useState<Tab>("usa");
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">SkyGuard AI<small>SIH26073 · AWS anomaly detection</small></div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={"tab" + (tab === t.key ? " active" : "")} onClick={() => setTab(t.key)}>{t.label}</button>
          ))}
        </nav>
        <div className="spacer" />
        <button className="theme-toggle" title="Toggle light / dark" onClick={toggle}>{mode === "dark" ? "☀" : "☾"}</button>
      </header>
      {tab === "usa" && <UsaPage p={palette} />}
      {tab === "benchmark" && <BenchmarkPage p={palette} />}
      {tab === "india" && <IndiaPage p={palette} />}
    </div>
  );
}
