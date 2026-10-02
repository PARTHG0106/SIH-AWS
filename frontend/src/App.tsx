import { useState } from "react";
import { useTheme } from "./ui";
import UsaPage from "./pages/UsaPage";
import IndiaPage from "./pages/IndiaPage";
import BenchmarkPage from "./pages/BenchmarkPage";
import LivePage from "./pages/LivePage";

type Tab = "live" | "usa" | "india" | "benchmark";
const TABS: { key: Tab; label: string }[] = [
  { key: "live", label: "Live detection" },
  { key: "usa", label: "USA · Real observations" },
  { key: "benchmark", label: "Detection benchmark" },
  { key: "india", label: "India · Synthetic scenarios" },
];

export default function App() {
  const [mode, palette, toggle] = useTheme();
  const [tab, setTab] = useState<Tab>("live");
  return (
    <div className="app">
      <a className="skip-link" href="#workspace">Skip to content</a>
      <header className="topbar">
        <div className="brand">SkyGuard AI<small>SIH26073 · AWS anomaly detection</small></div>
        <nav className="tabs" aria-label="Dashboard pages">
          {TABS.map((t) => (
            <button key={t.key} type="button" aria-current={tab === t.key ? "page" : undefined} className={"tab" + (tab === t.key ? " active" : "")} onClick={() => setTab(t.key)}>{t.label}</button>
          ))}
        </nav>
        <div className="spacer" />
        <button className="theme-toggle" type="button" aria-label={`Switch to ${mode === "dark" ? "light" : "dark"} theme`} title="Toggle light / dark" onClick={toggle}>{mode === "dark" ? "☀" : "☾"}</button>
      </header>
      <div id="workspace" tabIndex={-1}>
      {tab === "live" && <LivePage p={palette} />}
      {tab === "usa" && <UsaPage p={palette} />}
      {tab === "benchmark" && <BenchmarkPage p={palette} />}
      {tab === "india" && <IndiaPage p={palette} />}
      </div>
    </div>
  );
}
