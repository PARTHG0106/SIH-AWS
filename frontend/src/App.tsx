import { useState } from "react";
import { useTheme } from "./ui";
import UsaPage from "./pages/UsaPage";
import IndiaPage from "./pages/IndiaPage";

export default function App() {
  const [mode, palette, toggle] = useTheme();
  const [tab, setTab] = useState<"usa" | "india">("usa");
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">SkyGuard AI<small>SIH26073 · AWS anomaly detection</small></div>
        <nav className="tabs">
          <button className={"tab" + (tab === "usa" ? " active" : "")} onClick={() => setTab("usa")}>USA · Real observations</button>
          <button className={"tab" + (tab === "india" ? " active" : "")} onClick={() => setTab("india")}>India · Synthetic scenarios</button>
        </nav>
        <div className="spacer" />
        <button className="theme-toggle" title="Toggle light / dark" onClick={toggle}>{mode === "dark" ? "☀" : "☾"}</button>
      </header>
      {tab === "usa" ? <UsaPage p={palette} /> : <IndiaPage p={palette} />}
    </div>
  );
}
