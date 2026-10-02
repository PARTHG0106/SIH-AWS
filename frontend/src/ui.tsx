import { useEffect, useId, useMemo, useState, type DependencyList } from "react";
import { PALETTES, type Mode, type Palette } from "./palette";
import type { MaintenanceAdvisory } from "./api";

export function useTheme(): [Mode, Palette, () => void] {
  const [mode, setMode] = useState<Mode>(() => {
    const saved = localStorage.getItem("skyguard-theme") as Mode | null;
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    localStorage.setItem("skyguard-theme", mode);
  }, [mode]);
  const palette = useMemo(() => PALETTES[mode], [mode]);
  return [mode, palette, () => setMode((m) => (m === "dark" ? "light" : "dark"))];
}

export function useAsync<T>(fn: () => Promise<T>, deps: DependencyList): { data?: T; error?: string; loading: boolean } {
  // Tie results to the exact request. A new station/window must never display
  // the previous request's records, including the render before the effect runs.
  const key = useMemo(() => ({}), deps);
  const [state, setState] = useState<{ key: object; data?: T; error?: string; loading: boolean }>({ key, loading: true });
  useEffect(() => {
    let alive = true;
    setState({ key, loading: true });
    fn().then(
      (data) => alive && setState({ key, data, loading: false }),
      (err: unknown) => alive && setState({ key, error: err instanceof Error ? err.message : String(err), loading: false }),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state.key === key ? state : { loading: true };
}

export function MetricTile({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={"value" + (accent ? " accent" : "")}>{value}</div>
    </div>
  );
}

export function MaintenanceCard({ advisory }: { advisory?: MaintenanceAdvisory }) {
  if (!advisory) return null;
  const urgent = advisory.priority === "review_now" || advisory.priority === "check_feed_now";
  return <section className="maintenance-card" aria-label="Maintenance review recommendation">
    <div className="section-heading"><h2>Maintenance review</h2>
      <span className={"severity" + (urgent ? " high" : "")}>{advisory.priority.replace(/_/g, " ")}</span></div>
    <p>{advisory.action}</p>
    <p className="caption">{advisory.channel?.replace(/_/g, " ")} · {advisory.evidence.scored_rows} scored of {advisory.evidence.received_rows} received rows
      {advisory.evidence.window_minutes == null ? "" : ` across ${advisory.evidence.window_minutes.toFixed(0)} minutes`}.
      {" "}{advisory.history_sufficient_for_policy ? "History requirement met." : `Trend policy needs at least ${advisory.policy.minimum_scored_rows} scored readings over ${advisory.policy.minimum_window_minutes} minutes.`}</p>
    <p className="caption">{advisory.semantics} Hardware condition remains unknown.</p>
  </section>;
}

export function Banner({ kind = "info", children }: { kind?: "info" | "warn"; children: React.ReactNode }) {
  return <div className={`banner ${kind}`}>{children}</div>;
}

export function Accordion({ title, children, open }: { title: string; children: React.ReactNode; open?: boolean }) {
  return (
    <details className="accordion" open={open}>
      <summary>{title}</summary>
      <div className="body">{children}</div>
    </details>
  );
}
// __UI_MORE__


export function Field({ label, children, id }: { label: string; children: React.ReactNode; id?: string }) {
  return (
    <div className="field">
      {id ? <label htmlFor={id}>{label}</label> : <span className="field-label">{label}</span>}
      {children}
    </div>
  );
}

export interface Opt { value: string; label: string; }

export function Select({ label, value, onChange, options, disabled }: {
  label: string; value: string; onChange: (v: string) => void; options: Opt[]; disabled?: boolean;
}) {
  const id = useId();
  return (
    <Field label={label} id={id}>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </Field>
  );
}

export function Segmented({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void; options: Opt[];
}) {
  return (
    <Field label={label}>
      <div className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={o.value} type="button" aria-pressed={o.value === value} className={o.value === value ? "active" : ""} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </Field>
  );
}
