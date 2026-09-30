import { useEffect, useMemo, useState } from "react";
import { PALETTES, type Mode, type Palette } from "./palette";

export function useTheme(): [Mode, Palette, () => void] {
  const [mode, setMode] = useState<Mode>(() => {
    const saved = localStorage.getItem("skyguard-theme") as Mode | null;
    if (saved) return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    localStorage.setItem("skyguard-theme", mode);
  }, [mode]);
  const palette = useMemo(() => PALETTES[mode], [mode]);
  return [mode, palette, () => setMode((m) => (m === "dark" ? "light" : "dark"))];
}

export function useAsync<T>(fn: () => Promise<T>, deps: any[]): { data?: T; error?: string; loading: boolean } {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    fn().then(
      (data) => alive && setState({ data, loading: false }),
      (err) => alive && setState({ error: String(err), loading: false }),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function MetricTile({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={"value" + (accent ? " accent" : "")}>{value}</div>
    </div>
  );
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


export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
    </div>
  );
}

export interface Opt { value: string; label: string; }

export function Select({ label, value, onChange, options, disabled }: {
  label: string; value: string; onChange: (v: string) => void; options: Opt[]; disabled?: boolean;
}) {
  return (
    <Field label={label}>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
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
      <div className="seg">
        {options.map((o) => (
          <button key={o.value} className={o.value === value ? "active" : ""} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </Field>
  );
}
