import { useEffect, useMemo, useState } from "react";
import type { Palette } from "../palette";
import { api } from "../api";
import { EChart, timeSeriesOption } from "../Chart";
import { Accordion, Banner, MetricTile, Segmented, Select, useAsync } from "../ui";

const iso = (ms: number) => new Date(ms).toISOString();

export default function UsaPage({ p }: { p: Palette }) {
  const cat = useAsync(() => api.usaCatalog(), []);
  const [group, setGroup] = useState("");
  const [mode, setMode] = useState<"event" | "calendar">("event");
  const [eventId, setEventId] = useState("");
  const [ctx, setCtx] = useState(60);
  const [day, setDay] = useState("");
  const [days, setDays] = useState(1);
  const [show60, setShow60] = useState(false);

  const stations = cat.data?.stations ?? [];
  useEffect(() => { if (!group && stations.length) setGroup(stations[0].group); }, [stations, group]);
  const station = stations.find((s) => s.group === group);
  useEffect(() => { if (station && !day) setDay(station.last.slice(0, 10)); }, [station, day]);

  const ev = useAsync(() => (group ? api.usaEvents(group) : Promise.resolve({ group, events: [] })), [group]);
  const events = ev.data?.events ?? [];
  useEffect(() => {
    setMode(events.length ? "event" : "calendar");
    if (events.length) setEventId(events[0].event_id);
  }, [group, ev.data]);

  const range = useMemo<[string, string] | null>(() => {
    if (mode === "event" && events.length) {
      const e = events.find((x) => x.event_id === eventId) ?? events[0];
      const s = Date.parse(e.start) - ctx * 60000;
      const en = Math.min(Date.parse(e.end) + ctx * 60000, s + 7 * 86400000);
      return [iso(s), iso(en)];
    }
    if (day) { const s = Date.parse(day + "T00:00:00Z"); return [iso(s), iso(s + days * 86400000 - 1)]; }
    return null;
  }, [mode, events, eventId, ctx, day, days]);

  const win = useAsync(
    () => (group && range ? api.usaWindow(group, range[0], range[1]) : Promise.resolve(null as any)),
    [group, range?.[0], range?.[1]],
  );
  const health = useAsync(() => (group ? api.usaHealth(group) : Promise.resolve(null as any)), [group]);
  const stateColor: Record<string, string> = {
    healthy: p.accent2, watch: "#C79A3A", degraded: p.accent, critical: p.alert,
  };
  // __USA_BODY__
  if (cat.loading) return <div className="loading">Loading real-observation artifacts…</div>;
  if (cat.error) return <div className="err">Could not load artifacts: {cat.error}</div>;

  const s = win.data?.summary;
  return (
    <div className="layout">
      <aside className="panel">
        <h3>SkyGuard AI</h3>
        <p className="hint">SIH26073 · real-observation minute detector</p>
        <Select label="Station | source" value={group} onChange={setGroup}
          options={stations.map((st) => ({ value: st.group, label: `${st.station_id} · ${st.source}` }))} />
        {station && (
          <p className="hint">{station.records.toLocaleString()} archived rows<br />
            {station.first.slice(0, 10)} → {station.last.slice(0, 10)}</p>
        )}
        <Segmented label="Replay selection" value={mode} onChange={(v) => setMode(v as any)}
          options={[{ value: "event", label: "Candidate event" }, { value: "calendar", label: "Calendar window" }]} />
        {mode === "event" && events.length > 0 && (
          <>
            <Select label="Candidate to inspect" value={eventId} onChange={setEventId}
              options={events.map((e) => ({ value: e.event_id, label: `${e.start.slice(0, 16).replace("T", " ")} · ${e.channel}` }))} />
            <Select label="Context (minutes)" value={String(ctx)} onChange={(v) => setCtx(Number(v))}
              options={[15, 30, 60, 180, 360].map((n) => ({ value: String(n), label: `± ${n} min` }))} />
          </>
        )}
        {mode === "event" && events.length === 0 && <p className="hint">No candidate events for this stream.</p>}
        {mode === "calendar" && (
          <>
            <div className="field"><label>Window start (UTC)</label>
              <input type="date" value={day} min={station?.first.slice(0, 10)} max={station?.last.slice(0, 10)}
                onChange={(e) => setDay(e.target.value)} /></div>
            <Select label="Window length" value={String(days)} onChange={(v) => setDays(Number(v))}
              options={[1, 3, 7].map((n) => ({ value: String(n), label: `${n} day${n > 1 ? "s" : ""}` }))} />
          </>
        )}
        <label className="hint" style={{ display: "flex", gap: 8, alignItems: "center", cursor: "pointer" }}>
          <input type="checkbox" checked={show60} onChange={(e) => setShow60(e.target.checked)} />
          Show 60-minute model predictions
        </label>
      </aside>

      <main className="main">
        <h1 className="page-title">Observation replay</h1>
        <p className="page-sub">NOAA SURFRAD · historical native one-minute averages · all times UTC
          {cat.data?.policy ? ` · ${cat.data.policy}` : ""}</p>
        <Banner kind="info">Candidate alerts identify unusual observed behaviour. Hardware-fault status stays
          unknown without independent review; provider QC is separate quality evidence.</Banner>
        {health.data?.health && (
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", margin: "8px 0 2px" }}>
            <span className="caption" style={{ margin: 0 }}>Sensor health · last {health.data.window_days}d:</span>
            {Object.entries(health.data.health.channels).map(([ch, d]: any) => (
              <span key={ch} style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 11px",
                borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface)", fontSize: "0.82rem" }}>
                <span style={{ width: 9, height: 9, borderRadius: "50%", background: stateColor[d.state] }} />
                {ch.replace("_", " ").replace("_pct", "").replace("_c", "").replace("_hpa", "")}
                <b style={{ color: stateColor[d.state], textTransform: "capitalize" }}>{d.state}</b>
                <span style={{ color: "var(--muted)" }}>{(d.candidate_rate * 100).toFixed(1)}%{d.trend === "worsening" ? " ↑" : d.trend === "improving" ? " ↓" : ""}</span>
              </span>
            ))}
          </div>
        )}
        {/* __USA_CHARTS__ */}
        {win.loading && <div className="loading">Reading the selected interval…</div>}
        {win.error && <div className="err">{win.error}</div>}
        {win.data && s && (
          <>
            <div className="metrics">
              <MetricTile label="Observed rows" value={s.records.toLocaleString()} />
              <MetricTile label="Candidate rows" value={s.candidates.toLocaleString()} accent />
              <MetricTile label="Missing readings" value={s.missing_values.toLocaleString()} />
              <MetricTile label="Unreported minutes" value={s.unreported_slots.toLocaleString()} />
            </div>
            <p className="caption">{win.data.start.slice(0, 16).replace("T", " ")} — {win.data.end.slice(0, 16).replace("T", " ")} UTC ·
              {" "}{s.scored.toLocaleString()} rows have an anomaly score. Missing scores mean no score is available — not "normal".</p>
            {win.data.channels.length === 0 && <div className="chart-empty">No archived records in this interval.</div>}
            {win.data.channels.map((ch) => (
              <div className="chart-card" key={ch.key}>
                <h4>{ch.label}</h4>
                <EChart option={timeSeriesOption(ch, p, show60)} height={250} />
              </div>
            ))}
            {win.data.candidates.length > 0 && (
              <>
                <h4 style={{ fontFamily: "Inter", marginTop: 6 }}>Candidate reasons in this window</h4>
                <div className="table-wrap">
                  <table className="data">
                    <thead><tr><th>Time (UTC)</th><th>Reason codes</th><th>Score</th><th>Suggested type</th><th>Conf.</th></tr></thead>
                    <tbody>
                      {win.data.candidates.slice(0, 200).map((r, i) => (
                        <tr key={i}>
                          <td>{(r.timestamp || "").replace("T", " ").replace("Z", "")}</td>
                          <td>{r.reason_codes ?? ""}</td>
                          <td>{r.anomaly_score == null ? "—" : Number(r.anomaly_score).toFixed(3)}</td>
                          <td>{r.fault_type ?? "—"}</td>
                          <td>{r.type_confidence == null ? "—" : `${(Number(r.type_confidence) * 100).toFixed(0)}%`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="caption">Anomaly scores are detector evidence relative to calibrated thresholds — not probabilities of hardware failure. "Suggested type" is a classifier trained on synthetic injected faults; it's a triage hint, not a confirmed hardware fault.</p>
              </>
            )}
            <Accordion title="What do these numbers mean?">
              <strong>Observed rows</strong> — existing one-minute records in this window (missing minutes are never invented).<br />
              <strong>Candidate rows</strong> — minutes where ≥1 causal signal crossed its frozen per-station threshold; a proposal for review, not a confirmed fault.<br />
              <strong>Missing readings</strong> — absent temperature/humidity/pressure values inside existing rows.<br />
              <strong>Unreported minutes</strong> — clock-minute slots with no record at all (true gaps).
            </Accordion>
            <Accordion title="Reading the plots">
              The solid line is the native measurement (with a soft area). The terracotta dashed line is the 1-minute-ahead
              prediction; the teal dashed line (toggle) is the 60-minute prediction. Open crimson rings mark candidate readings.
              Gaps break the lines. Drag on the mini-axis to zoom; hover for a crosshair with exact values. Detection is
              unchanged by what you view.
            </Accordion>
          </>
        )}
      </main>
    </div>
  );
}
