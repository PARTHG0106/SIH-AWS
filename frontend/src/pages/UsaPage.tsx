import { useEffect, useMemo, useState } from "react";
import type { Palette } from "../palette";
import { api } from "../api";
import { EChart, timeSeriesOption } from "../Chart";
import { Accordion, Banner, MaintenanceCard, MetricTile, Segmented, Select, useAsync } from "../ui";
import { CorrectionsPanel, StreamPanel } from "../components/Panels";

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
  useEffect(() => { if (station && (!day || day < station.first.slice(0, 10) || day > station.last.slice(0, 10))) setDay(station.last.slice(0, 10)); }, [station, day]);

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
    () => (group && range ? api.usaWindow(group, range[0], range[1]) : Promise.resolve(null)),
    [group, range?.[0], range?.[1]],
  );
  const health = useAsync(() => (group ? api.usaHealth(group) : Promise.resolve(null)), [group]);
  const spatial = useAsync(() => (group ? api.usaSpatial(group) : Promise.resolve(null)), [group]);
  const corrections = useAsync(() => (group ? api.usaCorrections(group) : Promise.resolve(null)), [group]);
  const [stream, setStream] = useState<{ playing: boolean; rows: any[]; summary: any; group: string | null }>(
    { playing: false, rows: [], summary: null, group: null });
  const stateColor: Record<string, string> = {
    healthy: p.accent2, watch: "#C79A3A", degraded: p.accent, critical: p.alert, unknown: p.muted, unavailable: p.muted,
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
        <Segmented label="Replay selection" value={mode} onChange={(v) => setMode(v === "event" ? "event" : "calendar")}
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
            <div className="field"><label htmlFor="usa-day">Window start (UTC)</label>
              <input id="usa-day" type="date" value={day} min={station?.first.slice(0, 10)} max={station?.last.slice(0, 10)}
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
            <span className="caption" style={{ margin: 0 }}>Candidate activity · latest {health.data.window_days ?? 30} archive days:</span>
            {Object.entries(health.data.health.channels).map(([ch, d]) => (
              <span key={ch} style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 11px",
                borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface)", fontSize: "0.82rem" }}>
                <span style={{ width: 9, height: 9, borderRadius: "50%", background: stateColor[d.state] }} />
                {ch.replace("_", " ").replace("_pct", "").replace("_c", "").replace("_hpa", "")}
                <b style={{ color: stateColor[d.state] ?? p.muted }}>{d.state === "healthy" ? "Low activity" : d.state}</b>
                <span style={{ color: "var(--muted)" }}>{d.candidate_rate == null ? "No scores" : `${(d.candidate_rate * 100).toFixed(1)}%`}{d.trend === "worsening" ? " ↑" : d.trend === "improving" ? " ↓" : ""}</span>
              </span>
            ))}
          </div>
        )}
        {health.data?.health && <p className="caption">Rates describe candidate rows in each channel’s scored records over the latest archive window, independently of the interval below. Activity bands do not establish hardware health.</p>}
        {!health.loading && !health.data?.health && <p className="caption">Candidate activity unavailable. Hardware status: unknown.</p>}
        <MaintenanceCard advisory={health.data?.health?.maintenance_advisory} />
        {spatial.data && spatial.data.per_channel && (
          <div className="chart-card" style={{ marginTop: 12 }}>
            <h4>Spatial consistency (vs network median) · {spatial.data.snapshot_utc}</h4>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
              {Object.entries(spatial.data.per_channel).map(([ch, d]: any) => (
                <div key={ch} style={{ padding: 10, border: "1px solid var(--border)", borderRadius: 12 }}>
                  <div className="caption" style={{ margin: 0 }}>{ch.replace("_", " ")}</div>
                  <div style={{ fontFamily: "Fraunces", fontSize: "1.3rem", marginTop: 2,
                    color: d.consensus === "diverging" ? p.alert : d.consensus === "unknown" ? p.muted : p.accent2 }}>
                    {d.consensus}
                  </div>
                  <div className="caption" style={{ margin: 0 }}>{d.trigger ?? "—"}</div>
                </div>
              ))}
            </div>
            <p className="caption" style={{ marginTop: 8 }}>{spatial.data.policy}</p>
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
            {corrections.data?.candidates && corrections.data.candidates.length > 0 && (
              <CorrectionsPanel corr={corrections.data} p={p} />
            )}
            {win.data.candidates.length > 0 && (
              <>
                <h4 style={{ fontFamily: "Inter", marginTop: 6 }}>Candidate reasons in this window</h4>
                <div className="table-wrap">
                  <table className="data">
                    <thead><tr><th>Time (UTC)</th><th>Detector evidence</th><th>Score</th><th>Hardware status</th></tr></thead>
                    <tbody>
                      {win.data.candidates.slice(0, 200).map((r, i) => (
                        <tr key={i}>
                          <td>{(r.timestamp || "").replace("T", " ").replace("Z", "")}</td>
                          <td>{r.reason_codes ?? ""}</td>
                          <td>{r.anomaly_score == null ? "—" : Number(r.anomaly_score).toFixed(3)}</td>
                          <td>Unknown · review required</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="caption">Anomaly scores are detector evidence relative to calibrated thresholds. They are not probabilities of hardware failure; a reason code describes the detected pattern.</p>
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
        <CorrectionsPanel corr={corrections.data} p={p} />
        <StreamPanel stream={stream} setStream={setStream as any} group={group} p={p} />
      </main>
    </div>
  );
}
