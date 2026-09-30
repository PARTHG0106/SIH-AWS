import { useEffect, useMemo, useState } from "react";
import type { Palette } from "../palette";
import { api } from "../api";
import { EChart, scatterOption } from "../Chart";
import { Accordion, Banner, MetricTile, Select, useAsync } from "../ui";

export default function IndiaPage({ p }: { p: Palette }) {
  const cat = useAsync(() => api.indiaCatalog(), []);
  const [station, setStation] = useState("");
  const [day, setDay] = useState("");
  const [days, setDays] = useState(3);
  const [scenario, setScenario] = useState("spike");
  const [channel, setChannel] = useState("baseline_temperature_c");
  const [startTs, setStartTs] = useState("");
  const [duration, setDuration] = useState(12);
  const [magnitude, setMagnitude] = useState(8);

  const stations = cat.data?.stations ?? [];
  const st = stations.find((s) => s.station_id === station);
  useEffect(() => { if (!station && stations.length) setStation(stations[0].station_id); }, [stations, station]);
  useEffect(() => { if (st && !day) setDay(st.start.slice(0, 10)); }, [st, day]);
  useEffect(() => { setStartTs(""); }, [station, day, days, scenario, channel]);

  const params = useMemo<Record<string, string>>(() => {
    const q: Record<string, string> = { station, start: day, days: String(days), scenario, channel,
      duration: String(duration), magnitude: String(magnitude) };
    if (startTs) q.start_ts = startTs;
    return q;
  }, [station, day, days, scenario, channel, duration, magnitude, startTs]);

  const sc = useAsync(
    () => (station && day ? api.indiaScenario(params) : Promise.resolve(null as any)),
    [JSON.stringify(params)],
  );
  const summary = sc.data?.summary;
  // __INDIA_BODY__
  if (cat.loading) return <div className="loading">Loading Indian demonstration…</div>;
  if (cat.error) return <div className="err">Could not load the Indian demo: {cat.error}</div>;

  const scenarios = cat.data?.scenarios ?? [];
  const channels = cat.data?.channels ?? [];
  const times = sc.data?.window_timestamps ?? [];
  const fmt = (ms: number) => new Date(ms).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

  return (
    <div className="layout">
      <aside className="panel">
        <h3>India · demo</h3>
        <p className="hint">Archived NOAA ISD reports · explicitly synthetic scenarios</p>
        <Select label="Source station" value={station} onChange={setStation}
          options={stations.map((s) => ({ value: s.station_id, label: `${s.name} · ${s.station_id}` }))} />
        {st && <p className="hint">{(st.rows ?? 0).toLocaleString()} source records<br />{st.start.slice(0, 10)} → {st.end.slice(0, 10)}</p>}
        <div className="field"><label>Window start (UTC)</label>
          <input type="date" value={day} min={st?.start.slice(0, 10)} max={st?.end.slice(0, 10)}
            onChange={(e) => setDay(e.target.value)} /></div>
        <Select label="Window length" value={String(days)} onChange={(v) => setDays(Number(v))}
          options={[3, 1, 7].map((n) => ({ value: String(n), label: `${n} day${n > 1 ? "s" : ""}` }))} />
        <Select label="Scenario" value={scenario} onChange={setScenario}
          options={scenarios.map((x) => ({ value: x.key, label: x.label }))} />
        <Select label="Scenario variable" value={channel} onChange={setChannel} disabled={scenario === "baseline"}
          options={channels.map((x) => ({ value: x.key, label: x.label }))} />
        {scenario !== "baseline" && times.length > 0 && (
          <Select label="Scenario start (UTC)" value={startTs || String(sc.data?.scenario_start ?? times[0])} onChange={setStartTs}
            options={times.map((ms) => ({ value: String(ms), label: fmt(ms) }))} />
        )}
        {scenario !== "baseline" && (
          <Select label="Scenario interval" value={String(duration)} onChange={(v) => setDuration(Number(v))}
            options={[1, 3, 6, 12, 24, 48].map((n) => ({ value: String(n), label: `${n} h` }))} />
        )}
        {(scenario === "spike" || scenario === "drift") && (
          <div className="field"><label>{scenario === "spike" ? "Spike offset" : "Offset at interval end"}</label>
            <input type="number" value={magnitude} step={1} min={-100} max={100}
              onChange={(e) => setMagnitude(Number(e.target.value))} /></div>
        )}
      </aside>

      <main className="main">
        <h1 className="page-title">India · Station scenarios</h1>
        <p className="page-sub">Archived NOAA ISD reports from Indian stations · all times UTC</p>
        <Banner kind="warn"><strong>SYNTHETIC DEMONSTRATION</strong> · the scenario series contains changes you choose.
          It is excluded from real-data training and evaluation. Actual hardware-fault status is unknown.</Banner>
        <p className="caption">Temperature and sea-level pressure come from the source; relative humidity is <em>calculated</em>
          from source temperature and dew point (not independently measured), and pressure is sea-level, not station pressure.
          These are historical station reports, not a live IMD AWS feed.</p>
        {/* __INDIA_CHARTS__ */}
        {sc.loading && <div className="loading">Applying scenario…</div>}
        {sc.error && <div className="err">{sc.error}</div>}
        {sc.data && summary && (
          <>
            <div className="metrics">
              <MetricTile label="Source records" value={summary.records.toLocaleString()} />
              <MetricTile label="Modified records" value={summary.modified_records.toLocaleString()} accent />
              <MetricTile label="Modified values" value={summary.modified_values.toLocaleString()} accent />
              <MetricTile label="Missing baseline values" value={summary.missing_baseline_values.toLocaleString()} />
            </div>
            <p className="caption">{st?.name} · {station} · {summary.duplicate_timestamp_rows.toLocaleString()} rows
              share a timestamp with another report. None of these counts is an accuracy, detection, or confirmed-fault number.</p>
            {!sc.data.ready && <Banner kind="info">The selected channel has no source value in this interval, so no scenario is
              applied — the charts show an unchanged copy. Choose another channel or interval.</Banner>}
            {summary.records === 0 && <div className="chart-empty">No source records in this calendar window. Choose another date or a longer window.</div>}
            {sc.data.channels.map((ch) => {
              const has = ch.source.length + ch.synthetic.length > 0;
              return (
                <div className="chart-card" key={ch.key}>
                  <h4>{ch.label}</h4>
                  {has ? <EChart option={scatterOption(ch, p)} height={230} />
                       : <div className="chart-empty">No baseline or synthetic values for this channel in the window.</div>}
                </div>
              );
            })}
            {summary.records > 0 && sc.data.ready && (
              <a className="btn" href={api.indiaCsvUrl(params)}>Download labelled scenario CSV</a>
            )}
            <Accordion title="How to read this demonstration">
              <strong>Circles</strong> are source values (or RH derived from source T and dew point). <strong>Terracotta triangles</strong>
              are the synthetic scenario copy. <strong>Crimson diamonds</strong> mark existing values where the scenario was applied
              (drawn at the baseline value so a dropout stays visible). Every point uses an original report timestamp; reports may be
              irregular or duplicated, so points are never joined by lines and no missing-minute count is inferred. Missing source
              values stay missing.
            </Accordion>
            <Accordion title="Provenance & limits">
              This page never trains, scores, or overwrites source observations. Original fields, quality codes, source-row
              identifiers and file hashes remain in the download. Provider QC is evidence about reported data quality — it does not
              confirm a hardware cause. No claim is made of independently measured RH or station pressure, or of verified IMD AWS
              hardware. Manifest policy: {cat.data?.manifest?.dataset_policy ?? "demo_only_not_real_training"}.
            </Accordion>
          </>
        )}
      </main>
    </div>
  );
}
