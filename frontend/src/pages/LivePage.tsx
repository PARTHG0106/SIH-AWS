import { useEffect, useMemo, useRef, useState } from "react";
import { api, type LiveObservation, type LiveResult, type LiveScenario, type LiveSnapshot, type Pair, type UsaChannel } from "../api";
import { EChart, timeSeriesOption } from "../Chart";
import type { Palette } from "../palette";
import { Accordion, Banner, MaintenanceCard, MetricTile, Select, useAsync } from "../ui";

const CHANNELS = [
  { key: "temperature_c", label: "Temperature", unit: "°C" },
  { key: "relative_humidity_pct", label: "Relative humidity", unit: "%" },
  { key: "pressure_hpa", label: "Station pressure", unit: "hPa" },
];
const SCENARIOS = [
  { value: "spike", label: "Spike" }, { value: "bias", label: "Persistent offset" }, { value: "drift", label: "Gradual drift" },
  { value: "stuck", label: "Stuck value" }, { value: "dropout", label: "Missing input / dropout" },
  { value: "noise_burst", label: "Noise burst" }, { value: "clipping", label: "Clipping" },
  { value: "scale_error", label: "Scale change" }, { value: "sensor_swap", label: "Temperature / RH swap" },
];
type LiveChart = UsaChannel & { baseline: Pair[]; applied: Pair[] };
const utc = (timestamp: string) => timestamp.replace("T", " ").replace(/(?:\.\d+)?(?:Z|\+00:00)$/, "");
const number = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const evidence = (row: LiveResult) => [...(row.explanations ?? []), ...(row.physics_alerts ?? [])];

export default function LivePage({ p }: { p: Palette }) {
  const catalog = useAsync(() => api.usaCatalog(), []);
  const [group, setGroup] = useState("");
  const [streamMode, setStreamMode] = useState<"real" | "synthetic">("real");
  const [scenario, setScenario] = useState("spike");
  const [scenarioChannel, setScenarioChannel] = useState("temperature_c");
  const [scenarioInfo, setScenarioInfo] = useState<LiveScenario | null>(null);
  const [sessionGroup, setSessionGroup] = useState("");
  const [start, setStart] = useState("");
  const [hours, setHours] = useState(3);
  const [speed, setSpeed] = useState(10);
  const [running, setRunning] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(null);
  const [results, setResults] = useState<LiveResult[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [batchMs, setBatchMs] = useState<number | null>(null);
  const [inFlight, setInFlight] = useState(false);
  const replay = useRef<LiveObservation[]>([]);
  const cursor = useRef(0);
  const generation = useRef(0);
  const sending = useRef(false);
  const activeSession = useRef<string | null>(null);
  const stations = catalog.data?.stations ?? [];
  const station = stations.find((s) => s.group === group);
  const synthetic = streamMode === "synthetic";
  const events = useAsync(() => group ? api.usaEvents(group) : Promise.resolve(null), [group]);

  useEffect(() => { if (!group && stations.length) setGroup(stations[0].group); }, [group, stations]);
  useEffect(() => { if (station) setStart(`${station.last.slice(0, 10)}T00:00`); }, [station]);
  useEffect(() => () => {
    generation.current += 1;
    const id = activeSession.current;
    if (id) void api.liveDelete(id).catch(() => undefined);
  }, []);

  // Only arriving input rows are sent to the Python scorer. No score is fetched or
  // computed in advance; each successful batch advances this session's cursor.
  useEffect(() => {
    if (!running || !sessionId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const epoch = generation.current;
    const tick = async () => {
      const started = performance.now();
      if (!sending.current && cursor.current < replay.current.length) {
        sending.current = true;
        setInFlight(true);
        const batch = replay.current.slice(cursor.current, cursor.current + speed);
        try {
          const response = await api.liveObserve(sessionId, batch);
          if (generation.current !== epoch) return;
          cursor.current += batch.length;
          setResults((prior) => [...prior, ...response.results]);
          setSnapshot(response.snapshot);
          setBatchMs(performance.now() - started);
          if (cursor.current >= replay.current.length) setRunning(false);
        } catch (cause: unknown) {
          if (generation.current === epoch) { setError(message(cause)); setRunning(false); }
        } finally {
          sending.current = false;
          if (generation.current === epoch) setInFlight(false);
        }
      }
      if (!stopped && generation.current === epoch && cursor.current < replay.current.length) {
        timer = setTimeout(() => void tick(), Math.max(100, 1000 - (performance.now() - started)));
      }
    };
    void tick();
    return () => { stopped = true; clearTimeout(timer); };
  }, [running, sessionId, speed]);

  async function begin() {
    if (sessionId) { setError(null); setRunning(true); return; }
    const parsed = Date.parse(`${start}${start.length === 16 ? ":00" : ""}Z`);
    if (!group || !Number.isFinite(parsed)) { setError("Choose a station and a valid UTC start time."); return; }
    const epoch = generation.current;
    setPreparing(true); setError(null);
    try {
      const end = new Date(parsed + (synthetic ? 6 : hours) * 3600000).toISOString();
      const source = synthetic
        ? await api.liveScenario(group, new Date(parsed).toISOString(), end, scenario, scenario === "sensor_swap" ? "temperature_c" : scenarioChannel)
        : await api.liveReplay(group, new Date(parsed).toISOString(), end);
      if (generation.current !== epoch) return;
      if (!source.observations.length) throw new Error("No original observations in this interval. Choose another window.");
      const session = await api.liveCreate(source.group);
      if (generation.current !== epoch) { void api.liveDelete(session.session_id).catch(() => undefined); return; }
      replay.current = source.observations;
      cursor.current = 0;
      activeSession.current = session.session_id;
      setTotal(source.observations.length);
      setSessionGroup(source.group);
      setScenarioInfo(synthetic ? source as LiveScenario : null);
      setSnapshot(session.snapshot);
      setSessionId(session.session_id);
      setRunning(true);
    } catch (cause: unknown) {
      if (generation.current === epoch) setError(message(cause));
    } finally { if (generation.current === epoch) setPreparing(false); }
  }

  function reset() {
    generation.current += 1;
    setRunning(false); setPreparing(false); setInFlight(false); setError(null);
    const id = activeSession.current;
    activeSession.current = null;
    setSessionId(null); setSessionGroup(""); setScenarioInfo(null); setSnapshot(null); setResults([]); setTotal(0); setBatchMs(null);
    replay.current = []; cursor.current = 0;
    if (id) void api.liveDelete(id).catch(() => undefined);
  }

  const latest = results[results.length - 1];
  const state = snapshot?.groups?.[sessionGroup || group];
  const modified = results.filter((row) => row.scenario_modified === true).length;
  const provenance = synthetic ? latest?.baseline : latest;
  const candidates = results.filter((row) => row.is_candidate);
  const patternCandidates = results.filter((row) => row.pattern_evidence?.is_candidate);
  const alerts = results.filter((row) => row.is_candidate || row.physics_alerts?.length || row.pattern_evidence?.is_candidate);
  const pattern = latest?.pattern_evidence;
  const patternStatus = pattern?.status === "scored" ? "Scored" : pattern?.status === "warming_up" ? "Warming up" : "Unavailable";
  const scored = results.filter((row) => row.anomaly_score != null).length;
  const missing = results.reduce((sum, row) => sum + (row.availability?.missing_channels.length ?? 0), 0);
  const gaps = results.reduce((sum, row) => sum + (row.availability?.unreported_slots_before ?? 0), 0);
  const done = total > 0 && results.length >= total;
  const status = preparing ? synthetic ? "Preparing scenario" : "Loading originals" : running ? "Running" : inFlight ? "Pausing after current batch" : done ? "Stream complete" : sessionId ? "Paused" : "Ready";
  const charts = useMemo<LiveChart[]>(() => CHANNELS.map((channel) => {
    const output: LiveChart = { ...channel, observed: [], model_1m: [], model_60m: [], candidates: [], baseline: [], applied: [] };
    let prior: number | null = null;
    for (const row of results) {
      const time = Date.parse(row.timestamp);
      if (prior != null && time - prior > 60000) {
        output.observed.push([prior + 1, null]); output.model_1m.push([prior + 1, null]);
        output.baseline.push([prior + 1, null]);
      }
      output.observed.push([time, number(row[channel.key])]);
      output.model_1m.push([time, number(row[`${channel.key}__prediction`])]);
      if (synthetic) {
        const baseline = number(row[`baseline_${channel.key}`]);
        output.baseline.push([time, baseline]);
        if (row[`${channel.key}__scenario_modified`] === true) output.applied.push([time, baseline]);
      }
      if (row[`${channel.key}__alert`] === true) output.candidates.push([time, number(row[channel.key]), String(row[`${channel.key}__reason_codes`] ?? "")]);
      prior = time;
    }
    return output;
  }), [results, synthetic]);

  if (catalog.loading) return <div className="loading" role="status">Loading replay sources…</div>;
  if (catalog.error) return <div className="err" role="alert">Could not load replay sources: {catalog.error}</div>;

  return <div className="layout">
    <aside className="panel">
      <div className="eyebrow">Operations console</div>
      <h3>{synthetic ? "Stream a software scenario" : "Run an incremental replay"}</h3>
      <p className="hint">{synthetic ? "A labelled scenario copy arrives in order and is scored by the backend. Original baseline records stay separate." : "Original SURFRAD observations arrive in order and are scored by the backend as they arrive."}</p>
      <Select label="Input mode" value={streamMode} disabled={!!sessionId || preparing}
        onChange={(value) => { setStreamMode(value === "synthetic" ? "synthetic" : "real"); setError(null); setSpeed(value === "synthetic" ? 60 : 10); }}
        options={[{ value: "real", label: "Real archive replay" }, { value: "synthetic", label: "Synthetic scenario stream" }]} />
      <Select label={synthetic ? "Original baseline station" : "Station | source"} value={group} onChange={setGroup} disabled={!!sessionId || preparing}
        options={stations.map((s) => ({ value: s.group, label: `${s.station_id} · ${s.source}` }))} />
      <div className="field"><label htmlFor="live-start">{synthetic ? "Baseline start (UTC)" : "Replay start (UTC)"}</label>
        <input id="live-start" type="datetime-local" value={start} disabled={!!sessionId || preparing}
          min={station?.first.slice(0, 16)} max={station?.last.slice(0, 16)} onChange={(e) => setStart(e.target.value)} /></div>
      {synthetic ? <>
        <Select label="Applied software scenario" value={scenario} onChange={setScenario} disabled={!!sessionId || preparing} options={SCENARIOS} />
        <Select label="Scenario channel" value={scenario === "sensor_swap" ? "temperature_c" : scenarioChannel} onChange={setScenarioChannel}
          disabled={!!sessionId || preparing || scenario === "sensor_swap"} options={CHANNELS.map((ch) => ({ value: ch.key, label: ch.label }))} />
        <p className="hint">Six-hour stream · first 180 minutes unchanged for history, followed by the applied scenario and recovery.
          {scenario === "sensor_swap" && " This software scenario swaps temperature and RH values."}</p>
      </> : <Select label="Window length" value={String(hours)} onChange={(v) => setHours(Number(v))} disabled={!!sessionId || preparing}
        options={[1, 3, 6].map((n) => ({ value: String(n), label: `${n} hour${n === 1 ? "" : "s"}` }))} />}
      {!synthetic && !!events.data?.events.length && !sessionId && <button type="button" className="btn ghost example-btn" disabled={preparing}
        onClick={() => { setStart(new Date(Date.parse(events.data!.events[0].start) - 150 * 60000).toISOString().slice(0, 16)); setHours(6); }}>
        Use an archived candidate window
      </button>}
      <Select label="Target stream speed" value={String(speed)} onChange={(v) => setSpeed(Number(v))}
        options={[1, 10, 60].map((n) => ({ value: String(n), label: `${n} input record${n === 1 ? "" : "s"} / second` }))} />
      <div className="replay-buttons">
        {running ? <button className="btn" type="button" onClick={() => setRunning(false)}>Pause stream</button>
          : <button className="btn" type="button" onClick={() => void begin()} disabled={preparing || inFlight || done || !group || !start || (!!error && !!sessionId)}>
            {preparing ? "Preparing…" : sessionId ? "Resume stream" : synthetic ? "Start synthetic stream" : "Start replay"}</button>}
        <button className="btn ghost" type="button" onClick={reset} disabled={!sessionId && !preparing && !error}>Reset</button>
      </div>
      <p className="hint">Speed is capped by scoring time. Pause finishes the active batch. Reset starts a fresh detector session and unlocks the source controls.</p>
      <div className={"source-note" + (synthetic ? " synthetic-source" : "")}><strong>Source mode</strong><br />{synthetic ? "SYNTHETIC SCENARIO" : "Historical real observations"}<br /><span>Live sensor connection: not connected</span></div>
    </aside>

    <main className="main">
      <div className="page-heading"><div><div className="eyebrow">Score each arriving input</div>
        <h1 className="page-title">Live detection</h1></div>
        <span className={`status-pill ${running ? "running" : ""}`} role="status"><span aria-hidden="true" className="status-dot" />{status}</span></div>
      <p className="page-sub">{synthetic ? "Synthetic scenario stream on an archived NOAA SURFRAD baseline" : "Historical NOAA SURFRAD replay"} · backend scoring · all times UTC</p>
      <div className={synthetic ? "synthetic-mode-banner" : undefined}>
        {synthetic ? <Banner kind="warn"><strong>SYNTHETIC SCENARIO STREAM</strong> · the detector receives a deliberately modified copy of historical inputs.
          Applied-change markers are scenario instructions, separate from model detections. This is not a genuine live sensor feed or a measurement of hardware-fault accuracy.</Banner>
          : <Banner>Candidate alerts are proposals for inspection. Hardware-fault status remains <strong>unknown</strong>.
            The replay uses archived original observations; it is not a connected live weather feed.</Banner>}
      </div>
      {error && <div className="err" role="alert">{error} {sessionId && "Reset the session before trying again if the last batch was interrupted."}</div>}
      <div className="stream-strip">
        <div><span className="caption">{synthetic ? "Synthetic stream clock · historical UTC timestamps" : "Observation clock · UTC"}</span><strong>{latest ? utc(latest.timestamp) : "Waiting for the first input"}</strong></div>
        <div><span className="caption">Session progress</span><strong>{results.length.toLocaleString()} / {total ? total.toLocaleString() : "—"} records</strong></div>
      </div>
      <progress className="replay-progress" aria-label="Replay progress" value={results.length} max={total || 1} />
      <div className={"metrics" + (synthetic ? " synthetic-metrics" : "")}>
        <MetricTile label="Rows with a score" value={scored.toLocaleString()} />
        {synthetic && <MetricTile label="Applied modified rows" value={modified.toLocaleString()} accent />}
        <MetricTile label="Frozen-detector candidates" value={candidates.length.toLocaleString()} accent />
        {synthetic && <MetricTile label="Learned-model candidates" value={results.some((row) => row.pattern_evidence?.scenario_score != null) ? patternCandidates.length.toLocaleString() : "—"} />}
        <MetricTile label="Missing channel values" value={missing.toLocaleString()} />
        <MetricTile label="Unreported minutes" value={gaps.toLocaleString()} />
      </div>
      {scenarioInfo && <p className="caption"><strong>Applied scenario:</strong> {SCENARIOS.find((item) => item.value === scenarioInfo.scenario)?.label ?? scenarioInfo.scenario}
        {" · "}{scenarioInfo.event.channels.map((name) => CHANNELS.find((ch) => ch.key === name)?.label ?? name).join(" + ")}
        {" · scheduled "}{utc(scenarioInfo.event.scheduled_start)} to {utc(scenarioInfo.event.scheduled_end_exclusive)} UTC.
        {" "}{scenarioInfo.event.modified_rows.toLocaleString()} rows are modified in the complete scenario; the counter above includes only rows received so far. These counts are not accuracy metrics.</p>}
      {scenarioInfo?.event.modified_rows === 0 && <Banner kind="warn">No source values could be changed by this scenario in the selected window. Choose another channel or baseline interval.</Banner>}
      <p className="caption">{latest ? latest.anomaly_score != null ? "Available signals scored. Forecast and sustained signals may still need more history." : "No score available yet: required history or observations are missing." : "The detector builds causal history from arriving rows. Missing or warming-up scores remain unavailable."}
        {batchMs != null && ` Last batch request: ${batchMs < 1000 ? `${batchMs.toFixed(0)} ms` : `${(batchMs / 1000).toFixed(2)} s`} including transport.`}</p>

      <MaintenanceCard advisory={state?.health?.maintenance_advisory} />
      {synthetic && state?.health?.maintenance_advisory && <p className="caption">This service-policy response is a demonstration on synthetic inputs. It does not describe the physical station’s maintenance condition.</p>}
      <div className="channel-status-grid">
        {CHANNELS.map((channel) => {
          const h = state?.health?.channels[channel.key];
          const value = latest ? number(latest[channel.key]) : null;
          return <div className="channel-status" key={channel.key}>
            <h4>{channel.label}</h4><div className="reading">{value == null ? "—" : value.toFixed(channel.key === "pressure_hpa" ? 1 : 2)} <small>{channel.unit}</small></div>
            <div className={`availability ${!latest ? "waiting" : value == null ? "unavailable" : ""}`}>Input availability: {!latest ? "waiting" : value == null ? "missing" : synthetic ? "synthetic copy value" : "observed"}</div>
            <p>Candidate activity: {h?.candidate_rate == null ? "unavailable" : `${(h.candidate_rate * 100).toFixed(1)}% of scored rows`}</p>
            <p>Forecasts: 1-min {latest && number(latest[`${channel.key}__prediction`]) != null ? "available" : "unavailable"} · 60-min {latest && number(latest[`${channel.key}__prediction_60m`]) != null ? "available" : "unavailable"}</p>
            <span className="caption">Hardware status: unknown</span>
          </div>;
        })}
      </div>

      {pattern && <section className="pattern-card" aria-label="Learned scenario pattern evidence">
        <div className="section-heading"><h2>Experimental scenario pattern triage</h2><span className="status-pill">{patternStatus}</span></div>
        <p className="caption">Learned evidence is calibrated on synthetic scenarios. Reliable transfer to new stations is not established; these scores are not probabilities of real hardware failure.</p>
        {pattern.status === "warming_up" ? <p className="caption">The backend requires {pattern.required_contiguous_rows ?? 181} contiguous minute records, including the current input, before producing scenario evidence. Gaps restart this history requirement.</p>
          : pattern.status !== "scored" ? <p className="caption">The optional scenario model could not produce evidence{pattern.reason ? ` (${pattern.reason})` : ""}.</p>
          : <><div className="pattern-values">
            <div><span>Latest suggested pattern</span><strong>{(pattern.suggested_pattern ?? "Unavailable").replace(/_/g, " ")}</strong></div>
            <div><span>Learned candidate rows</span><strong>{patternCandidates.length.toLocaleString()}</strong></div>
            <div><span>Scenario score / threshold</span><strong>{pattern.scenario_score?.toFixed(3) ?? "—"} / {pattern.threshold?.toFixed(3) ?? "—"}</strong></div>
            <div><span>Pattern confidence</span><strong>{pattern.pattern_confidence?.toFixed(3) ?? "—"}</strong></div>
          </div>
          <p className="caption">{pattern.confidence_semantics ?? "Calibrated on synthetic scenarios; not probability of real hardware failure."}
            {" "}Learned candidates are counted separately from the frozen detector above.</p></>}
      </section>}

      {!results.length && <div className="empty-state"><div className="eyebrow">Ready when you are</div><h2>{synthetic ? "Send a scenario through the detector" : "Watch observations become evidence"}</h2>
        <p>{synthetic ? "Press Start synthetic stream to prepare a labelled copy and score it incrementally. The original baseline, applied changes and detected candidates will be shown separately." : "Choose a station and press Start replay. Measurements, causal predictions and candidate reasons will appear as the backend processes each batch."}</p></div>}
      {synthetic && results.length > 0 && <p className="caption"><strong>Plot key:</strong> grey line = original baseline; dark line = synthetic input; teal diamonds = applied changes, drawn at the baseline value so a dropout remains visible; open crimson rings = frozen-detector candidates. Markers are not interchangeable.</p>}
      {results.length > 0 && charts.map((ch) => <div className="chart-card" key={ch.key}><h4>{ch.label}{synthetic ? " · synthetic input" : ""}</h4><EChart option={timeSeriesOption(ch, p, false, synthetic ? { inputLabel: "Synthetic input", baseline: ch.baseline, applied: ch.applied } : undefined)} height={synthetic ? 250 : 220} /></div>)}

      <div className="section-heading"><h2>Recent inspection alerts</h2><span className="caption">Latest 30 · review proposals</span></div>
      {alerts.length ? <div className="table-wrap alert-table" tabIndex={0} aria-label="Recent inspection alerts, scrollable">
        <table className="data"><caption className="sr-only">Backend candidate or range evidence and suggested inspection actions; hardware causes are unknown.</caption>
          <thead><tr><th scope="col">Time (UTC)</th><th scope="col">Priority</th><th scope="col">Evidence</th><th scope="col">Suggested inspection</th></tr></thead>
          <tbody>{alerts.slice(-30).reverse().map((row, index) => <tr key={`${row.timestamp}-${index}`}>
            <td className="nowrap">{utc(row.timestamp)}</td><td><span className={`severity ${row.severity}`}>{row.severity === "high" ? "High" : "Review"}</span></td>
            <td>{evidence(row).length ? evidence(row).map((e, i) => <p key={i}><strong>{e.channel.replace(/_/g, " ")}</strong> · {e.reason || e.signal}</p>) : row.reason_codes || (row.is_candidate ? "Candidate threshold crossed" : "")}
              {row.pattern_evidence?.is_candidate && <p><strong>Learned software pattern:</strong> {(row.pattern_evidence.suggested_pattern ?? "unspecified").replace(/_/g, " ")}
                {" · scenario score "}{row.pattern_evidence.scenario_score?.toFixed(3) ?? "unavailable"} · experimental synthetic-pattern evidence</p>}
              <small>Score: {row.anomaly_score == null ? "unavailable" : row.anomaly_score.toFixed(3)} · hardware status unknown</small></td>
            <td>{evidence(row).length ? [...new Set(evidence(row).map((e) => e.action).filter(Boolean))].map((action) => <p key={action}>{action}</p>) : "Inspect the original record, provider quality flags and adjacent readings."}</td>
          </tr>)}</tbody></table></div>
        : <div className="chart-empty">{scored ? "No candidate alerts among the rows scored so far. Hardware condition is still unknown." : "No scored candidate evidence yet."}</div>}
      <Accordion title="How this session works">
        {synthetic ? "The browser sends an explicitly synthetic scenario copy to a separate synthetic_demo backend session. Original baseline values, source IDs and hashes remain separate. The known applied mask never serves as detector output. " : "The browser sends ordered original observations to a fresh backend detector session. "}The detector uses only prior and current
        inputs with its frozen parameters. Predictions remain separate from inputs, and gaps break the plots.
        Abrupt-change and flatline signals can become available before the forecasts. Forecast contexts require 61 or 120 minutes,
        and the sustained signal can require 149 minutes; gaps can delay readiness. Activity rates apply to the scored rows seen in this session. Missing values and absent expected minutes are availability evidence,
        not a confirmed hardware cause. Suggested inspections are evidence-based triage steps; their confidence is not a hardware-failure probability.
      </Accordion>
      {latest && <Accordion title={synthetic ? "Original baseline provenance and provider QC" : "Latest observation provenance and provider QC"}>
        <dl className="provenance-grid">
          <dt>{synthetic ? "Synthetic input ID" : "Observation"}</dt><dd>{latest.observation_id ?? "Not supplied"}</dd>
          {synthetic && <><dt>Baseline observation ID</dt><dd>{latest.baseline_observation_id ?? "Not supplied"}</dd><dt>Baseline station | source</dt><dd>{latest.baseline_group ?? scenarioInfo?.baseline_group ?? "Not supplied"}</dd></>}
          <dt>Original file</dt><dd>{String(provenance?.raw_file_name ?? "Not supplied")} · row {String(provenance?.raw_row_number ?? "unknown")}</dd>
          <dt>Original SHA-256</dt><dd><code>{String(provenance?.raw_file_sha256 ?? "Not supplied")}</code></dd>
          {CHANNELS.map((channel) => <div className="provenance-row" key={channel.key}><dt>{channel.label} · raw QC</dt><dd>{String(provenance?.[`${channel.key}__raw_qc`] ?? "Not supplied")}</dd></div>)}
        </dl>
        Provider quality codes describe the original source report{synthetic ? ", not the modified synthetic values" : ""}. They are separate from detector candidates and do not establish hardware causes.
      </Accordion>}
    </main>
  </div>;
}
