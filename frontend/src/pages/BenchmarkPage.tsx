import { useState } from "react";
import type { Palette } from "../palette";
import { api, type BenchmarkResponse, type ScenarioBenchmark, type ScenarioReport } from "../api";
import { EChart, heatmapOption } from "../Chart";
import { Accordion, Banner, MetricTile, Segmented, useAsync } from "../ui";

const pct = (v: number | null | undefined) => v == null ? "—" : (v * 100).toFixed(1) + "%";
const f3 = (v: number | null | undefined) => v == null ? "—" : v.toFixed(3);
const minutes = (v: number | null | undefined) => v == null ? "—" : v.toFixed(1) + " min";
const label = (name: string) => name === "no_injection" ? "No injected scenario" : name.replace(/_/g, " ");
const partitionLabel = (name: string) => name === "test_temporal" ? "Held-out time" : name === "test_station" ? "Held-out station" : label(name);
const modern = (m: NonNullable<BenchmarkResponse["metrics"]>): m is ScenarioBenchmark => m.schema_version === "sih_benchmark_v2";

function MethodTable({ report, baselines }: { report: ScenarioReport; baselines: Record<string, ScenarioReport> }) {
  const methods: [string, ScenarioReport][] = [["SkyGuard scenario model", report], ...Object.entries(baselines).map(([name, value]): [string, ScenarioReport] => [label(name), value])];
  return <div className="table-wrap benchmark-table" tabIndex={0} aria-label="Comparison of evaluated methods, scrollable">
    <table className="data"><caption className="sr-only">All methods evaluated on the selected final partition</caption>
      <thead><tr><th scope="col">Method</th><th scope="col">Row F1</th><th scope="col">PR-AUC</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Unmodified-row candidate rate</th></tr></thead>
      <tbody>{methods.map(([name, result]) => <tr key={name} className={name === "SkyGuard scenario model" ? "model-row" : ""}><td>{name}</td>
        <td>{f3(result.point.f1)}</td><td>{f3(result.point.pr_auc)}</td><td>{pct(result.point.precision)}</td>
        <td>{pct(result.point.recall)}</td><td>{pct(result.point.unmodified_row_candidate_rate)}</td></tr>)}</tbody>
    </table>
  </div>;
}

function LegacyResults({ data }: { data: BenchmarkResponse }) {
  const metrics = data.metrics;
  if (!metrics || modern(metrics)) return null;
  return <>
    <p className="page-sub">Historical synthetic experiment · no independent final benchmark loaded</p>
    <Banner kind="warn"><strong>HISTORICAL · UNVALIDATED</strong> — legacy classifier results do not establish independent
      generalisation and are excluded from headline performance claims. Targets identify applied software changes, not confirmed hardware faults.</Banner>
    <Accordion title="Inspect historical experiment results">
      <div className="metrics">
        <MetricTile label="Historical detection F1" value={f3(metrics.point?.f1)} />
        <MetricTile label="Historical ROC-AUC" value={f3(metrics.point?.roc_auc)} />
        <MetricTile label="Unvalidated type macro-F1" value={f3(data.classifier?.macro_f1_faults)} />
        <MetricTile label="Unmodified background candidate rate" value={pct(metrics.clean_candidate_rate)} />
      </div>
      <p>These historical results do not support a claim of IMD accuracy or superiority to other implementations. An unmodified background has no injected scenario; its hardware condition is unknown.</p>
    </Accordion>
  </>;
}

export default function BenchmarkPage({ p }: { p: Palette }) {
  const benchmark = useAsync(() => api.benchmark(), []);
  const [selected, setSelected] = useState("test_temporal");
  if (benchmark.loading) return <div className="loading" role="status">Loading benchmark evidence…</div>;
  if (benchmark.error) return <div className="err" role="alert">{benchmark.error}</div>;
  const data = benchmark.data;
  const metrics = data?.metrics;
  const v2 = metrics && modern(metrics) ? metrics : null;
  const partitions = v2?.status === "final_test_complete" ? v2.final_test?.partitions ?? {} : {};
  const key = partitions[selected] ? selected : Object.keys(partitions)[0];
  const partition = partitions[key];
  const report = partition?.model;
  const stationReport = partitions.test_station?.model;
  const classification = report?.classification;
  const reliability = report?.reliability;

  return <main className="main benchmark-page">
    <div className="eyebrow">Evidence & evaluation</div>
    <h1 className="page-title">Detection benchmark</h1>
    {!metrics && <Banner>No final scenario benchmark is available in this dashboard yet.</Banner>}
    {data && metrics && !v2 && <LegacyResults data={data} />}
    {v2 && <>
      <p className="page-sub">Synthetic software scenarios on original observations · independent development and final evaluation</p>
      <Banner kind="warn"><strong>SYNTHETIC BENCHMARK</strong> · targets describe applied software scenarios.
        Unmodified observations have unknown hardware condition. These metrics are separate from real-observation replay and do not measure IMD hardware-fault accuracy.</Banner>
      {stationReport && <Banner kind="warn"><strong>HELD-OUT STATION · TRANSFER NOT ESTABLISHED</strong>
        <p>The model flagged <strong>{pct(stationReport.unmodified_baseline_candidate_rate)}</strong> of rows in wholly unmodified background copies on the held-out station.
          {" "}Held-out-station detection F1: <strong>{f3(stationReport.point.f1)}</strong>; expected calibration error: <strong>{f3(stationReport.reliability?.ece)}</strong>.</p>
        <p>These final results do not establish reliable transfer to new stations. The held-out-time result is a separate evaluation. Background hardware condition remains unknown.</p>
      </Banner>}
      <div className="protocol-strip">
        <div><span>Source windows split before generation</span><strong>{v2.split_audit.split_before_generation ? "Yes" : "Not verified"}</strong></div>
        <div><span>Development windows audited</span><strong>{v2.split_audit.unique_source_windows.toLocaleString()}</strong></div>
        <div><span>Development overlap detected</span><strong>{v2.split_audit.overlapping_source_windows.toLocaleString()}</strong></div>
        <div><span>Final evaluation</span><strong>{report ? "Complete" : "Awaiting final test"}</strong></div>
      </div>
      {!report && <div className="empty-state"><h2>Model frozen; final results pending</h2>
        <p>Selection and calibration evidence is available below. Final performance will appear only after the frozen configuration has been evaluated on the reserved test partitions.</p></div>}
      {report && partition && <>
        <div className="benchmark-selection"><Segmented label="Final evaluation partition" value={key} onChange={setSelected}
          options={Object.keys(partitions).map((name) => ({ value: name, label: partitionLabel(name) }))} />
          <p className="caption">{partition.source_windows.toLocaleString()} source windows · {partition.rows.toLocaleString()} evaluated rows
            {v2.final_test?.evaluated_at_utc ? " · evaluated " + v2.final_test.evaluated_at_utc.slice(0, 10) : ""}</p>
        </div>
        <div className="metrics">
          <MetricTile label="Final detection F1" value={f3(report.point.f1)} accent />
          <MetricTile label="Final PR-AUC" value={f3(report.point.pr_auc)} accent />
          <MetricTile label="Scenario type macro-F1" value={f3(classification?.macro_f1_faults)} />
          <MetricTile label="Unmodified-row candidate rate" value={pct(report.point.unmodified_row_candidate_rate)} />
        </div>
        <p className="caption">Strict row scores: precision {pct(report.point.precision)} · recall {pct(report.point.recall)} · ROC-AUC {f3(report.point.roc_auc)}.
          No point adjustment. Candidate rate on wholly unmodified scenario copies: {pct(report.unmodified_baseline_candidate_rate)}.</p>

        <div className="section-heading"><h2>Measured method comparison</h2><span className="caption">Same final partition · frozen operating points</span></div>
        <MethodTable report={report} baselines={partition.baselines} />
        <p className="caption">These are the methods evaluated by this benchmark. They are not performance claims about uninspected public projects. Unmodified-row candidate rate includes unchanged rows within scenario copies and is not a real hardware false-positive rate.</p>

        <div className="section-heading"><h2>Detection by applied scenario</h2></div>
        <div className="table-wrap benchmark-table" tabIndex={0} aria-label="Scenario detection metrics, scrollable">
          <table className="data"><thead><tr><th scope="col">Applied scenario</th><th scope="col">Row recall</th><th scope="col">Event recall</th><th scope="col">Events detected</th><th scope="col">Median delay</th><th scope="col">P95 delay</th><th scope="col">Modified rows</th></tr></thead>
            <tbody>{Object.entries(report.per_fault).map(([name, result]) => <tr key={name}><td>{label(name)}</td>
              <td>{pct(result.row_recall)}</td><td>{pct(result.event_recall)}</td><td>{result.detected_events} / {result.events}</td>
              <td>{minutes(result.median_latency_min)}</td><td>{minutes(result.p95_latency_min)}</td><td>{result.modified_rows.toLocaleString()}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="caption">An event is detected when at least one modified row is flagged within its applied interval.
          Delays are measured from the first modified row and cover detected events only. {Object.values(report.per_fault).reduce((n, result) => n + result.not_applied_events, 0)} scenario events with no applicable source values were excluded from event recall.</p>

        {classification && <div className="chart-card" style={{ marginTop: 20 }}>
          <h4>Scenario type confusion · row-normalised</h4>
          <p className="caption">Macro-F1 averages the applied scenario classes. “No injected scenario” means unchanged software input, not confirmed normal hardware.</p>
          <EChart option={heatmapOption(classification.confusion_matrix, classification.labels.map(label), p)} height={420} />
        </div>}
        {reliability && <Accordion title="Confidence reliability on synthetic scenarios">
          <p>Brier score {f3(reliability.brier)} · expected calibration error {f3(reliability.ece)}. These compare predicted scenario scores with software modification frequency; they are not probabilities of real hardware failure.</p>
          <div className="table-wrap"><table className="data"><thead><tr><th>Score interval</th><th>Rows</th><th>Mean predicted score</th><th>Observed modification frequency</th></tr></thead>
            <tbody>{reliability.bins.map((bin) => <tr key={bin.lower}><td>{pct(bin.lower)}–{pct(bin.upper)}</td><td>{bin.rows.toLocaleString()}</td>
              <td>{pct(bin.predicted_frequency)}</td><td>{pct(bin.observed_modification_frequency)}</td></tr>)}</tbody></table></div>
        </Accordion>}
      </>}
      <Accordion title="Development evidence · selection and calibration">
        <p>These partitions guided model choice and calibration. They are development evidence and do not replace the independent final evaluation.</p>
        <div className="table-wrap"><table className="data"><thead><tr><th>Partition</th><th>Row F1</th><th>PR-AUC</th><th>Scenario macro-F1</th><th>Unmodified-row candidate rate</th></tr></thead>
          <tbody>{([ ["Selection", v2.selection], ["Calibration", v2.calibration] ] as const).map(([name, result]) => <tr key={name}><td>{name}</td>
            <td>{f3(result.point.f1)}</td><td>{f3(result.point.pr_auc)}</td><td>{f3(result.classification?.macro_f1_faults)}</td><td>{pct(result.point.unmodified_row_candidate_rate)}</td></tr>)}</tbody></table></div>
      </Accordion>
      <Accordion title="Evaluation protocol and scope">
        <p>{v2.split_audit.context_policy}. Original source observations, provider quality codes and provenance are retained separately from applied scenario values.</p>
        <p>The model is selected and calibrated before final evaluation. Overlapping history and related scenario events stay in one source partition. Previously examined periods cannot become untouched final tests.</p>
        <p>Detection and scenario classification answer different questions: whether a software change is present, and which applied pattern the model predicts. Neither establishes the physical cause of an unusual weather-station reading.</p>
        {v2.final_test?.frozen_sha256 && <p className="hash-line">Frozen protocol SHA-256: <code>{v2.final_test.frozen_sha256}</code></p>}
      </Accordion>
    </>}
  </main>;
}
