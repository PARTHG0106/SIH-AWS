import type { Palette } from "../palette";
import { api } from "../api";
import { EChart, heatmapOption, barOption } from "../Chart";
import { Accordion, Banner, MetricTile, useAsync } from "../ui";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const f3 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3));

export default function BenchmarkPage({ p }: { p: Palette }) {
  const bench = useAsync(() => api.benchmark(), []);
  if (bench.loading) return <div className="loading">Loading benchmark…</div>;
  if (bench.error) return <div className="err">{bench.error}</div>;
  const m = bench.data?.metrics;
  const c = bench.data?.classifier;
  if (!m) return <div className="main" style={{ padding: 28 }}>
    <h1 className="page-title">Detection benchmark</h1>
    <Banner kind="info">No benchmark artifact yet. Run <code>python scripts/run_injection_benchmark.py</code> and <code>run_fault_classifier.py</code>.</Banner>
  </div>;

  const faults = Object.keys(m.per_fault);
  const importances = (c?.shap_mean_abs && Array.isArray(c.shap_mean_abs))
    ? c.shap_mean_abs.map((d: any) => ({ name: d.feature, value: d.mean_abs_shap }))
    : (c?.feature_importances || []).map((d: any) => ({ name: d.feature, value: d.importance }));

  return (
    <div className="main" style={{ padding: "24px 28px 56px", maxWidth: 1180, margin: "0 auto" }}>
      <h1 className="page-title">Detection benchmark</h1>
      <p className="page-sub">SIH is evaluated on anomaly-injected data. Nine fault types injected into held-out real
        SURFRAD, scored by the frozen detector; the fault-type classifier is trained on those injected labels.</p>
      <Banner kind="warn"><strong>SYNTHETIC INJECTION</strong> — labelled synthetic faults on real observations, kept
        separate from the real-observation replay. Not real hardware-fault labels or IMD accuracy.</Banner>

      <div className="metrics">
        <MetricTile label="Detection F1" value={f3(m.point?.f1)} accent />
        <MetricTile label="ROC-AUC" value={f3(m.point?.roc_auc)} accent />
        <MetricTile label="Classifier macro-F1" value={f3(c?.macro_f1_faults)} accent />
        <MetricTile label="Clean false-positive rate" value={pct(m.clean_candidate_rate)} />
      </div>

      <h4 style={{ fontFamily: "Inter", marginTop: 14 }}>Per-fault detection (frozen detector; +physical-consistency layer)</h4>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Fault type</th><th>Row recall</th><th>+ physical</th><th>Event recall</th><th>Median latency (min)</th><th>Rows</th></tr></thead>
          <tbody>
            {faults.map((f) => { const d = m.per_fault[f]; return (
              <tr key={f}>
                <td>{f}</td><td>{pct(d.row_recall)}</td><td>{pct(d.row_recall_combined)}</td>
                <td>{pct(d.event_recall)}</td><td>{d.median_latency_min == null ? "—" : d.median_latency_min.toFixed(0)}</td>
                <td>{d.rows}</td>
              </tr>); })}
          </tbody>
        </table>
      </div>

      {c?.confusion_matrix && (
        <div className="chart-card" style={{ marginTop: 16 }}>
          <h4>Fault-type confusion (row-normalised · macro-F1 {f3(c.macro_f1_faults)}, accuracy {pct(c.accuracy)})</h4>
          <EChart option={heatmapOption(c.confusion_matrix, c.labels, p)} height={420} />
        </div>
      )}
      {importances.length > 0 && (
        <div className="chart-card">
          <h4>Explainability — top signal importances{c?.shap_mean_abs && Array.isArray(c.shap_mean_abs) ? " (SHAP mean |value|)" : " (permutation)"}</h4>
          <EChart option={barOption(importances.slice(0, 12), p)} height={300} />
        </div>
      )}
      <Accordion title="What this proves (and its limits)">
        The detector flags injected faults at ROC-AUC {f3(m.point?.roc_auc)} with a {pct(m.clean_candidate_rate)} clean
        false-positive rate; the learned classifier names the fault type at macro-F1 {f3(c?.macro_f1_faults)}. Gradual
        multiplicative faults (scale_error, clipping) remain hard for a forecast-residual detector — reported honestly, not
        tuned away. These are synthetic-injection results on real SURFRAD; real observations carry no fabricated labels.
      </Accordion>
    </div>
  );
}
