import { useState } from "react";
import type { Palette } from "../palette";
import { api } from "../api";

const fmt = (ts: string) => (ts || "").replace("T", " ").replace("Z", "");

export function CorrectionsPanel({ corr, p }: { corr: any; p: Palette }) {
  if (!corr) return null;
  const cands = corr.candidates || [];
  if (!cands.length) {
    return (
      <div className="chart-card" style={{ marginTop: 12 }}>
        <h4>Bounded correction suggestions</h4>
        <p className="caption">No current candidates propose changes from peer-channel medians over the same window. Source values are never mutated — suggestions are review-time only.</p>
      </div>
    );
  }
  return (
    <div className="chart-card" style={{ marginTop: 12 }}>
      <h4>Bounded correction suggestions · {corr.candidates.length} candidate(s)</h4>
      <p className="caption">{corr.policy}</p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Time (UTC)</th><th>Channel</th><th>Observed</th><th>Suggested</th><th>Δ</th><th>Stability</th>
            </tr>
          </thead>
          <tbody>
            {cands.slice(0, 50).flatMap((propos: any) => propos.items.map((it: any) => (
              <tr key={`${propos.timestamp}-${it.channel}`}>
                <td>{fmt(propos.timestamp)}</td>
                <td>{it.channel}</td>
                <td>{it.observed?.toFixed?.(2) ?? "—"}</td>
                <td>{it.suggested?.toFixed?.(2) ?? "—"}</td>
                <td>{it.delta?.toFixed?.(2) ?? "—"}</td>
                <td>{it.stability_check_passed ? <span style={{ color: p.accent2 }}>stable</span>
                  : <span style={{ color: p.alert }}>review</span>}</td>
              </tr>
            )))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function StreamPanel({ stream, setStream, group, p }:
  { stream: any; setStream: any; group: string; p: Palette }) {
  const [loading, setLoading] = useState(false);
  const start = async (minutes: number, speed: number) => {
    setLoading(true);
    try {
      const r = await api.usaStream(group, minutes, speed);
      setStream({ playing: true, rows: r.rows, summary: r.summary, group });
    } finally {
      setLoading(false);
    }
  };
  const stop = () => setStream((s: any) => ({ ...s, playing: false }));
  const alerts = stream.rows?.filter((r: any) => r.is_anomaly).length || 0;
  return (
    <div className="chart-card" style={{ marginTop: 12 }}>
      <h4>Real-time replay · fast-path detection on observed values</h4>
      <p className="caption">Replays held-out real observations one-by-one through a fast-path z-score detector. Latency is per-row; alerts are proposals, never confirmed faults.</p>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        {!stream.playing ? (
          <>
            <button className="btn" disabled={loading} onClick={() => start(60, 600)}>{loading ? "Loading…" : "Play 60 min @ 10×"}</button>
            <button className="btn ghost" disabled={loading} onClick={() => start(120, 1200)}>Play 2 h @ 60×</button>
            <button className="btn ghost" disabled={loading} onClick={() => start(30, 60)}>Play 30 min @ 1× (real-time pace)</button>
          </>
        ) : (
          <button className="btn ghost" onClick={stop}>Stop</button>
        )}
        {stream.summary && (
          <span className="caption">
            {stream.summary.rows} rows · {alerts} alerts · latency ≈ {stream.summary.latency_us_mean} μs · {stream.summary.policy}
          </span>
        )}
      </div>
      {stream.playing && stream.rows.length > 0 && (
        <div style={{ maxHeight: 220, overflow: "auto", marginTop: 8, border: "1px solid var(--border)", borderRadius: 12 }}>
          <table className="data">
            <thead>
              <tr><th>Time (UTC)</th><th>Score</th><th>Threshold</th><th>Top channel</th><th>Alert</th></tr>
            </thead>
            <tbody>
              {stream.rows.slice(0, 100).map((r: any, i: number) => (
                <tr key={i} style={{ background: r.is_anomaly ? "color-mix(in srgb, " + p.alert + " 12%, transparent)" : undefined }}>
                  <td>{fmt(r.ts)}</td>
                  <td>{r.score}</td><td>{r.threshold}</td>
                  <td>{r.top_channel ?? "—"}</td>
                  <td>{r.is_anomaly ? "🔴" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}