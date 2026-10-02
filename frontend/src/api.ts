// Typed fetch helpers for the SkyGuard Starlette API. Relative "/api" works both in
// dev (Vite proxy → :8600) and in production (same-origin, served by uvicorn).
export type Pair = [number, number | null];
export type Cand = [number, number | null, string];
export type Point = [number, number | null, number | null];

export interface Station { group: string; station_id: string; source: string; records: number; first: string; last: string; }
export interface UsaCatalog { artifact_dir: string; policy: string | null; stations: Station[]; metrics: Record<string, unknown>; detector_config: Record<string, unknown>; }
export interface UsaEvent { event_id: string; start: string; end: string; channel: string; max_score: number | null; reason_codes: string; }
export interface UsaChannel { key: string; label: string; unit: string; observed: Pair[]; model_1m: Pair[]; model_60m: Pair[]; candidates: Cand[]; }
export interface UsaSummary { records: number; candidates: number; scored: number; missing_values: number; unreported_slots: number; }
export interface UsaCandidate { timestamp: string; reason_codes: string | null; anomaly_score: number | null; scoring_status?: string; fault_type?: string; type_confidence?: number | null; }
export interface UsaWindow { group: string; start: string; end: string; summary: UsaSummary; channels: UsaChannel[]; candidates: UsaCandidate[]; gaps: { from: string; to: string; unreported: number }[]; }
export interface MaintenanceAdvisory {
  state: string; priority: string; action: string; history_sufficient_for_policy: boolean; channel?: string;
  evidence: { scored_rows: number; received_rows: number; window_minutes: number | null; scored_fraction: number | null; candidate_rate: number | null };
  policy: { minimum_scored_rows: number; minimum_window_minutes: number; minimum_scored_fraction: number; maximum_missing_fraction: number };
  semantics: string;
}
export interface ChannelHealth { state: string; trend: string; candidate_rate: number | null; alerts: number; scored_rows?: number; missing_rows?: number; missing_fraction?: number | null; dominant_reason: string | null; maintenance_advisory?: MaintenanceAdvisory; }
export interface StationHealth { overall_state: string; rows: number; channels: Record<string, ChannelHealth>; semantics: string; maintenance_advisory?: MaintenanceAdvisory; }
export interface UsaHealth { group: string; window_days?: number; health: StationHealth | null; }

export interface InStation { station_id: string; name: string; latitude: number | null; longitude: number | null; elevation_m: number | null; rows: number | null; start: string; end: string; }
export interface InCatalog { artifact_dir: string; manifest: { dataset_policy?: string }; stations: InStation[]; scenarios: { key: string; label: string }[]; channels: { key: string; label: string }[]; }
export interface InChannel { key: string; label: string; source_label: string; source: Point[]; synthetic: Point[]; applied: Point[]; }
export interface InSummary { records: number; modified_records: number; modified_values: number; missing_baseline_values: number; duplicate_timestamp_rows: number; }
export interface InScenario { station: string; ready: boolean; summary: InSummary; scenario_start: number; window_timestamps: number[]; channels: InChannel[]; }

export interface LiveObservation {
  timestamp: string; station_id: string; source: string; observation_id?: string;
  temperature_c: number | null; pressure_hpa: number | null; relative_humidity_pct: number | null;
  synthetic?: boolean; scenario_modified?: boolean; scenario_label?: string; scenario_id?: string;
  baseline?: Record<string, unknown>; baseline_observation_id?: string; baseline_group?: string;
  [key: string]: unknown;
}
export interface LiveExplanation { channel: string; signal: string; pattern: string; reason: string; action: string; confidence: number | null; }
export interface PhysicsAlert { channel: string; signal: string; reason: string; action: string; hardware_fault_status: string; }
export interface PatternEvidence {
  status?: string; reason?: string; scenario_score?: number | null; is_candidate?: boolean; suggested_pattern?: string;
  pattern_confidence?: number | null; threshold?: number; required_contiguous_rows?: number;
  confidence_semantics?: string; model_version?: string;
}
export interface LiveResult extends LiveObservation {
  anomaly_score: number | null; is_candidate: boolean; reason_codes: string; scoring_status: string;
  hardware_fault_status: string; severity: "info" | "review" | "high";
  explanations: LiveExplanation[]; physics_alerts?: PhysicsAlert[]; pattern_evidence?: PatternEvidence;
  availability: { missing_channels: string[]; unreported_slots_before: number; expected_cadence_minutes: number };
}
export interface LiveGroup {
  latest: LiveResult | null; health: StationHealth | null; rows_seen: number; last_timestamp: string | null;
  buffered_observations: number; availability: Record<string, unknown>;
}
export interface LiveSnapshot { groups: Record<string, LiveGroup>; limits?: Record<string, unknown>; model_info?: Record<string, unknown>; }
export interface LiveSession { session_id: string; snapshot: LiveSnapshot; }
export interface LiveBatch { session_id: string; results: LiveResult[]; snapshot: LiveSnapshot; processing_ms?: number; }
export interface LiveReplay { observations: LiveObservation[]; group: string; rows: number; start: string; end: string; semantics: string; warmup_minutes: { forecast_1m: number; forecast_60m: number; sustained: number }; }
export interface LiveScenario extends LiveReplay {
  mode: "synthetic_demo"; baseline_group: string; scenario: string; channel: string; seed: number; threshold_reference: string;
  event: { scenario_id: string; scenario: string; channels: string[]; scheduled_start: string; scheduled_end_exclusive: string;
    first_modified: string | null; last_modified: string | null; modified_rows: number; modified_values: number;
    parameters: Record<string, unknown>; semantics: string };
}

export interface DetectionMetrics {
  precision: number; recall: number; f1: number; pr_auc?: number; roc_auc?: number;
  tp: number; fp: number; fn: number; tn: number; rows: number; unmodified_row_candidate_rate: number | null;
}
export interface ScenarioMetrics {
  modified_rows: number; row_recall: number | null; events: number; detected_events: number; event_recall: number | null;
  median_latency_min: number | null; p95_latency_min: number | null; not_applied_events: number;
}
export interface ClassificationMetrics {
  labels: string[]; macro_f1_faults: number; accuracy: number; confusion_matrix: number[][];
  per_class: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
}
export interface ScenarioReport {
  point: DetectionMetrics; per_fault: Record<string, ScenarioMetrics>; classification?: ClassificationMetrics;
  unmodified_baseline_candidate_rate: number | null; semantics?: string;
  reliability?: { brier: number; ece: number; semantics?: string; bins: {
    lower: number; upper: number; rows: number; predicted_frequency: number; observed_modification_frequency: number;
  }[] };
}
export interface BenchmarkPartition { model: ScenarioReport; baselines: Record<string, ScenarioReport>; source_windows: number; rows: number; }
export interface ScenarioBenchmark {
  schema_version: "sih_benchmark_v2"; policy: string; status: string; disclaimer?: string;
  split_audit: { unique_source_windows: number; overlapping_source_windows: number; split_before_generation: boolean; context_policy: string };
  selection: ScenarioReport; calibration: ScenarioReport;
  final_test: { partitions: Record<string, BenchmarkPartition>; evaluated_at_utc?: string; frozen_sha256?: string } | null;
}
export interface LegacyBenchmark {
  schema_version?: string; point?: Partial<DetectionMetrics>; clean_candidate_rate?: number;
  per_fault?: Record<string, { row_recall?: number; event_recall?: number; median_latency_min?: number | null; rows?: number }>;
}
export interface BenchmarkResponse { available: boolean; metrics: ScenarioBenchmark | LegacyBenchmark | null; classifier?: { macro_f1_faults?: number } | null; }

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`/api${path}${qs}`);
  if (!res.ok) throw await responseError(res, path);
  return res.json();
}

async function responseError(res: Response, path: string): Promise<Error> {
  let detail = "";
  try {
    const payload: { error?: unknown; detail?: unknown } = await res.json();
    detail = typeof payload.error === "string" ? payload.error : typeof payload.detail === "string" ? payload.detail : "";
  } catch { /* HTTP status remains useful if the server cannot return JSON. */ }
  return new Error(detail || `${path} → ${res.status}`);
}

async function mutate<T>(path: string, method: "POST" | "DELETE", body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await responseError(res, path);
  return res.status === 204 ? undefined as T : res.json();
}

export const api = {
  usaCatalog: () => get<UsaCatalog>("/usa/catalog"),
  usaEvents: (group: string) => get<{ group: string; events: UsaEvent[] }>("/usa/events", { group }),
  usaWindow: (group: string, start: string, end: string) => get<UsaWindow>("/usa/window", { group, start, end }),
  usaHealth: (group: string) => get<UsaHealth>("/usa/health", { group }),
  usaStream: (group: string, minutes: number, speed: number) => get<any>("/usa/stream", { group, minutes: String(minutes), speed: String(speed) }),
  usaSpatial: (group: string) => get<any>("/usa/spatial", { group }),
  usaCorrections: (group: string) => get<any>("/usa/corrections", { group }),
  indiaCatalog: () => get<InCatalog>("/india/catalog"),
  indiaScenario: (p: Record<string, string>) => get<InScenario>("/india/scenario", p),
  indiaCsvUrl: (p: Record<string, string>) => `/api/india/scenario.csv?${new URLSearchParams(p).toString()}`,
  benchmark: () => get<BenchmarkResponse>("/benchmark"),
  liveReplay: (group: string, start: string, end: string) => get<LiveReplay>("/live/replay", { group, start, end }),
  liveScenario: (group: string, start: string, end: string, scenario: string, channel: string) => get<LiveScenario>("/live/scenario", { group, start, end, scenario, channel, seed: "26073" }),
  liveCreate: (group: string) => mutate<LiveSession>("/live/sessions", "POST", { group, expected_cadence_minutes: 1 }),
  liveObserve: (sessionId: string, observations: LiveObservation[]) => mutate<LiveBatch>(`/live/sessions/${encodeURIComponent(sessionId)}/observations`, "POST", { observations }),
  liveDelete: (sessionId: string) => mutate<unknown>(`/live/sessions/${encodeURIComponent(sessionId)}`, "DELETE"),
};
