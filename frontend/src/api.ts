// Typed fetch helpers for the SkyGuard Starlette API. Relative "/api" works both in
// dev (Vite proxy → :8600) and in production (same-origin, served by uvicorn).
export type Pair = [number, number | null];
export type Cand = [number, number | null, string];
export type Point = [number, number | null, number | null];

export interface Station { group: string; station_id: string; source: string; records: number; first: string; last: string; }
export interface UsaCatalog { artifact_dir: string; policy: string | null; stations: Station[]; metrics: any; detector_config: any; }
export interface UsaEvent { event_id: string; start: string; end: string; channel: string; max_score: number | null; reason_codes: string; }
export interface UsaChannel { key: string; label: string; unit: string; observed: Pair[]; model_1m: Pair[]; model_60m: Pair[]; candidates: Cand[]; }
export interface UsaSummary { records: number; candidates: number; scored: number; missing_values: number; unreported_slots: number; }
export interface UsaWindow { group: string; start: string; end: string; summary: UsaSummary; channels: UsaChannel[]; candidates: any[]; gaps: any[]; }

export interface InStation { station_id: string; name: string; latitude: number | null; longitude: number | null; elevation_m: number | null; rows: number | null; start: string; end: string; }
export interface InCatalog { artifact_dir: string; manifest: any; stations: InStation[]; scenarios: { key: string; label: string }[]; channels: { key: string; label: string }[]; }
export interface InChannel { key: string; label: string; source_label: string; source: Point[]; synthetic: Point[]; applied: Point[]; }
export interface InSummary { records: number; modified_records: number; modified_values: number; missing_baseline_values: number; duplicate_timestamp_rows: number; }
export interface InScenario { station: string; ready: boolean; summary: InSummary; scenario_start: number; window_timestamps: number[]; channels: InChannel[]; }

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`/api${path}${qs}`);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export const api = {
  usaCatalog: () => get<UsaCatalog>("/usa/catalog"),
  usaEvents: (group: string) => get<{ group: string; events: UsaEvent[] }>("/usa/events", { group }),
  usaWindow: (group: string, start: string, end: string) => get<UsaWindow>("/usa/window", { group, start, end }),
  usaHealth: (group: string) => get<any>("/usa/health", { group }),
  indiaCatalog: () => get<InCatalog>("/india/catalog"),
  indiaScenario: (p: Record<string, string>) => get<InScenario>("/india/scenario", p),
  indiaCsvUrl: (p: Record<string, string>) => `/api/india/scenario.csv?${new URLSearchParams(p).toString()}`,
  benchmark: () => get<any>("/benchmark"),
};
