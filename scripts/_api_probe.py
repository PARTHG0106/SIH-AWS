"""Ad-hoc in-process probe of app.api endpoints (no server/curl needed)."""
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from app import api


def call(fn, **params):
    resp = fn(SimpleNamespace(query_params=params))
    return json.loads(bytes(resp.body))


def main():
    cat = call(api.usa_catalog)
    print("USA policy:", cat.get("policy"), "| stations:", len(cat["stations"]))
    g = max(cat["stations"], key=lambda s: s["records"])
    print("  group:", g["group"], "records:", g["records"], g["first"], "->", g["last"])
    ev = call(api.usa_events, group=g["group"])
    print("  events:", len(ev["events"]))
    if ev["events"]:
        e = ev["events"][0]
        start, end = e["start"], e["end"]
    else:
        end = g["last"]; start = end[:10] + "T00:00:00Z"
    win = call(api.usa_window, group=g["group"], start=start, end=end)
    print("  window summary:", win["summary"])
    for ch in win["channels"]:
        print(f"    {ch['key']}: obs={len(ch['observed'])} m1={len(ch['model_1m'])} "
              f"m60={len(ch['model_60m'])} cand={len(ch['candidates'])} unit={ch['unit']!r}")
    print("  candidate rows:", len(win["candidates"]), "| gaps:", len(win["gaps"]))

    icat = call(api.india_catalog)
    print("\nINDIA policy:", icat["manifest"].get("dataset_policy"), "| stations:", len(icat["stations"]),
          "| scenarios:", [s["key"] for s in icat["scenarios"]])
    st = icat["stations"][0]
    print("  station:", st["name"], st["station_id"], st["start"], "->", st["end"])
    start_day = (st["start"] or "2024-01-01")[:10]
    sc = call(api.india_scenario, station=st["station_id"], start=start_day, days="3",
              scenario="spike", channel="baseline_temperature_c", duration="12", magnitude="8")
    print("  scenario summary:", sc["summary"], "ready:", sc["ready"])
    for ch in sc["channels"]:
        print(f"    {ch['key']}: source={len(ch['source'])} synth={len(ch['synthetic'])} applied={len(ch['applied'])}")
    print("OK")


if __name__ == "__main__":
    main()
