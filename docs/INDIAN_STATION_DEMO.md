# Indian station scenario demonstration

The India page uses separately prepared original Indian NOAA ISD reports. It
preserves temperature/dew-point/sea-level-pressure raw fields, provider quality
codes, timestamps, duplicate reports and raw-file hashes. The humidity baseline
is derived from temperature/dew point, not an independently measured RH input.
Sea-level pressure is not station pressure. These airport/surface reports do
not establish validation on IMD AWS instruments.

The controls compare the unchanged baseline with visibly synthetic spike,
drift, stuck-value and dropout copies. Markers identify deliberately modified
software values, not detector hits. Missing original readings stay missing and
the CSV download includes the synthetic demonstration notice.

Create the bundle with `python scripts/build_indian_demo.py --help` and the
documented CLI arguments for the available local originals. The active API uses
`data/indian_demo_20260929` by default; `SKYGUARD_INDIAN_DEMO` selects another
prepared bundle. Bundle loaders verify their manifest and file hashes.

The separately authorized September 30 scenario-model benchmark uses original
measured SURFRAD T/RH/station-pressure windows. The Indian demonstration is not
silently mixed into that model or used to claim Indian hardware-fault accuracy.
