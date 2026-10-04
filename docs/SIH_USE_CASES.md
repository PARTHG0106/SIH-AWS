# Current operational use cases

## Sudden sensor-report change

A new minute arrives with an abrupt change. The frozen detector compares it to
past readings and causal predictions. The alert identifies channel, UTC time,
signal value, frozen threshold, exceedance and inspection action. Its score is
not a failure probability. The synthetic-trained pattern model may separately
suggest a spike/step pattern with its scenario-calibration confidence.

## Genuine rapid weather change

Several channels change together. Their temporal relationships contribute to
the learned pattern features; the real detector can still flag unusual weather.
The interface preserves the original values and asks for weather/logger context.
It does not automatically diagnose equipment failure from a rare observation.
Sparse, distant SURFRAD stations are not treated as a dense local buddy network.

## Stuck or quantized readings

Exact-repeat duration is accumulated as observations arrive, including across
packet boundaries. Missing values and timestamp gaps reset continuity. The
inspection action explicitly considers sensor resolution and steady weather.
Synthetic stuck/clipping cases test software recognition; they are not hardware
maintenance records.

## Sensor fields missing or packets overdue

A missing field remains null. Its unavailable score cannot count as healthy.
When one-minute cadence has been explicitly configured, a heartbeat reports
overdue expected slots even if no new observation arrives. It never inserts a
measurement. The operator is advised to check the feed/logger/transport before
assigning a sensor cause. Delayed original packets can resolve overdue slots.

## Possible degradation requiring inspection

The health summary reports candidate activity on scored rows, coverage and the
change between elapsed-time halves. With sufficient history/coverage, rising or
persistent activity produces an explicit inspection priority and evidence.
This predicts a need for review according to a declared service policy, not a
remaining hardware lifetime or a confirmed failure date.

## Traceability and review

An operator opens source provenance/QC for a candidate. Original row IDs and
hashes remain available; predictions and synthetic values are separate fields.
Independent review uses the existing evidence-backed event workflow. Unknown
cases remain unknown. Suggested correction is optional under SIH and is not
automatically applied to any original observation.
