"""Contract software fixtures only: none are eligible observations or datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.data.training_contract import (
    CONTRACT_FILENAME, EVIDENCE_ROLES, REQUIRED_INPUTS,
    TrainingDataContractError, inspect_training_data,
    require_eligible_training_data, write_training_report,
)


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()

    def write_json(self, name, payload):
        (self.processed / name).write_text(json.dumps(payload), encoding="utf-8")

    def codes(self, report):
        return {item["code"] for item in report["blockers"]}

    def declaration_fixture(self):
        """Untrusted statements plus integrity refs; never provider evidence."""
        evidence = self.processed / "test_only_evidence.txt"
        evidence.write_text("Software fixture, not observations or source documentation.", encoding="utf-8")
        sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
        return {
            "schema_version": 1,
            "evidence": {role: {"path": evidence.name, "sha256": sha} for role in EVIDENCE_ROLES},
            "inputs": {
                "temperature_c": {"origin": "observed"},
                "pressure_hpa": {"origin": "observed", "quantity": "station_pressure"},
                "relative_humidity_pct": {"origin": "observed", "measurement": "independently_measured"},
            },
            "missing_observations": "preserved",
            "labels": {"unknown": "preserved", "provider_qc_is_fault_truth": False,
                       "normal_requires_independent_review": True},
        }

    def test_legacy_bundle_is_refused_without_reading_observations(self):
        self.write_json("source_metadata.json", {
            "status": "legacy_local_snapshot",
            "rh_policy": "NOAA RH derived from TMP+DEW; Open-Meteo RH reanalysis",
            "pressure_policy": "SLP and pressure_msl are sea-level pressure",
        })
        (self.processed / "injection_events.parquet").write_bytes(b"test fixture")
        (self.processed / "train_clean.parquet").write_bytes(b"test fixture")
        original_open = Path.open

        def no_parquet_open(path, *args, **kwargs):
            if path.suffix == ".parquet":
                raise AssertionError("Preflight must not read observation arrays")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", no_parquet_open):
            report = inspect_training_data(self.processed)
        self.assertFalse(report["eligible"])
        self.assertTrue({"legacy_dataset", "derived_relative_humidity", "sea_level_pressure",
                         "reanalysis_input", "injected_dataset", "training_contract_missing"} <= self.codes(report))
        self.assertFalse(report["verification"]["observations_loaded"])

    def test_exception_keeps_machine_readable_report_and_writes_json(self):
        path = self.root / "outputs" / "preflight.json"
        with self.assertRaises(TrainingDataContractError) as caught:
            require_eligible_training_data(self.processed, report_path=path)
        written = json.loads(path.read_text())
        self.assertEqual(written, caught.exception.report)
        self.assertIn("Real-observation training preflight failed", str(caught.exception))

    def test_self_declared_eligibility_cannot_enable_training(self):
        index = self.declaration_fixture()
        index.update({"eligible": True, "verified": True, "source_verified": True})
        self.write_json(CONTRACT_FILENAME, index)
        report = inspect_training_data(self.processed)
        self.assertEqual(self.codes(report), {"provider_verifier_unimplemented"})
        self.assertFalse(report["eligible"])
        self.assertFalse(report["verification"]["raw_to_processed_lineage_verified"])

    def test_non_observed_required_inputs_remain_ineligible(self):
        for origin in ("derived", "interpolated", "reanalysis", "predicted", "edited", "constant"):
            with self.subTest(origin=origin):
                index = self.declaration_fixture()
                for field in REQUIRED_INPUTS:
                    index["inputs"][field]["origin"] = origin
                self.write_json(CONTRACT_FILENAME, index)
                report = inspect_training_data(self.processed)
                self.assertIn("non_observed_input", self.codes(report))
                self.assertFalse(report["eligible"])

    def test_derived_rh_and_slp_cannot_be_renamed_to_required_inputs(self):
        index = self.declaration_fixture()
        index["inputs"]["relative_humidity_pct"]["measurement"] = "calculated_from_dewpoint"
        index["inputs"]["pressure_hpa"]["quantity"] = "sea_level_pressure"
        self.write_json(CONTRACT_FILENAME, index)
        codes = self.codes(inspect_training_data(self.processed))
        self.assertTrue({"unverified_rh_measurement", "unverified_pressure_quantity"} <= codes)

    def test_unknown_labels_are_allowed_without_manufactured_normal_labels(self):
        index = self.declaration_fixture()
        self.write_json(CONTRACT_FILENAME, index)
        report = inspect_training_data(self.processed)
        self.assertEqual(self.codes(report), {"provider_verifier_unimplemented"})
        self.assertTrue(report["label_policy"]["unknown_allowed"])
        self.assertFalse(report["label_policy"]["unknown_is_normal"])
        self.assertFalse(report["label_policy"]["supervised_fault_accuracy_eligible"])
        del index["labels"]
        self.write_json(CONTRACT_FILENAME, index)
        self.assertEqual(self.codes(inspect_training_data(self.processed)), {"provider_verifier_unimplemented"})

    def test_normal_by_default_and_qc_truth_are_rejected(self):
        index = self.declaration_fixture()
        index["labels"] = {"unknown": "normal", "provider_qc_is_fault_truth": True,
                           "normal_requires_independent_review": False}
        self.write_json(CONTRACT_FILENAME, index)
        codes = self.codes(inspect_training_data(self.processed))
        self.assertTrue({"unknown_labels_not_preserved", "qc_not_fault_truth", "normal_label_without_review"} <= codes)

    def test_missing_evidence_hash_mismatch_and_escape_are_reported(self):
        index = self.declaration_fixture()
        del index["evidence"]["acquisition_manifest"]
        index["evidence"]["field_lineage"]["sha256"] = "0" * 64
        index["evidence"]["raw_file_manifest"]["path"] = "../outside.txt"
        self.write_json(CONTRACT_FILENAME, index)
        codes = self.codes(inspect_training_data(self.processed))
        self.assertTrue({"missing_evidence", "evidence_hash_mismatch", "unsafe_evidence_path"} <= codes)

    def test_malformed_metadata_and_missing_directory_fail_closed(self):
        (self.processed / "source_metadata.json").write_text("{", encoding="utf-8")
        self.assertIn("unreadable_metadata", self.codes(inspect_training_data(self.processed)))
        self.assertIn("processed_directory_missing", self.codes(inspect_training_data(self.root / "absent")))

    def test_report_cannot_overwrite_dataset_input(self):
        report = inspect_training_data(self.processed)
        with self.assertRaises(ValueError):
            write_training_report(report, self.processed / "source_metadata.json")
        self.assertFalse((self.processed / "source_metadata.json").exists())

    def test_cli_reports_blocked_exit_code_without_ml_dependencies(self):
        output = self.root / "preflight.json"
        completed = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / "preflight_training.py"),
                                    "--processed-dir", str(self.processed), "--output", str(output)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertFalse(json.loads(output.read_text())["eligible"])
        self.assertEqual(json.loads(completed.stdout)["status"], "blocked")

    def test_newer_builder_log_is_separate_from_remote_data_verification(self):
        log = self.root / "builder.txt"
        raw = ("14s\t10\tsource manifest sha256: " + "a" * 64 + "\n"
               "14s\t11\tbuild id: 20260925T105454Z-example | build UTC: 2026-09-25T10:54:54+00:00\n"
               "15s\t12\tlabel provenance: {'synthetic_fraction_by_split': {'val': 0.96}}\n").encode()
        log.write_bytes(raw)
        output = self.root / "preflight.json"
        completed = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / "preflight_training.py"),
                                    "--processed-dir", str(self.processed), "--output", str(output),
                                    "--builder-log", str(log)], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        evidence = json.loads(output.read_text())["builder_run_log"]
        self.assertEqual(evidence["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(evidence["reported"]["build_id"], "20260925T105454Z-example")
        self.assertEqual(evidence["evidence_lines"]["label_provenance"], 3)
        self.assertIn("not downloaded or verified", evidence["verification_scope"])


if __name__ == "__main__":
    unittest.main()
