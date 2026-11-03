import sys
from pathlib import Path
import unittest
import os
import math
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_binding import frozen, match_rows, section_identity
from interchange_supervisor import admissible_result, identify, publish, supervise
from joint_interchange import returned
from joint_protocol import HERE, read, save, sha


def primary_args(root, result=None):
    out = root / "run"
    out.mkdir()
    save(out / "results.json", {} if result is None else result)
    pin = sha(out / "results.json")
    save(out / "status.json", {"status": "complete", "results_sha256": pin})
    args = SimpleNamespace(out=out, result_pin=pin, receipt=root / "export.json")
    identify(args)
    return args


def replay_reports(a, pin, _seconds, edit=None):
    x = read(HERE / "interchange_results.json")
    save(a.attempt / "worker-start.json", {"claim_sha256": pin})
    save(a.attempt / "candidate.json", x)
    reports = a.attempt / "native-scores"
    reports.mkdir()
    for section, group in read(HERE / "score_roster.json")["groups"].items():
        book, i = section.rsplit("-", 1)
        rows = []
        for name in x["returned_order"]:
            lora, geometry = x["owners_lora_then_geometry"][name]
            nll = x["section_nll"][book][i][name]
            rows.append(
                {
                    **group["common"],
                    "lora_artifact_id": lora,
                    "geometry_artifact_id": geometry,
                    "nll": nll,
                    "perplexity": math.exp(nll),
                }
            )
        report = {
            "schema": "spiraltorch.native_causal_lm_evaluation_comparison.v1",
            "shared_prefix_layers": 8,
            "evaluations": rows,
        }
        if edit is not None:
            edit(report)
        save(reports / (section + ".json"), report)


class Bindings(unittest.TestCase):
    def test_receipt_cannot_alias_private_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = primary_args(root)
            alt = root / "different-parent"
            alt.symlink_to(a.out, target_is_directory=True)
            alias = a.out / ".INTERCHANGE-CLAIMS" / a.attempt.name / "candidate.json"
            for path in (alias, alt / "candidate.json"):
                a.receipt = path
                with (
                    self.subTest(path=path),
                    self.assertRaises((ValueError, FileNotFoundError)),
                ):
                    identify(a)
            a.attempt.mkdir(parents=True)
            a.receipt = alias
            with self.assertRaises((ValueError, FileNotFoundError)):
                identify(a)
            a.receipt = root / "export.json"
            a.receipt.symlink_to(a.attempt / "candidate.json")
            with self.assertRaises(ValueError):
                identify(a)
            self.assertFalse((a.attempt / "worker-start.json").exists())

    def test_registry_cannot_be_relocated_under_an_external_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = primary_args(root)
            destination = root / "elsewhere"
            destination.mkdir()
            a.attempt.parent.symlink_to(destination, target_is_directory=True)
            a.receipt = destination / a.attempt.name / "candidate.json"
            with self.assertRaisesRegex(ValueError, "claim registry"):
                identify(a)

    def test_failed_secondary_worker_retains_its_only_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = primary_args(Path(tmp))
            cmd = [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).with_name("joint_interchange.py")),
                "--receipt",
                str(a.receipt),
                "--result-pin",
                a.result_pin,
                "--out",
                str(a.out),
            ]
            for k in ("data", "old", "model", "wheel"):
                cmd.extend(["--" + k, str(Path(tmp) / ("absent-" + k))])
            self.assertNotEqual(
                subprocess.run(cmd, capture_output=True, timeout=15).returncode, 0
            )
            attempt = a.attempt
            first = (attempt / "claim.json").read_bytes()
            self.assertEqual(
                read(attempt / "claim.json")["limits"]["per_worker_seconds"], 2400
            )
            self.assertTrue((attempt / "worker-start.json").is_file())
            first_error = (attempt / "first-error.json").read_bytes()
            alt = Path(tmp) / "alternate.json"
            cmd[cmd.index("--receipt") + 1] = str(alt)
            self.assertNotEqual(
                subprocess.run(cmd, capture_output=True, timeout=15).returncode, 0
            )
            self.assertEqual((attempt / "claim.json").read_bytes(), first)
            self.assertEqual((attempt / "first-error.json").read_bytes(), first_error)
            self.assertEqual(list(attempt.parent.iterdir()), [attempt])
            self.assertFalse(a.receipt.exists() or alt.exists())

    def test_staged_candidate_after_timeout_is_not_a_result(self):
        def late_worker(a, pin, _seconds):
            save(a.attempt / "worker-start.json", {"claim_sha256": pin})
            save(a.attempt / "candidate.json", {"primary_result_sha256": a.result_pin})
            raise TimeoutError("fixture: expired after staging")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            a = primary_args(Path(tmp))
            with self.assertRaises(TimeoutError):
                supervise(a, launch=late_worker)
            self.assertTrue((a.attempt / "candidate.json").is_file())
            self.assertEqual(
                read(a.attempt / "first-error.json")["type"], "TimeoutError"
            )
            self.assertFalse((a.attempt / "status.json").exists() or a.receipt.exists())
            a.receipt = Path(tmp) / "another.json"
            identify(a)
            with self.assertRaisesRegex(ValueError, "failed attempt"):
                publish(a)
            with self.assertRaisesRegex(ValueError, "failed attempt"):
                admissible_result(a.attempt, a.result_pin)
            with self.assertRaises(FileExistsError):
                supervise(a, launch=lambda *_: self.fail("second worker launched"))
            self.assertFalse(a.receipt.exists())

    def test_success_requires_raw_sections_then_exports_without_a_second_worker(self):
        primary = read(HERE / "results.json")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            a = primary_args(Path(tmp), primary)
            supervise(a, launch=replay_reports)
            self.assertEqual(read(a.attempt / "status.json")["attempts"], 1)
            first = a.receipt
            self.assertEqual(
                first.read_bytes(), (HERE / "interchange_results.json").read_bytes()
            )
            a.receipt = Path(tmp) / "alternate.json"
            identify(a)
            with self.assertRaises(FileExistsError):
                supervise(a, launch=lambda *_: self.fail("second worker launched"))
            publish(a)
            with self.assertRaises(FileExistsError):
                publish(a)
            self.assertEqual(a.receipt.read_bytes(), first.read_bytes())
            # Damaging a retained native report blocks a subsequent export.
            reports = a.attempt / "native-scores"
            next(reports.glob("*.json")).rename(reports / "not-a-chapter")
            a.receipt = Path(tmp) / "unverified.json"
            with self.assertRaisesRegex(ValueError, "raw native section roster"):
                publish(a)
            self.assertFalse(a.receipt.exists())

    def test_partial_or_inaccurate_native_report_cannot_be_sealed(self):
        for edit in (
            lambda r: r.pop("schema"),
            lambda r: r["evaluations"][0].pop("config"),
            lambda r: r["evaluations"][0].update(target_tokens=755),
            lambda r: r["evaluations"][0].update(perplexity=1.0),
        ):
            with (
                self.subTest(edit=edit),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ),
            ):
                a = primary_args(Path(tmp), read(HERE / "results.json"))
                with self.assertRaisesRegex(ValueError, "native report"):
                    supervise(a, launch=lambda *args: replay_reports(*args, edit=edit))
                self.assertFalse(
                    a.receipt.exists() or (a.attempt / "status.json").exists()
                )

    def test_zero_exit_without_native_sections_cannot_complete(self):
        def incomplete(a, pin, _seconds):
            save(a.attempt / "worker-start.json", {"claim_sha256": pin})
            save(a.attempt / "candidate.json", read(HERE / "interchange_results.json"))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            a = primary_args(Path(tmp), read(HERE / "results.json"))
            with self.assertRaisesRegex(ValueError, "raw native section roster"):
                supervise(a, launch=incomplete)
            self.assertTrue((a.attempt / "first-error.json").exists())
            self.assertFalse((a.attempt / "status.json").exists() or a.receipt.exists())

    def test_fixed_chapter_plan_is_shared_but_content_is_not(self):
        a = section_identity([[i] * 64 for i in range(12)])
        b = section_identity([[i + 1] * 64 for i in range(12)])
        self.assertEqual(a["plan_id"], b["plan_id"])
        self.assertNotEqual(a["data_id"], b["data_id"])
        with self.assertRaisesRegex(ValueError, "another chapter"):
            match_rows(a, [b] * 13, 13)
        self.assertEqual(frozen()["sections"]["mansfield"], [5, 13, 21, 29, 37, 45])

    def test_off_diagonal_swap_rejects_even_if_both_diagonals_still_match(self):
        i = {"data_id": "d", "plan_id": "p"}
        names = ["AA", "AW", "WA", "WW"]
        owners = {k: list(k) for k in names}
        rows = [
            {
                **i,
                "lora_artifact_id": k[0],
                "geometry_artifact_id": k[1],
                "nll": float(p),
            }
            for p, k in enumerate(names)
        ]
        self.assertTrue(returned(names, owners, i, rows))
        with self.assertRaisesRegex(ValueError, "returned crossover owner"):
            returned(names, owners, i, [rows[0], rows[2], rows[1], rows[3]])


if __name__ == "__main__":
    unittest.main()
