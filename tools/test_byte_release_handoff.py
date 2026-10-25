"""CPU-only archive and saved-evidence checks; no native imports or network."""
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_handoff as handoff
import run_byte_release_qualification as runner
from test_byte_release_artifacts import wheel as fake_wheel
from test_byte_release_contract import fabricated_admission, fabricated_result
from test_run_byte_release_qualification import fake_round

c = handoff.contract


def accepted_ledger(root):
    wheel = fake_wheel(root)
    admission = fabricated_admission()
    admission["wheels"]["macos"] = handoff.artifacts.wheel_inventory(wheel)
    admission["builds"]["macos"]["wheel_sha256"] = admission["wheels"]["macos"]["sha256"]
    ledger = root / "ledger"
    c.initialize_ledger(ledger)
    directory = c.admit(ledger, admission)
    environment = c.claim_environment(directory, "local-metal")
    environment.mkdir()
    schedule = fabricated_result(admission)
    c.save(environment / "start.json", dict(schedule, status="incomplete", rounds=[]))
    for index in range(3):
        location = environment / f"round-{index + 1}"
        record = fake_round(location, admission, {"synthetic": True})
        record["round"] = index + 1
        (location / "worker-result.json").write_bytes(c.encoded(record))
        with patch.object(c, "source_identity", return_value=admission["source"]):
            sha = runner.verify_round(location, index, admission, {"synthetic": True})
        row = schedule["rounds"][index]
        row["evidence_sha256"] = sha
        c.save(location / "result.json", row)
    c.save(environment / "result.json", schedule)
    c.finish_environment(directory, "local-metal", schedule)
    return ledger, wheel, admission


class HandoffOffline(unittest.TestCase):
    def test_accepted_local_chain_and_complete_packet_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, wheel, admission = accepted_ledger(root)
            self.assertEqual(handoff.verify_environment(ledger / "candidate-1", "local-metal"), admission)
            archive = root / "input.zip"
            receipt = handoff.pack_input(ledger, wheel, archive)
            output = root / "decoded"
            manifest = handoff.unpack(archive, output, receipt["archive_sha256"], "input", c.digest(admission))
            self.assertIn("wheel/" + wheel.name, manifest["files"])
            self.assertEqual(handoff.verify_environment(output / "ledger/candidate-1", "local-metal"), admission)
            self.assertFalse((ledger / "candidate-1/hosted-metal.start.json").exists())
            self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())
            self.assertFalse(receipt["release_ready"])
            with self.assertRaises(FileExistsError):
                handoff.unpack(archive, output, receipt["archive_sha256"], "input", c.digest(admission))

    def test_changed_round_bytes_and_receipt_do_not_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, _wheel, _admission = accepted_ledger(root)
            directory = ledger / "candidate-1"
            path = directory / "local-metal/round-1/worker-result.json"
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                handoff.verify_environment(directory, "local-metal")
            path.write_bytes(original)
            receipt = directory / "local-metal.result.json"
            value = c.read(receipt)
            value["evidence_sha256"] = "0" * 64
            receipt.write_bytes(c.encoded(value))
            with self.assertRaisesRegex(ValueError, "acceptance receipt"):
                handoff.verify_environment(directory, "local-metal")

    def test_resealed_result_cannot_borrow_the_other_environment_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, _wheel, _admission = accepted_ledger(root)
            directory = ledger / "candidate-1"
            result_path = directory / "local-metal/result.json"
            receipt_path = directory / "local-metal.result.json"
            result, receipt = c.read(result_path), c.read(receipt_path)
            result["budget_seconds"] = 14400
            receipt["evidence_sha256"] = c.digest(result)
            result_path.write_bytes(c.encoded(result))
            receipt_path.write_bytes(c.encoded(receipt))
            with self.assertRaisesRegex(ValueError, "acceptance receipt"):
                handoff.verify_environment(directory, "local-metal")

    def test_hosted_reservation_cannot_be_repacked_as_a_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, wheel, _admission = accepted_ledger(root)
            c.save(ledger / "candidate-1/hosted-metal.dispatch.json", {"synthetic": True})
            with self.assertRaisesRegex(ValueError, "already reserved"):
                handoff.pack_input(ledger, wheel, root / "forbidden.zip")

    def test_wrong_outer_digest_kind_and_admission_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "data.json"
            source.write_bytes(b"{}")
            archive = root / "result.zip"
            receipt = handoff.pack({"data.json": source}, archive, "result", "a" * 64)
            for index, (sha, kind, admission) in enumerate((("0" * 64, "result", "a" * 64),
                    (receipt["archive_sha256"], "input", "a" * 64),
                    (receipt["archive_sha256"], "result", "b" * 64))):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    handoff.unpack(archive, root / str(index), sha, kind, admission)

    def test_unsafe_archive_members_and_incomplete_manifest_reject(self):
        for defect in ("traversal", "link", "duplicate", "missing", "changed", "extra", "private_key"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                member = "../escape" if defect == "traversal" else "input-key.txt" if defect == "private_key" else "data.json"
                manifest = dict(schema=handoff.SCHEMA, kind="result", admission_sha256="a" * 64,
                                release_ready=False, files={member: {"bytes": 2, "sha256": c.digest({})}})
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output, warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    output.writestr(handoff.MANIFEST, c.encoded(manifest))
                    if defect != "missing":
                        info = zipfile.ZipInfo(member)
                        if defect == "link":
                            info.create_system = 3
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        output.writestr(info, b"[]" if defect == "changed" else b"{}")
                    if defect == "duplicate":
                        output.writestr(member, b"{}")
                    if defect == "extra":
                        output.writestr("extra.json", b"{}")
                with self.assertRaises(ValueError):
                    handoff.unpack(archive, root / "decoded", handoff.artifacts.sha256(archive), "result", "a" * 64)

    def test_pinned_file_change_and_limits_fail_before_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "data.json"
            source.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                handoff.pack({"data.json": source}, root / "wrong.zip", "result", "a" * 64,
                             {"data.json": {"bytes": 2, "sha256": "0" * 64}})
            with patch.object(handoff, "MAX_TOTAL", 1), self.assertRaisesRegex(ValueError, "total size"):
                handoff.pack({"data.json": source}, root / "large.zip", "result", "a" * 64)
            source.unlink()
            source.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                handoff.pack({"data.json": source}, root / "link.zip", "result", "a" * 64)


if __name__ == "__main__":
    unittest.main()
