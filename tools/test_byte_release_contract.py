"""CPU-only candidate/attempt tests. Fabricated identities are not admissions."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_contract as contract
import prepare_byte_release_qualification as preparation
from test_byte_release_artifacts import wheel as fabricated_wheel


def fabricated_admission():
    value = dict(schema=contract.SCHEMA, decision=contract.DECISION, candidate=1,
                 tests=[list(row) for row in contract.TESTS], rounds=3, budget_seconds=5400,
                 long_request_sha256=contract.long_resume.REQUEST_SHA256, release_ready=False,
                 version="0.4.33", source={"commit": "a" * 40, "tree": "b" * 40,
                                          "files": {"Cargo.lock": "c" * 64}}, wheels={}, builds={})
    for index, platform in enumerate(("linux", "macos", "windows")):
        value["wheels"][platform] = dict(platform=platform, version=value["version"], sha256=str(index) * 64,
                                         native={"path": "spiraltorch/spiraltorch.abi3.so", "sha256": "d" * 64})
        value["builds"][platform] = dict(source_commit=value["source"]["commit"],
                                         wheel_sha256=value["wheels"][platform]["sha256"],
                                         receipt_sha256="e" * 64, toolchain="synthetic test only", features=["wgpu"])
    return value


def fabricated_result(admission, environment="local-metal", status="verified_rounds"):
    return dict(schema="spiraltorch.byte_release.environment_schedule.v1",
                admission_sha256=contract.digest(admission), environment=environment,
                planned_rounds=3, budget_seconds=5400, status=status, release_ready=False,
                rounds=[dict(round=i + 1, status="verified", returncode=0, evidence_sha256="f" * 64)
                        for i in range(3)] if status == "verified_rounds" else [])


class AdmissionOffline(unittest.TestCase):
    def test_freeze_inventories_all_final_bytes_and_original_receipts(self):
        source = fabricated_admission()["source"]
        with tempfile.TemporaryDirectory() as temporary, patch.object(contract, "source_identity", return_value=source):
            root = Path(temporary)
            wheels, receipts = {}, {}
            for platform, target in (("linux", "manylinux2014_x86_64"), ("macos", "macosx_14_0_universal2"),
                                     ("windows", "win_amd64")):
                wheels[platform] = fabricated_wheel(root, target=target)
                receipts[platform] = root / (platform + ".json")
                contract.save(receipts[platform], dict(source_commit=source["commit"],
                              wheel_sha256=contract.artifacts.sha256(wheels[platform]),
                              toolchain="synthetic only", features=["wgpu", "logic", "kdsl"]))
            value = preparation.freeze(root, wheels, receipts)
            self.assertEqual(value["tests"], [list(row) for row in contract.TESTS])
            for platform in wheels:
                self.assertEqual(value["builds"][platform]["receipt_sha256"], contract.artifacts.sha256(receipts[platform]))
                self.assertIn("spiraltorch/__init__.py", value["wheels"][platform]["files"])
            with self.assertRaisesRegex(ValueError, "another platform"):
                preparation.freeze(root, dict(wheels, macos=wheels["linux"]), receipts)
            with patch.object(contract, "source_identity", side_effect=[source, {}]):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    preparation.freeze(root, wheels, receipts)
            receipts["linux"].write_bytes(b'{"source_commit":"wrong"}')
            with self.assertRaisesRegex(ValueError, "build receipt"):
                preparation.freeze(root, wheels, receipts)

    def test_frozen_matrix_and_types(self):
        value = fabricated_admission()
        self.assertIs(contract.validate_admission(value), value)
        mutations = [lambda v: v.update(candidate=3), lambda v: v.update(candidate=True),
                     lambda v: v.update(version="0.4.32"), lambda v: v["tests"].pop(),
                     lambda v: v["tests"].reverse(), lambda v: v.update(rounds=1),
                     lambda v: v.update(budget_seconds=5401), lambda v: v.update(release_ready=True),
                     lambda v: v["wheels"].pop("windows"), lambda v: v["builds"].pop("linux"),
                     lambda v: v["source"]["files"].update({"../escape": "a" * 64}),
                     lambda v: v["wheels"]["macos"].update(native=None),
                     lambda v: v["builds"]["linux"].update(wheel_sha256="f" * 64),
                     lambda v: v["builds"]["macos"].update(features=[None]),
                     lambda v: v["builds"]["windows"].update(source_commit="f" * 40),
                     lambda v: v.update(source=None), lambda v: v.update(long_request_sha256="f" * 64)]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                contract.validate_admission(changed)

    def test_strict_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    contract.read(path)
            path.unlink()
            contract.save(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                contract.save(path, {"a": 2})
            with self.assertRaises(ValueError):
                contract.read(path, limit=1)

    def test_source_is_committed_clean_and_uncached(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT)
            git("init", "-q")
            (root / "tools").mkdir()
            (root / "tools/test.py").write_text("pass\n")
            (root / "Cargo.lock").write_text("# synthetic\n")
            (root / ".gitignore").write_text("*.pyc\n")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic source")
            identity = contract.source_identity(root)
            self.assertEqual(set(identity["files"]), {"Cargo.lock", "tools/test.py"})
            (root / "untracked.py").write_text("pass\n")
            with self.assertRaisesRegex(ValueError, "clean committed"):
                contract.source_identity(root)
            (root / "untracked.py").unlink()
            (root / "tools/test.pyc").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "bytecode"):
                contract.source_identity(root)


class LedgerOffline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ledger = Path(self.temporary.name) / "ledger"
        self.admission = fabricated_admission()

    def admitted(self):
        contract.initialize_ledger(self.ledger)
        return contract.admit(self.ledger, self.admission)

    def repair(self):
        value = copy.deepcopy(self.admission)
        value.update(candidate=2, prior_admission_sha256=contract.digest(self.admission),
                     repair_review="https://github.com/RyoSpiralArchitect/SpiralTorch/pull/999999")
        value["source"]["tree"] = "f" * 40
        return value

    def test_execution_never_creates_or_resets_a_ledger(self):
        with self.assertRaises(FileNotFoundError):
            contract.admit(self.ledger, self.admission)
        directory = self.admitted()
        self.assertEqual(contract.admit(self.ledger, self.admission), directory)
        changed = copy.deepcopy(self.admission)
        changed["source"]["tree"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "different artifacts"):
            contract.admit(self.ledger, changed)
        with self.assertRaises(FileExistsError):
            contract.initialize_ledger(self.ledger)

    def test_local_claim_is_single_use_even_without_output(self):
        directory = self.admitted()
        output = contract.claim_environment(directory, "local-metal")
        self.assertFalse(output.exists())
        with self.assertRaises(FileExistsError):
            contract.claim_environment(directory, "local-metal")
        with self.assertRaises(FileNotFoundError):
            contract.claim_environment(directory, "hosted-metal")

    def test_hosted_requires_matching_three_round_local_success(self):
        directory = self.admitted()
        contract.claim_environment(directory, "local-metal")
        result = fabricated_result(self.admission)
        for mutate in (lambda r: r["rounds"].pop(), lambda r: r["rounds"].reverse(),
                       lambda r: r.update(admission_sha256="0" * 64),
                       lambda r: r["rounds"][1].update(returncode=1)):
            bad = copy.deepcopy(result)
            mutate(bad)
            with self.assertRaises(ValueError):
                contract.finish_environment(directory, "local-metal", bad)
        contract.finish_environment(directory, "local-metal", result)
        self.assertEqual(contract.claim_environment(directory, "hosted-metal"), directory / "hosted-metal")
        with self.assertRaises(FileExistsError):
            contract.claim_environment(directory, "hosted-metal")

    def test_failure_blocks_hosted_and_admits_only_one_changed_repair(self):
        directory = self.admitted()
        contract.claim_environment(directory, "local-metal")
        repair = self.repair()
        with self.assertRaisesRegex(ValueError, "unfinished"):
            contract.admit(self.ledger, repair)
        contract.finish_environment(directory, "local-metal", fabricated_result(self.admission, status="failed"))
        with self.assertRaisesRegex(ValueError, "completed matching local"):
            contract.claim_environment(directory, "hosted-metal")
        for changed, message in ((dict(repair, prior_admission_sha256="0" * 64), "wrong predecessor"),
                                 (dict(repair, source=self.admission["source"]), "unchanged retry")):
            with self.assertRaisesRegex(ValueError, message):
                contract.admit(self.ledger, changed)
        second = contract.admit(self.ledger, repair)
        self.assertEqual(second.name, "candidate-2")
        with self.assertRaisesRegex(ValueError, "superseded"):
            contract.claim_environment(directory, "local-metal")
        with self.assertRaises(ValueError):
            contract.admit(self.ledger, dict(repair, candidate=3))

    def test_success_is_not_permission_for_another_candidate(self):
        directory = self.admitted()
        contract.claim_environment(directory, "local-metal")
        contract.finish_environment(directory, "local-metal", fabricated_result(self.admission))
        with self.assertRaisesRegex(ValueError, "terminal failure"):
            contract.admit(self.ledger, self.repair())


if __name__ == "__main__":
    unittest.main()
