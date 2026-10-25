"""Hosted ownership tests with mocked GitHub; never submit a real workflow."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_hosted as hosted
from test_byte_release_contract import fabricated_admission
from test_byte_release_handoff import accepted_ledger

c = hosted.c
HEAD = "f" * 40
RECIPIENT = "age1" + "q" * 58  # Shape-only fake; age validates real checksums before execution.
SECRET = {"name": "ST_BYTE_RELEASE_CANDIDATE_1_INPUT_IDENTITY"}


def fake_run():
    return dict(id=123, run_attempt=1, event="workflow_dispatch", head_sha=HEAD,
                head_branch=hosted.branch(1), status="in_progress", conclusion=None)


class HostedOffline(unittest.TestCase):
    def test_verified_remote_return_is_bound_and_rechecked_before_import(self):
        for conclusion in ("success", "failure"):
            with self.subTest(conclusion=conclusion), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, _wheel, admission = accepted_ledger(root)
                directory = ledger / "candidate-1"
                c.save(directory / "hosted-metal.dispatch.json", dict(admission_sha256=c.digest(admission), controller_commit=HEAD))
                c.save(directory / "hosted-metal.run.json", dict(admission_sha256=c.digest(admission), controller_commit=HEAD, run_id=123))
                bundle, evidence = root / "bundle", root / "evidence"
                remote = bundle / "ledger/candidate-1"
                remote.mkdir(parents=True)
                evidence.mkdir()
                c.save(evidence / "hosted-context.json", dict(schema=hosted.SCHEMA, admitted=True, run_id=123,
                       run_attempt=1, controller_commit=HEAD, source_commit=admission["source"]["commit"],
                       admission_sha256=c.digest(admission), release_ready=False))
                c.save(remote / "admission.json", admission)
                shutil.copytree(directory / "local-metal", remote / "hosted-metal")
                result = c.read(remote / "hosted-metal/result.json")
                result["environment"] = "hosted-metal"
                (remote / "hosted-metal/result.json").write_bytes(c.encoded(result))
                c.save(remote / "hosted-metal.start.json", dict(environment="hosted-metal", admission_sha256=c.digest(admission)))
                c.finish_environment(remote, "hosted-metal", result)
                archive = root / "result.zip"
                packet = hosted.pack_result(evidence, bundle, 1, archive, c.digest(admission))
                with patch.object(hosted, "unique_run", return_value=dict(fake_run(), status="completed", conclusion=conclusion)):
                    imported = hosted.recover_result(directory, archive, packet["archive_sha256"])
                    self.assertEqual(imported["status"], "verified_rounds" if conclusion == "success" else "failed")
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        hosted.recover_result(directory, archive, packet["archive_sha256"])
                self.assertTrue((directory / "hosted-return-123/hosted-context.json").is_file())
                self.assertFalse(imported["release_ready"])
                if conclusion == "success":
                    hosted.handoff.verify_environment(directory, "hosted-metal")
                else:
                    derived = c.read(directory / "hosted-metal/result.json")
                    self.assertEqual(derived["origin"], "coordinator_observed_terminal_run")
                    self.assertEqual(derived["gpu_execution"], "not_inferred")

    def test_live_and_incomplete_returns_never_become_a_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, _receipt, admission = self.prepare(root)
            directory = ledger / "candidate-1"
            c.save(directory / "hosted-metal.dispatch.json", dict(admission_sha256=c.digest(admission), controller_commit=HEAD))
            c.save(directory / "hosted-metal.run.json", dict(admission_sha256=c.digest(admission), controller_commit=HEAD, run_id=123))
            evidence = root / "evidence"
            evidence.mkdir()
            c.save(evidence / "hosted-context.json", {"admitted": False, "synthetic": True})
            archive = root / "partial.zip"
            packet = hosted.pack_result(evidence, root / "missing-bundle", 1, archive, c.digest(admission))
            with patch.object(hosted, "unique_run", return_value=fake_run()), self.assertRaisesRegex(ValueError, "still live"):
                hosted.recover_result(directory, archive, packet["archive_sha256"])
            self.assertFalse((directory / "hosted-return-123").exists())
            with patch.object(hosted, "unique_run", return_value=dict(fake_run(), status="completed", conclusion="cancelled")):
                imported = hosted.recover_result(directory, archive, packet["archive_sha256"])
            self.assertEqual(imported["status"], "failed")
            self.assertEqual(c.read(directory / "hosted-return-123/hosted-context.json"), {"admitted": False, "synthetic": True})

    def prepare(self, root):
        admission = fabricated_admission()
        ledger = root / "ledger"
        c.initialize_ledger(ledger)
        c.admit(ledger, admission)
        receipt = root / "packet.json"
        c.save(receipt, dict(schema=hosted.handoff.SCHEMA, kind="input", admission_sha256=c.digest(admission),
                             archive_sha256="d" * 64, release_ready=False))
        return ledger, receipt, admission

    def test_default_dispatch_only_preflights_and_explicit_dispatch_is_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, receipt, admission = self.prepare(root)
            with patch.object(hosted.handoff, "verify_environment", return_value=admission), \
                    patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "e" * 64})), \
                    patch.object(hosted, "api", side_effect=[{"object": {"sha": HEAD}}, {"total_count": 0, "workflow_runs": []}, SECRET]) as api:
                result = hosted.dispatch(ledger, "unused", receipt, RECIPIENT)
                self.assertFalse(result["dispatched"])
                self.assertEqual(api.call_count, 3)
                self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())
            with patch.object(hosted.handoff, "verify_environment", return_value=admission), \
                    patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "e" * 64})), \
                    patch.object(hosted, "api", side_effect=[{"object": {"sha": HEAD}}, {"total_count": 0, "workflow_runs": []}, SECRET, None]) as api:
                result = hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                self.assertTrue(result["dispatched"])
                self.assertEqual(api.call_count, 4)
                self.assertEqual(api.call_args.args[1]["inputs"]["admission_sha256"], c.digest(admission))
                with self.assertRaisesRegex(ValueError, "already reserved"):
                    hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                self.assertEqual(api.call_count, 4)

    def test_ambiguous_post_retains_claim_and_never_resubmits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, receipt, admission = self.prepare(root)
            with patch.object(hosted.handoff, "verify_environment", return_value=admission), \
                    patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "e" * 64})), \
                    patch.object(hosted, "api", side_effect=[{"object": {"sha": HEAD}}, {"total_count": 0, "workflow_runs": []}, SECRET, TimeoutError()]) as api:
                with self.assertRaises(TimeoutError):
                    hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                with self.assertRaisesRegex(ValueError, "already reserved"):
                    hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                self.assertEqual(api.call_count, 4)
            response = c.read(ledger / "candidate-1/hosted-metal.dispatch-response.json")
            self.assertEqual(response["outcome"], "ambiguous_do_not_retry")

    def test_prior_run_cannot_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, receipt, admission = self.prepare(root)
            with patch.object(hosted.handoff, "verify_environment", return_value=admission), \
                    patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "e" * 64})), \
                    patch.object(hosted, "api", side_effect=[{"object": {"sha": HEAD}}, {"total_count": 1, "workflow_runs": [fake_run()]}]) as api:
                with self.assertRaisesRegex(ValueError, "already has a dispatch"):
                    hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                self.assertEqual(api.call_count, 2)
                self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())

    def test_wrong_remote_or_missing_secret_stops_before_claim_or_submission(self):
        cases = (
            ([{"object": {"sha": "0" * 40}}], "remote controller"),
            ([{"object": {"sha": HEAD}}, {"total_count": 0, "workflow_runs": []}, {}], "temporary input secret"),
        )
        for responses, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, receipt, admission = self.prepare(root)
                with patch.object(hosted.handoff, "verify_environment", return_value=admission), \
                        patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "e" * 64})), \
                        patch.object(hosted, "api", side_effect=responses) as api:
                    with self.assertRaisesRegex(ValueError, message):
                        hosted.dispatch(ledger, "unused", receipt, RECIPIENT, execute=True)
                    self.assertEqual(api.call_count, len(responses))
                    self.assertTrue(all(len(call.args) == 1 for call in api.call_args_list))
                    self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())

    def test_unique_run_rejects_duplicates_reruns_and_wrong_identity(self):
        for change in (dict(run_attempt=2), dict(head_sha="0" * 40), dict(event="push"), dict(head_branch="main")):
            with patch.object(hosted, "runs", return_value={"total_count": 1, "workflow_runs": [dict(fake_run(), **change)]}):
                with self.assertRaises(ValueError):
                    hosted.unique_run(1, HEAD)
        for count in (0, 2):
            with patch.object(hosted, "runs", return_value={"total_count": count, "workflow_runs": [fake_run()] * count}):
                with self.assertRaisesRegex(ValueError, "unique first"):
                    hosted.unique_run(1, HEAD)
        with patch.object(hosted, "runs", return_value={"total_count": 1, "workflow_runs": [fake_run()]}):
            self.assertEqual(hosted.unique_run(1, HEAD, 123)["id"], 123)
            with self.assertRaises(ValueError):
                hosted.unique_run(1, HEAD, 456)

    def test_binding_is_read_only_remote_and_idempotent_locally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, _receipt, admission = self.prepare(root)
            directory = ledger / "candidate-1"
            c.save(directory / "hosted-metal.dispatch.json", dict(candidate=1, admission_sha256=c.digest(admission), controller_commit=HEAD))
            with patch.object(hosted, "unique_run", return_value=fake_run()):
                first = hosted.bind_run(directory)
                self.assertEqual(first, hosted.bind_run(directory))
            with patch.object(hosted, "unique_run", return_value=dict(fake_run(), id=456)):
                with self.assertRaisesRegex(ValueError, "different run binding"):
                    hosted.bind_run(directory)

    def test_controller_can_only_add_one_ciphertext_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT).decode().strip()
            def commit():
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic")
            git("init", "-q")
            (root / "source.txt").write_text("synthetic")
            commit()
            source = git("rev-parse", "HEAD")
            (root / ".github").mkdir()
            (root / hosted.PACKET).write_bytes(b"synthetic ciphertext")
            commit()
            head, data = hosted.controller_identity(root, source)
            self.assertEqual(head, git("rev-parse", "HEAD"))
            self.assertTrue(c.is_hash(data["sha256"]))
            (root / "source.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "only one ciphertext"):
                hosted.controller_identity(root, source)

    def test_hosted_rejection_is_preserved_without_accessing_native_or_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "context.json"
            env = dict(CANDIDATE="1", SOURCE_COMMIT="a" * 40, GITHUB_SHA=HEAD,
                       GITHUB_REPOSITORY=hosted.REPO, GITHUB_RUN_ATTEMPT="2", GITHUB_REF="refs/heads/" + hosted.branch(1))
            with patch.object(hosted, "api") as api, self.assertRaises(ValueError):
                hosted.admit_run(path, env)
            api.assert_not_called()
            self.assertFalse(c.read(path)["admitted"])

    def test_hosted_first_run_admits_only_its_pinned_controller_and_ciphertext(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "context.json"
            env = dict(CANDIDATE="1", SOURCE_COMMIT="a" * 40, GITHUB_SHA=HEAD,
                       GITHUB_REPOSITORY=hosted.REPO, GITHUB_RUN_ATTEMPT="1",
                       GITHUB_REF="refs/heads/" + hosted.branch(1), GITHUB_RUN_ID="123",
                       ADMISSION_SHA256="b" * 64, INPUT_PLAINTEXT_SHA256="c" * 64,
                       INPUT_CIPHERTEXT_SHA256="d" * 64, ImageOS="synthetic-os", ImageVersion="synthetic-image")
            with patch.object(hosted, "controller_identity", return_value=(HEAD, {"sha256": "d" * 64})) as controller, \
                    patch.object(hosted, "unique_run", return_value=fake_run()) as run:
                record = hosted.admit_run(path, env)
                self.assertTrue(record["admitted"])
                self.assertEqual(c.read(path), record)
                self.assertFalse(record["release_ready"])
                controller.assert_called_once_with(hosted.ROOT, "a" * 40)
                run.assert_called_once_with(1, HEAD, 123)
            for index, identity in enumerate((("0" * 40, {"sha256": "d" * 64}), (HEAD, {"sha256": "0" * 64}))):
                rejected = path.with_name(f"rejected-{index}.json")
                with patch.object(hosted, "controller_identity", return_value=identity), \
                        patch.object(hosted, "unique_run") as run, self.assertRaisesRegex(ValueError, "source or ciphertext"):
                    hosted.admit_run(rejected, env)
                run.assert_not_called()
                self.assertFalse(c.read(rejected)["admitted"])

    def test_transported_input_must_match_the_execution_source_and_same_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, wheel, admission = accepted_ledger(root)
            archive = root / "input.zip"
            receipt = hosted.handoff.pack_input(ledger, wheel, archive)
            output = root / "decoded"
            hosted.handoff.unpack(archive, output, receipt["archive_sha256"], "input", c.digest(admission))
            with patch.object(c, "source_identity", return_value=admission["source"]):
                verified = hosted.verify_input(output, 1, c.digest(admission))
                self.assertEqual(verified["admission_sha256"], c.digest(admission))
                self.assertEqual(Path(verified["wheel"]).read_bytes(), wheel.read_bytes())
                with self.assertRaisesRegex(ValueError, "execution source"):
                    hosted.verify_input(output, 1, "0" * 64)
            with patch.object(c, "source_identity", return_value=dict(admission["source"], commit="0" * 40)), \
                    self.assertRaisesRegex(ValueError, "execution source"):
                hosted.verify_input(output, 1, c.digest(admission))
            self.assertFalse((output / "ledger/candidate-1/hosted-metal.start.json").exists())

    def test_workflow_is_manual_ciphertext_only_and_uses_the_same_frozen_source(self):
        source = (hosted.ROOT / ".github/workflows" / hosted.WORKFLOW).read_text()
        self.assertIn("workflow_dispatch:", source)
        self.assertNotIn("pull_request:", source)
        self.assertNotIn("schedule:", source)
        self.assertIn("timeout-minutes: 120", source)
        self.assertIn("cancel-in-progress: false", source)
        self.assertEqual(source.count("--environment hosted-metal --execute"), 1)
        self.assertIn("--no-index --no-deps --no-cache-dir --no-compile", source)
        self.assertLess(source.index("recipient-probe.age"), source.index("tools/byte_release_hosted.py admit-run"))
        self.assertLess(source.index("tools/byte_release_hosted.py admit-run"), source.index("INPUT_IDENTITY:"))
        self.assertLess(source.index("unset INPUT_IDENTITY"), source.index("run_byte_release_qualification.py"))
        upload = source.split("name: Upload only ciphertext and its digest", 1)[1]
        self.assertIn("public/result.zip.age", upload)
        self.assertNotIn("/evidence/", upload)
        self.assertNotIn("/input/", upload)
        self.assertNotIn("cargo", source)
        self.assertNotIn("maturin", source)


if __name__ == "__main__":
    unittest.main()
