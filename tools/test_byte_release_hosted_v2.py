"""Mocked hosted transport: no network, GPU, real keys or production-ledger writes."""
import copy
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_hosted_v2 as h
from test_byte_release_handoff import accepted_ledger
from test_byte_release_transport_v2 import Offline, records, controller_fixture, commit, RUNTIME_FILES

t, c = h.t, h.c
RECIPIENT = "age1" + "q" * 58
WORKFLOW_ID = 987
RUN_ID = 123


class Remote:
    def __init__(self, amendment):
        self.amendment = amendment
        self.rows = []
        self.posts = []
        self.calls = []
        self.error = None
        self.path = ".github/workflows/" + t.WORKFLOW
        self.missing_secret = False

    def run(self, **changes):
        return dict(dict(id=RUN_ID, workflow_id=WORKFLOW_ID, run_attempt=1, event="workflow_dispatch",
                         head_sha=self.amendment["controller_commit"], head_branch=t.branch(1),
                         status="in_progress", conclusion=None), **changes)

    def __call__(self, path, payload=None):
        self.calls.append((path, payload))
        if path == "git/ref/heads/" + t.branch(1):
            return {"object": {"sha": self.amendment["controller_commit"]}}
        if path == "actions/workflows/" + t.WORKFLOW:
            return {"id": WORKFLOW_ID, "path": self.path}
        if path == f"actions/workflows/{WORKFLOW_ID}/runs?event=workflow_dispatch&branch={t.branch(1)}&per_page=2":
            return {"total_count": len(self.rows), "workflow_runs": copy.deepcopy(self.rows)}
        if path.startswith("actions/secrets/"):
            return {} if self.missing_secret else {"name": path.rsplit("/", 1)[1]}
        if path == f"actions/workflows/{WORKFLOW_ID}/dispatches" and payload is not None:
            self.posts.append(payload)
            if self.error:
                raise self.error
            return None
        raise AssertionError("unexpected API route: " + path)


def fixture(root):
    ledger, wheel, admission = accepted_ledger(root)
    archive = root / "input.zip"
    t.v1.pack_input(ledger, wheel, archive)
    local = c.read(ledger / "candidate-1/local-metal.result.json")
    amendment, descriptor = records(admission, archive, local["evidence_sha256"])
    controller = root / "controller"
    (controller / ".github").mkdir(parents=True)
    c.save(controller / t.DESCRIPTOR, descriptor)
    return ledger, admission, archive, amendment, descriptor, controller


def environment(root, amendment, descriptor):
    return dict(CANDIDATE="1", SOURCE_COMMIT=amendment["source_commit"],
                ADMISSION_SHA256=amendment["admission_sha256"],
                RESULT_RECIPIENT=amendment["result_recipient"],
                INPUT_PLAINTEXT_SHA256=descriptor["plaintext"]["sha256"],
                INPUT_CIPHERTEXT_SHA256=descriptor["ciphertext"]["sha256"],
                TRANSPORT_AMENDMENT=c.encoded(amendment).decode(),
                TRANSPORT_AMENDMENT_SHA256=c.digest(amendment),
                GITHUB_REPOSITORY=t.REPO, GITHUB_RUN_ATTEMPT="1",
                GITHUB_REF="refs/heads/" + t.branch(1), GITHUB_SHA=amendment["controller_commit"],
                GITHUB_RUN_ID=str(RUN_ID), EVIDENCE_ROOT=str(root / "evidence"))


def context(amendment, descriptor):
    return dict(schema=h.SCHEMA, admitted=True, release_ready=False, candidate=1,
                run_id=RUN_ID, run_attempt=1, workflow_id=WORKFLOW_ID,
                controller_commit=amendment["controller_commit"], source_commit=amendment["source_commit"],
                admission_sha256=amendment["admission_sha256"], amendment_sha256=c.digest(amendment),
                descriptor_sha256=c.digest(descriptor))


class HostedOffline(Offline):
    def setUp(self):
        super().setUp()
        self.forbid_api = patch.object(h, "api", side_effect=AssertionError("real API forbidden"))
        self.forbid_api.start()
        self.addCleanup(self.forbid_api.stop)

    def submit(self, ledger, admission, amendment, controller, remote, execute=True):
        with patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                patch.object(c, "source_identity", return_value=admission["source"]), patch.object(h, "api", remote):
            return h.dispatch(ledger, controller, "synthetic-source", amendment, c.digest(amendment), RECIPIENT, execute)

    def test_missing_runtime_cannot_reach_ledger_or_api_checks(self):
        for name in RUNTIME_FILES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source, controller, amendment, _descriptor = controller_fixture(root)
                (controller / name).unlink()
                head, tree = commit(controller)
                amendment.update(controller_commit=head, controller_tree=tree)
                with patch.object(h.handoff, "verify_environment", side_effect=AssertionError("ledger reached")) as ledger, \
                        patch.object(h, "api", side_effect=AssertionError("API reached")) as api:
                    with self.assertRaisesRegex(ValueError, "runtime modules"):
                        h.dispatch(root / "unused-ledger", controller, source, amendment,
                                   c.digest(amendment), RECIPIENT, execute=True)
                    ledger.assert_not_called()
                    api.assert_not_called()

    def test_default_is_read_only_and_explicit_post_is_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
            remote = Remote(amendment)
            preview = self.submit(ledger, admission, amendment, controller, remote, False)
            self.assertFalse(preview["dispatched"])
            self.assertEqual(remote.posts, [])
            self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())
            self.assertTrue(self.submit(ledger, admission, amendment, controller, remote)["dispatched"])
            self.assertEqual(len(remote.posts), 1)
            claim = c.read(ledger / "candidate-1/hosted-metal.dispatch.json")
            self.assertEqual(claim["amendment_sha256"], c.digest(amendment))
            with self.assertRaisesRegex(ValueError, "reserved"):
                self.submit(ledger, admission, amendment, controller, remote)
            self.assertEqual(len(remote.posts), 1)

    def test_timeout_keeps_claim_and_cannot_submit_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
            remote = Remote(amendment)
            remote.error = TimeoutError("synthetic")
            with self.assertRaises(TimeoutError):
                self.submit(ledger, admission, amendment, controller, remote)
            with self.assertRaisesRegex(ValueError, "reserved"):
                self.submit(ledger, admission, amendment, controller, remote)
            self.assertEqual(len(remote.posts), 1)
            response = c.read(ledger / "candidate-1/hosted-metal.dispatch-response.json")
            self.assertEqual(response["outcome"], "ambiguous_do_not_retry")

    def test_every_old_or_new_claim_blocks_before_api(self):
        for suffix in ("dispatch", "dispatch-response", "start", "run", "result", "import"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
                c.save(ledger / f"candidate-1/hosted-metal.{suffix}.json",
                       {"schema": "spiraltorch.byte_release.hosted.v1", "synthetic": True})
                remote = Remote(amendment)
                with self.assertRaisesRegex(ValueError, "reserved"):
                    self.submit(ledger, admission, amendment, controller, remote)
                self.assertEqual(remote.calls, [])

    def test_prior_remote_run_renamed_workflow_and_missing_secrets_reject(self):
        for defect in ("prior_v1", "renamed_workflow", "missing_secret"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
                remote = Remote(amendment)
                if defect == "prior_v1":
                    remote.rows = [remote.run()]
                elif defect == "renamed_workflow":
                    remote.path = ".github/workflows/byte_release_qualification_v2.yml"
                else:
                    remote.missing_secret = True
                with self.assertRaises(ValueError):
                    self.submit(ledger, admission, amendment, controller, remote)
                self.assertEqual(remote.posts, [])
                self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())

    def test_changed_local_proof_and_superseded_candidate_cannot_dispatch(self):
        for defect in ("evidence", "superseded"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
                if defect == "evidence":
                    path = next((ledger / "candidate-1/local-metal").rglob("worker-result.json"))
                    path.write_bytes(path.read_bytes() + b" ")
                else:
                    (ledger / "candidate-2").mkdir()
                remote = Remote(amendment)
                with self.assertRaises(ValueError):
                    self.submit(ledger, admission, amendment, controller, remote)
                self.assertEqual(remote.calls, [])

    def test_unique_run_checks_namespace_types_and_full_identity(self):
        amendment, _descriptor = records()
        remote = Remote(amendment)
        mutations = ({"workflow_id": WORKFLOW_ID + 1}, {"run_attempt": 2}, {"run_attempt": True},
                     {"head_sha": "0" * 40}, {"head_branch": "fresh"}, {"event": "push"}, {"id": True})
        with patch.object(h, "api", remote):
            for change in mutations:
                remote.rows = [remote.run(**change)]
                with self.subTest(change=change), self.assertRaises(ValueError):
                    h.unique_run(1, amendment["controller_commit"], RUN_ID)
            for count in (0, 2):
                remote.rows = [remote.run()] * count
                with self.assertRaises(ValueError):
                    h.unique_run(1, amendment["controller_commit"])

    def test_malformed_remote_metadata_and_listing_fail_closed(self):
        amendment, _descriptor = records()
        remote = Remote(amendment)
        metadata = {"id": WORKFLOW_ID, "path": remote.path}
        for value in (None, [], dict(metadata, id=True), dict(metadata, id=0)):
            with self.subTest(metadata=value), patch.object(h, "api", return_value=value), self.assertRaises(ValueError):
                h.workflow_id()
        for value in (None, [], {"total_count": True, "workflow_runs": []},
                      {"total_count": -1, "workflow_runs": []}, {"total_count": 0, "workflow_runs": None}):
            with self.subTest(listing=value), patch.object(h, "api", side_effect=[metadata, value]), self.assertRaises(ValueError):
                h.runs(1)

    def test_partial_directory_or_broken_claim_link_cannot_reset_ownership(self):
        for kind in ("partial_directory", "broken_link"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                if kind == "partial_directory":
                    (root / "hosted-metal").mkdir()
                else:
                    (root / "hosted-metal.dispatch.json").symlink_to(root / "absent")
                with self.assertRaisesRegex(ValueError, "reserved"):
                    h.no_hosted_claim(root)

    def test_binding_only_observes_original_run_and_keeps_first_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _archive, amendment, _descriptor, controller = fixture(root)
            remote = Remote(amendment)
            self.submit(ledger, admission, amendment, controller, remote)
            remote.rows = [remote.run()]
            with patch.object(h, "api", remote):
                first = h.bind_run(ledger / "candidate-1")
                self.assertEqual(h.bind_run(ledger / "candidate-1"), first)
                remote.rows = [remote.run(id=RUN_ID + 1)]
                with self.assertRaisesRegex(ValueError, "different first"):
                    h.bind_run(ledger / "candidate-1")
            self.assertEqual(len(remote.posts), 1)

    def test_admission_binds_records_before_any_input_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _ledger, _admission, _archive, amendment, descriptor, controller = fixture(root)
            (root / "evidence").mkdir()
            env = environment(root, amendment, descriptor)
            remote = Remote(amendment)
            remote.rows = [remote.run()]
            with patch.object(h, "ROOT", controller), patch.object(h, "api", remote), \
                    patch.object(t, "validate_controller", return_value=amendment["controller_commit"]):
                record = h.admit_run("synthetic-source", root / "evidence/hosted-context.json", env)
            self.assertTrue(record["admitted"])
            self.assertEqual(c.read(root / "evidence/transport-amendment.json"), amendment)
            self.assertFalse(any("secrets" in path for path, _payload in remote.calls))

    def test_rejected_invocation_is_retained_without_native_or_remote_call(self):
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main"),
                           ("GITHUB_SHA", "0" * 40), ("INPUT_PLAINTEXT_SHA256", "0" * 64)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                _ledger, _admission, _archive, amendment, descriptor, controller = fixture(root)
                (root / "evidence").mkdir()
                env = dict(environment(root, amendment, descriptor), **{key: value})
                with patch.object(h, "ROOT", controller), self.assertRaises(ValueError):
                    h.admit_run("synthetic-source", root / "evidence/hosted-context.json", env)
                self.assertFalse(c.read(root / "evidence/hosted-context.json")["admitted"])

    def test_preparation_uses_original_verifier_without_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _ledger, admission, archive, amendment, descriptor, controller = fixture(root)
            (root / "evidence").mkdir()
            c.save(root / "evidence/hosted-context.json", context(amendment, descriptor))
            env = environment(root, amendment, descriptor)
            output = root / "decoded"
            expected = dict(admission_sha256=c.digest(admission), release_ready=False,
                            wheel=str(output / "wheel" / admission["wheels"]["macos"]["filename"]))

            def verifier(command, **kwargs):
                kwargs["stdout"].write(c.encoded(expected))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(h, "ROOT", controller), \
                    patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                    patch.object(c, "source_identity", return_value=admission["source"]), \
                    patch.object(h.owned_process, "run", side_effect=verifier) as child:
                for key in ("INPUT_OBJECT_URL", "INPUT_IDENTITY", "GH_TOKEN", "GITHUB_TOKEN"):
                    with self.assertRaisesRegex(ValueError, "credentials"):
                        h.prepare_input("original-source", archive, output, dict(env, **{key: "test-only"}))
                child.assert_not_called()
                self.assertEqual(h.prepare_input("original-source", archive, output, env), expected)
                self.assertEqual(child.call_args.args[0][:5],
                                 [sys.executable, "-I", "-S", "-B", "tools/byte_release_hosted.py"])
                self.assertEqual(child.call_args.kwargs["cwd"], "original-source")
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertEqual(c.read(root / "evidence/frozen-input-verifier-process.json")["returncode"], 0)
            self.assertFalse((output / "ledger/candidate-1/hosted-metal.start.json").exists())

    def test_frozen_verifier_failure_keeps_logs_and_does_not_launch_learning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, archive, amendment, descriptor, controller = fixture(root)
            (root / "evidence").mkdir()
            c.save(root / "evidence/hosted-context.json", context(amendment, descriptor))
            env = environment(root, amendment, descriptor)

            def failure(command, **kwargs):
                kwargs["stdout"].write(b"first synthetic verifier output\n")
                kwargs["stderr"].write(b"first synthetic verifier failure\n")
                return subprocess.CompletedProcess(command, 1)

            with patch.object(h, "ROOT", controller), \
                    patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                    patch.object(c, "source_identity", return_value=admission["source"]), \
                    patch.object(h.owned_process, "run", side_effect=failure) as child:
                with self.assertRaisesRegex(ValueError, "frozen input verifier failed"):
                    h.prepare_input("original-source", archive, root / "decoded", env)
                self.assertEqual(child.call_count, 1)
            self.assertEqual((root / "evidence/frozen-input-verifier.stderr").read_bytes(),
                             b"first synthetic verifier failure\n")
            self.assertEqual(c.read(root / "evidence/frozen-input-verifier-process.json")["returncode"], 1)
            self.assertFalse((ledger / "candidate-1/hosted-metal.start.json").exists())
            self.assertFalse((root / "decoded/ledger/candidate-1/hosted-metal.start.json").exists())

    def test_successful_verifier_exit_does_not_accept_a_different_receipt(self):
        for defect in ("wheel", "admission_sha256", "release_ready", "extra"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, archive, amendment, descriptor, controller = fixture(root)
                (root / "evidence").mkdir()
                c.save(root / "evidence/hosted-context.json", context(amendment, descriptor))
                env = environment(root, amendment, descriptor)
                output = root / "decoded"
                receipt = dict(admission_sha256=c.digest(admission), release_ready=False,
                               wheel=str(output / "wheel" / admission["wheels"]["macos"]["filename"]))
                receipt[defect] = True if defect == "release_ready" else "synthetic-different-value"

                def verifier(command, **kwargs):
                    kwargs["stdout"].write(c.encoded(receipt))
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(h, "ROOT", controller), \
                        patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                        patch.object(c, "source_identity", return_value=admission["source"]), \
                        patch.object(h.owned_process, "run", side_effect=verifier) as child:
                    with self.assertRaisesRegex(ValueError, "receipt differs"):
                        h.prepare_input("original-source", archive, output, env)
                    self.assertEqual(child.call_count, 1)
                self.assertEqual(c.read(root / "evidence/frozen-input-verifier.stdout"), receipt)
                self.assertFalse((ledger / "candidate-1/hosted-metal.start.json").exists())

    def test_context_and_return_recipient_drift_reject_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _archive, amendment, descriptor, controller = fixture(root)
            (root / "evidence").mkdir()
            path = root / "evidence/hosted-context.json"
            original = context(amendment, descriptor)
            c.save(path, original)
            env = environment(root, amendment, descriptor)
            with patch.object(h, "ROOT", controller):
                for key, value in (("run_attempt", True), ("release_ready", True),
                                   ("candidate", 2), ("source_commit", "0" * 40),
                                   ("admission_sha256", "0" * 64), ("workflow_id", True)):
                    path.write_bytes(c.encoded(dict(original, **{key: value})))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        h.admitted_context(env)
                path.write_bytes(c.encoded(original))
                with self.assertRaises(ValueError):
                    h.environment_records(dict(env, RESULT_RECIPIENT="age1" + "p" * 58))
            remote = Remote(amendment)
            with patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                    patch.object(c, "source_identity", return_value=admission["source"]), patch.object(h, "api", remote):
                with self.assertRaisesRegex(ValueError, "recipient differs"):
                    h.dispatch(ledger, controller, "source", amendment, c.digest(amendment), "age1" + "p" * 58)
            self.assertEqual(remote.calls, [])

    def return_packet(self, root, ledger, admission, amendment, descriptor, defect=None):
        evidence, bundle = root / "evidence", root / "bundle"
        evidence.mkdir()
        record = context(amendment, descriptor)
        if defect == "context":
            record["amendment_sha256"] = "0" * 64
        if defect == "incomplete":
            record["admitted"] = False
        c.save(evidence / "hosted-context.json", record)
        c.save(evidence / "transport-amendment.json", amendment)
        c.save(evidence / "input-descriptor.json", descriptor)
        if defect != "incomplete":
            remote = bundle / "ledger/candidate-1"
            remote.mkdir(parents=True)
            c.save(remote / "admission.json", admission)
            shutil.copytree(ledger / "candidate-1/local-metal", remote / "hosted-metal")
            path = remote / "hosted-metal/result.json"
            result = c.read(path)
            result.update(environment="hosted-metal", budget_seconds=14400)
            path.write_bytes(c.encoded(result))
            c.save(remote / "hosted-metal.start.json",
                   dict(environment="hosted-metal", admission_sha256=c.digest(admission)))
            c.finish_environment(remote, "hosted-metal", result)
        archive = root / "return.zip"
        receipt = h.old.pack_result(evidence, bundle, 1, archive, c.digest(admission))
        return archive, receipt

    def test_return_is_bound_preserves_first_outcome_and_never_turns_failure_green(self):
        for defect in (None, "context", "incomplete", "failed_job"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _input, amendment, descriptor, controller = fixture(root)
                remote = Remote(amendment)
                self.submit(ledger, admission, amendment, controller, remote)
                remote.rows = [remote.run(status="completed", conclusion="failure" if defect == "failed_job" else "success")]
                with patch.object(h, "api", remote):
                    h.bind_run(ledger / "candidate-1")
                    archive, receipt = self.return_packet(root, ledger, admission, amendment, descriptor, defect)
                    result = h.recover_result(ledger / "candidate-1", archive, receipt["archive_sha256"])
                    self.assertEqual(result["status"], "verified_rounds" if defect is None else "failed")
                    self.assertFalse(result["release_ready"])
                    self.assertTrue((ledger / f"candidate-1/hosted-return-{RUN_ID}/hosted-context.json").is_file())
                    if defect is None:
                        t.v1.verify_environment(ledger / "candidate-1", "hosted-metal")
                    else:
                        failure = c.read(ledger / "candidate-1/hosted-metal/result.json")
                        self.assertEqual(failure["gpu_execution"], "not_inferred")
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        h.recover_result(ledger / "candidate-1", archive, receipt["archive_sha256"])

    def test_nonterminal_and_wrong_outer_hash_cannot_create_accepted_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _input, amendment, descriptor, controller = fixture(root)
            remote = Remote(amendment)
            self.submit(ledger, admission, amendment, controller, remote)
            remote.rows = [remote.run()]
            with patch.object(h, "api", remote):
                h.bind_run(ledger / "candidate-1")
                archive, receipt = self.return_packet(root, ledger, admission, amendment, descriptor)
                with self.assertRaisesRegex(ValueError, "still live"):
                    h.recover_result(ledger / "candidate-1", archive, receipt["archive_sha256"])
                remote.rows = [remote.run(status="completed", conclusion="success")]
                with self.assertRaises(ValueError):
                    h.recover_result(ledger / "candidate-1", archive, "0" * 64)
            self.assertFalse((ledger / "candidate-1/hosted-metal.result.json").exists())
            self.assertTrue((ledger / "candidate-1/hosted-metal.dispatch.json").exists())

    def test_returned_records_must_match_canonical_pins_not_python_equality(self):
        for defect in ("amendment_candidate_bool", "descriptor_size_float", "admission_candidate_bool"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _input, amendment, descriptor, controller = fixture(root)
                remote = Remote(amendment)
                self.submit(ledger, admission, amendment, controller, remote)
                remote.rows = [remote.run(status="completed", conclusion="success")]
                with patch.object(h, "api", remote):
                    h.bind_run(ledger / "candidate-1")
                    self.return_packet(root, ledger, admission, amendment, descriptor)
                    if defect == "amendment_candidate_bool":
                        path, changed = root / "evidence/transport-amendment.json", copy.deepcopy(amendment)
                        changed["candidate"] = True
                    elif defect == "descriptor_size_float":
                        path, changed = root / "evidence/input-descriptor.json", copy.deepcopy(descriptor)
                        changed["plaintext"]["bytes"] = float(changed["plaintext"]["bytes"])
                    else:
                        path, changed = root / "bundle/ledger/candidate-1/admission.json", copy.deepcopy(admission)
                        changed["candidate"] = True
                    original = c.read(path)
                    self.assertEqual(changed, original)
                    self.assertNotEqual(c.digest(changed), c.digest(original))
                    path.write_bytes(c.encoded(changed))
                    archive = root / "changed-return.zip"
                    receipt = h.old.pack_result(root / "evidence", root / "bundle", 1, archive, c.digest(admission))
                    result = h.recover_result(ledger / "candidate-1", archive, receipt["archive_sha256"])
                    self.assertEqual(result["status"], "failed")
                    self.assertFalse(result["release_ready"])
                    raw = ledger / f"candidate-1/hosted-return-{RUN_ID}"
                    payload_base = root / ("bundle" if defect == "admission_candidate_bool" else "evidence")
                    raw_record = raw / path.relative_to(payload_base)
                    self.assertEqual(c.digest(c.read(raw_record)), c.digest(changed))


class FetchOffline(Offline):
    def test_owned_transfer_success_verifies_bytes_and_cannot_overwrite(self):
        amendment, descriptor = records()
        env = dict(INPUT_OBJECT_URL="https://storage.example.test/objects/synthetic.age?secret=test-only")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "cipher.age"

            def worker(command, **kwargs):
                with output.open("xb") as destination:
                    destination.write(b"testonly")
                kwargs["stdout"].write(b"synthetic worker finished\n")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(h.owned_process, "run", side_effect=worker) as child:
                self.assertEqual(h.fetch_input(output, env), dict(status="verified_ciphertext", release_ready=False))
                with self.assertRaises(FileExistsError):
                    h.fetch_input(output, env)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn(env["INPUT_OBJECT_URL"], child.call_args.args[0])
            self.assertNotIn("test-only", (root / "cipher.age.fetch.log").read_text())

    def test_direct_response_is_streamed_and_secrets_do_not_enter_errors(self):
        _amendment, descriptor = records()
        url = "https://storage.example.test/objects/synthetic.age?secret=never-log-this"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            response = io.BytesIO(b"testonly")
            response.status, response.headers = 200, {"Content-Length": "8"}
            response.geturl = lambda: url
            opener = Mock()
            opener.open.return_value = response
            self.assertEqual(h.download_ciphertext(url, descriptor, root / "cipher.age", opener), descriptor["ciphertext"])
            opener.open.assert_called_once_with(url, timeout=30)
            opener.open.side_effect = RuntimeError(url)
            with self.assertRaises(ValueError) as caught:
                h.download_ciphertext(url, descriptor, root / "bad.age", opener)
            self.assertNotIn("never-log-this", str(caught.exception))

    def test_redirect_status_length_and_body_changes_reject_without_retry(self):
        _amendment, descriptor = records()
        url = "https://storage.example.test/objects/synthetic.age?secret=test-only"
        for defect in ("redirect", "status", "length", "body"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                response = io.BytesIO(b"changed!" if defect == "body" else b"testonly")
                response.status = 302 if defect == "status" else 200
                response.headers = {"Content-Length": "9" if defect == "length" else "8"}
                response.geturl = lambda: "https://other.example.test/input.age" if defect == "redirect" else url
                opener = Mock()
                opener.open.return_value = response
                with self.assertRaises(ValueError):
                    h.download_ciphertext(url, descriptor, root / "cipher.age", opener)
                self.assertEqual(opener.open.call_count, 1)
        with self.assertRaises(ValueError):
            h.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example.test")

    def test_owned_transfer_has_one_fixed_deadline_and_never_restarts(self):
        amendment, descriptor = records()
        env = dict(INPUT_OBJECT_URL="https://storage.example.test/objects/synthetic.age?secret=test-only")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "cipher.age"
            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(h.owned_process, "run", side_effect=subprocess.TimeoutExpired("synthetic", 300)) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    h.fetch_input(output, env)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs["timeout"], 300)
                self.assertNotIn(env["INPUT_OBJECT_URL"], run.call_args.args[0])
            self.assertTrue((root / "cipher.age.fetch.log").exists())


if __name__ == "__main__":
    unittest.main()
