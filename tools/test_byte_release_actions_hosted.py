"""Offline candidate routing: no live services, secrets, native imports or GPU."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_download as download
import byte_release_hosted_v2 as h
import byte_release_transport_workflow_v2 as workflow
from test_byte_release_actions_input import records as input_records, zip_bytes, PAYLOAD
from test_byte_release_github_asset import TOKEN
from test_byte_release_hosted_v2 import fixture, Remote, RECIPIENT
from test_byte_release_transport_v2 import Offline, records, controller_fixture, commit, ROOT

t, c = h.t, h.c


def actions_records(amendment=None, descriptor=None):
    if amendment is None:
        amendment, descriptor = records()
    pin, run, artifact = input_records(zip_bytes())
    descriptor = dict(descriptor, schema=t.ACTIONS_DESCRIPTOR_SCHEMA, object=pin,
        ciphertext={key: pin["ciphertext"][key] for key in ("bytes", "sha256")})
    amendment = dict(amendment, descriptor_sha256=c.digest(descriptor))
    return amendment, descriptor, run, artifact


def actions_controller(root):
    source, controller, amendment, descriptor = controller_fixture(root)
    amendment, descriptor, _, _ = actions_records(amendment, descriptor)
    (controller / t.DESCRIPTOR).write_bytes(c.encoded(descriptor))
    for name in t.ACTIONS_RUNTIME_FILES:
        (controller / name).write_bytes((ROOT / name).read_bytes())
    (controller / ".github/workflows" / t.WORKFLOW).write_text(
        workflow.render((source / ".github/workflows" / t.WORKFLOW).read_text(), actions_artifact=True))
    head, tree = commit(controller)
    amendment.update(controller_commit=head, controller_tree=tree)
    return source, controller, amendment, descriptor


class CandidateActionsOffline(Offline):
    def test_candidate_identity_is_bound_to_the_complete_pin(self):
        amendment, descriptor, _, _ = actions_records()
        t.validate_amendment(amendment, c.digest(amendment), descriptor)
        for defect in ("ciphertext", "attempt", "schema", "foreign_repo"):
            changed = copy.deepcopy(descriptor)
            if defect == "ciphertext":
                changed["ciphertext"]["sha256"] = "0" * 64
            elif defect == "attempt":
                changed["object"]["bridge"]["run_attempt"] = 2
            elif defect == "schema":
                changed["schema"] = t.GITHUB_DESCRIPTOR_SCHEMA
            else:
                changed["object"]["repository"] = "other/repo"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                t.validate_descriptor(changed)

    def test_workflow_preserves_numerics_read_permissions_and_private_first_bytes(self):
        baseline = (ROOT / ".github/workflows" / t.WORKFLOW).read_text()
        rendered = workflow.render(baseline, actions_artifact=True)
        start = "      - name: Install the same wheel offline without cached Python code\n"
        end = "      - name: Encrypt the first outcome including incomplete evidence\n"
        self.assertEqual(baseline[baseline.index(start):baseline.index(end)],
                         rendered[rendered.index(start):rendered.index(end)])
        self.assertIn("permissions:\n  contents: read\n  actions: read\n", rendered)
        self.assertIn("timeout-minutes: 270", rendered)
        self.assertEqual(rendered.count('"$EVIDENCE_ROOT/input.zip.age"'), 2)
        self.assertLess(rendered.index("unset GH_TOKEN GITHUB_TOKEN"), rendered.index("prepare-input"))
        for forbidden in ("contents: write", "actions: write", "INPUT_OBJECT_URL", "_INPUT_URL"):
            self.assertNotIn(forbidden, rendered)
        for changes in (dict(actions_artifact=1), dict(actions_artifact=True, github_asset=True)):
            with self.assertRaises(ValueError):
                workflow.render(baseline, **changes)

    def test_controller_requires_all_dependencies_and_exact_readonly_envelope(self):
        for defect in (None, *t.ACTIONS_RUNTIME_FILES, "write", "wrong_route", "model_change"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                source, controller, amendment, descriptor = actions_controller(Path(temporary).resolve())
                path = controller / ".github/workflows" / t.WORKFLOW
                if defect in t.ACTIONS_RUNTIME_FILES:
                    (controller / defect).unlink()
                elif defect == "write":
                    path.write_text(path.read_text().replace("contents: read", "contents: write"))
                elif defect == "wrong_route":
                    path.write_text(workflow.render((source / ".github/workflows" / t.WORKFLOW).read_text()))
                elif defect == "model_change":
                    (controller / "Cargo.lock").write_text("not the frozen code\n")
                if defect:
                    head, tree = commit(controller)
                    amendment.update(controller_commit=head, controller_tree=tree)
                    with self.assertRaises(ValueError):
                        t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))
                else:
                    t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))

    def test_dispatch_checks_live_producer_before_claim_and_has_no_url_secret(self):
        for expired in (False, True):
            with self.subTest(expired=expired), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                ledger, admission, _, amendment, descriptor, controller = fixture(root)
                amendment, descriptor, run, artifact = actions_records(amendment, descriptor)
                artifact["expired"] = expired
                (controller / t.DESCRIPTOR).write_bytes(c.encoded(descriptor))
                remote = Remote(amendment)
                pin = descriptor["object"]
                calls = []

                def api(path, payload=None):
                    calls.append(path)
                    if path == f"actions/runs/{pin['bridge']['run_id']}":
                        return run
                    if path == f"actions/artifacts/{pin['artifact']['id']}":
                        return artifact
                    return remote(path, payload)

                with patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                        patch.object(c, "source_identity", return_value=admission["source"]), patch.object(h, "api", api):
                    if expired:
                        with self.assertRaises(ValueError):
                            h.dispatch(ledger, controller, "synthetic", amendment, c.digest(amendment), RECIPIENT, True)
                        self.assertFalse((ledger / "candidate-1/hosted-metal.dispatch.json").exists())
                        self.assertEqual(remote.posts, [])
                    else:
                        self.assertFalse(h.dispatch(ledger, controller, "synthetic", amendment,
                            c.digest(amendment), RECIPIENT)["dispatched"])
                        self.assertTrue(h.dispatch(ledger, controller, "synthetic", amendment,
                            c.digest(amendment), RECIPIENT, True)["dispatched"])
                        with self.assertRaises(ValueError):
                            h.dispatch(ledger, controller, "synthetic", amendment, c.digest(amendment), RECIPIENT, True)
                        self.assertEqual(len(remote.posts), 1)
                self.assertFalse(any("INPUT_URL" in path for path in calls))

    def test_worker_routes_to_readonly_archive_in_private_evidence_without_fallback(self):
        amendment, descriptor, _, _ = actions_records()
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary).resolve()
            output = evidence / "input.zip.age"
            env = dict(GH_TOKEN=TOKEN, EVIDENCE_ROOT=str(evidence))
            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(download, "download", return_value={"release_ready": False}) as fetch, \
                    patch.object(h, "download_ciphertext", side_effect=AssertionError("no fallback")):
                h.fetch_worker(output, env)
                fetch.assert_called_once_with(descriptor["object"], TOKEN, evidence / "actions-input.zip", output)
                for invalid in (dict(env, INPUT_OBJECT_URL="https://example.invalid"),
                                dict(env, EVIDENCE_ROOT=str(evidence / "other"))):
                    with self.assertRaises(ValueError):
                        h.fetch_worker(output, invalid)
                self.assertEqual(fetch.call_count, 1)

    def test_shared_deadline_retains_first_files_and_forbids_new_attempt(self):
        amendment, descriptor, _, _ = actions_records()
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary).resolve()
            output = evidence / "input.zip.age"
            env = dict(GH_TOKEN=TOKEN, EVIDENCE_ROOT=str(evidence))

            def worker(command, **kwargs):
                (evidence / "actions-input.zip").write_bytes(zip_bytes())
                output.write_bytes(PAYLOAD)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(h.owned_process, "run", side_effect=worker) as child:
                self.assertEqual(h.fetch_input(output, env)["status"], "verified_ciphertext")
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn(TOKEN, str(child.call_args))
                with self.assertRaises(FileExistsError):
                    h.fetch_input(output, env)
                self.assertEqual(child.call_count, 1)

    def test_return_packet_retains_partial_first_transport_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            evidence = root / "evidence"
            evidence.mkdir()
            payloads = {"actions-input.zip": b"partial archive", "input.zip.age": b"partial ciphertext",
                        "input.zip.age.fetch.log": b"safe error code"}
            for name, data in payloads.items():
                (evidence / name).write_bytes(data)
            archive = root / "result.zip"
            receipt = h.old.pack_result(evidence, root / "no-input", 1, archive, "a" * 64)
            unpacked = root / "recovered"
            h.handoff.unpack(archive, unpacked, receipt["archive_sha256"], "result", "a" * 64)
            for name, data in payloads.items():
                self.assertEqual((unpacked / name).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
