"""CPU-only bridge checks with synthetic metadata and no real credentials."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_input_bridge as b
from test_byte_release_github_asset import github_records, TOKEN
from test_byte_release_transport_v2 import Offline


def fixture():
    _amendment, descriptor = github_records()
    spec = dict(schema=b.SCHEMA, branch=b.a.BRIDGE_PREFIX + "synthetic",
                workflow_id=303, artifact_name="opaque-input", input=descriptor, release_ready=False)
    env = dict(GITHUB_REPOSITORY=b.t.REPO, GITHUB_REF="refs/heads/" + spec["branch"],
               GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="202",
               GITHUB_SHA="a" * 40, EXPECTED_BRIDGE_HEAD="a" * 40,
               EXPECTED_BRIDGE_SHA256=b.c.digest(spec), GH_TOKEN=TOKEN)
    context = dict(schema=b.SCHEMA, head="a" * 40, tree="b" * 40, spec_sha256=b.c.digest(spec),
                   run_id=202, run_attempt=1, branch=spec["branch"], workflow_id=303,
                   fetch_seconds=300, release_ready=False)
    transfer = dict(status="verified_ciphertext", **descriptor["ciphertext"],
                    **{key: descriptor["object"][key] for key in ("repository_id", "release_id", "asset_id")},
                    path="direct_api", release_ready=False)
    run = dict(id=202, run_attempt=1, head_sha=context["head"], head_branch=spec["branch"],
               workflow_id=303, path=".github/workflows/" + b.t.WORKFLOW, event="workflow_dispatch",
               status="in_progress", conclusion=None,
               repository=dict(id=101, full_name=b.t.REPO), head_repository=dict(id=101, full_name=b.t.REPO))
    return spec, env, context, transfer, run


class BridgeOffline(Offline):
    def test_spec_separates_bridge_from_candidate_and_limits_input(self):
        spec, *_ = fixture()
        self.assertEqual(b.validate_spec(spec), spec)
        for key, value in (("schema", "other"), ("release_ready", 0), ("branch", b.t.branch(1)),
                           ("branch", b.a.BRIDGE_PREFIX + "bad;id"), ("workflow_id", True),
                           ("artifact_name", "../escape"), ("url", "https://example.invalid")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_spec(dict(spec, **{key: value}))
        bad = copy.deepcopy(spec)
        bad["input"]["ciphertext"]["bytes"] = b.t.MAX_CIPHER + 1
        with self.assertRaises(ValueError):
            b.validate_spec(bad)

    def test_context_binds_clean_head_event_attempt_descriptor_and_no_key(self):
        spec, env, context, *_ = fixture()
        with patch.object(b.t, "read_record", return_value=spec), \
                patch.object(b.c, "source_identity", return_value=dict(commit="a" * 40, tree="b" * 40)):
            self.assertEqual(b.checked_context(env), (context, spec))
            for key, value in (("GITHUB_REPOSITORY", "fork/repo"), ("GITHUB_SHA", "c" * 40),
                               ("EXPECTED_BRIDGE_HEAD", "no"), ("GITHUB_REF", "refs/heads/main"),
                               ("GITHUB_EVENT_NAME", "push"), ("GITHUB_RUN_ATTEMPT", "2"),
                               ("GITHUB_RUN_ID", "-1"), ("EXPECTED_BRIDGE_SHA256", "0" * 64),
                               ("GH_TOKEN", ""), ("GITHUB_TOKEN", TOKEN),
                               ("INPUT_IDENTITY", "synthetic-key"), ("INPUT_OBJECT_URL", "synthetic-url")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    b.checked_context(dict(env, **{key: value}))
            with patch.object(b.c, "source_identity", return_value=dict(commit="c" * 40, tree="b" * 40)), \
                    self.assertRaises(ValueError):
                b.checked_context(env)

    def test_unique_run_is_get_only_and_rejects_repeated_foreign_or_completed_runs(self):
        spec, _, context, _, run = fixture()
        opener = object()
        with patch.object(b.g, "_metadata", return_value=dict(total_count=1, workflow_runs=[run])) as metadata:
            b.unique_invocation(context, spec, TOKEN, opener)
            metadata.assert_called_once_with(opener,
                f"https://api.github.com/repos/{b.t.REPO}/actions/workflows/303/runs"
                f"?event=workflow_dispatch&branch={spec['branch']}&per_page=2", TOKEN)
        bad_rows = [dict(run, **{key: value}) for key, value in
                    (("id", 203), ("id", True), ("run_attempt", 2), ("head_sha", "c" * 40),
                     ("head_branch", "main"), ("workflow_id", 304), ("event", "push"),
                     ("status", "completed"), ("conclusion", "failure"), ("path", "other.yml"),
                     ("head_repository", dict(id=101, full_name="fork/repo")))]
        listings = [dict(total_count=1, workflow_runs=[row]) for row in bad_rows]
        listings += [dict(total_count=True, workflow_runs=[run]), dict(total_count=2, workflow_runs=[run, run]),
                     dict(total_count=0, workflow_runs=[])]
        for listing in listings:
            with self.subTest(listing=listing), patch.object(b.g, "_metadata", return_value=listing), \
                    self.assertRaises(ValueError):
                b.unique_invocation(context, spec, TOKEN, opener)

    def test_owner_preserves_first_success_and_timeout_no_token_in_argv(self):
        spec, _, context, transfer, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "bridge"

            def worker(command, **kwargs):
                (output / "ciphertext" / spec["input"]["object"]["asset_name"]).write_bytes(b"testonly")
                b.c.save(output / "verified.json", dict(context=context, transfer=transfer))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(b, "checked_context", return_value=(context, spec)), \
                    patch.object(b.owned_process, "run", side_effect=worker) as child:
                result = b.run(output)
                self.assertEqual(result["status"], "verified_ciphertext_for_staging")
                self.assertFalse(result["release_ready"])
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn(TOKEN, str(child.call_args))
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)
                with self.assertRaises(FileExistsError):
                    b.run(output)
                self.assertEqual(child.call_count, 1)

    def test_owner_keeps_failed_receipt_and_never_retries_worker(self):
        spec, _, context, transfer, _ = fixture()
        for mode in ("exit", "timeout", "wrong_bytes", "wrong_receipt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "bridge"

                def worker(command, **kwargs):
                    if mode == "timeout":
                        raise subprocess.TimeoutExpired(command, 300)
                    if mode == "exit":
                        return subprocess.CompletedProcess(command, 1)
                    (output / "ciphertext" / spec["input"]["object"]["asset_name"]).write_bytes(
                        b"notright" if mode == "wrong_bytes" else b"testonly")
                    receipt = dict(context=context, transfer=dict(transfer, asset_id=999))
                    b.c.save(output / "verified.json", receipt)
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(b, "checked_context", return_value=(context, spec)), \
                        patch.object(b.owned_process, "run", side_effect=worker) as child:
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                        b.run(output)
                    self.assertEqual(b.t.read_record(output / "result.json")["status"], "failed")
                    with self.assertRaises(FileExistsError):
                        b.run(output)
                    self.assertEqual(child.call_count, 1)

    def test_worker_checks_claim_and_unique_run_before_download(self):
        spec, _, context, transfer, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            b.c.save(output / "claim.json", context)
            with patch.object(b, "checked_context", return_value=(context, spec)), \
                    patch.dict(os.environ, dict(GH_TOKEN=TOKEN), clear=True), \
                    patch.object(b, "unique_invocation") as unique, \
                    patch.object(b.g, "download", return_value=transfer) as download:
                b.fetch_worker(output)
                unique.assert_called_once_with(context, spec, TOKEN)
                download.assert_called_once_with(spec["input"], TOKEN,
                    output / "ciphertext" / spec["input"]["object"]["asset_name"])
                unique.side_effect = ValueError("duplicate run")
                with self.assertRaises(ValueError):
                    b.fetch_worker(output)
                self.assertEqual(download.call_count, 1)

    def test_workflow_has_only_one_cpu_write_job_no_decryption_or_learning(self):
        spec, *_ = fixture()
        text = b.render(spec)
        self.assertEqual(text.count("contents: write"), 1)
        self.assertIn("jobs:\n  ciphertext-bridge:\n    runs-on: ubuntu-24.04", text)
        self.assertIn("timeout-minutes: 15", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("compression-level: 0", text)
        self.assertIn("ciphertext/test-input.zip.age", text)
        self.assertEqual(text.count("GH_TOKEN:"), 1)
        self.assertEqual(text.count("always()"), 1)
        self.assertLess(text.index("Stage only"), text.index("always()"))
        for forbidden in ("secrets[", "--decrypt", "pip install", "cargo", "INPUT_IDENTITY", "actions: write", "gpu"):
            self.assertNotIn(forbidden, text)

    def test_failed_bridge_retains_partial_ciphertext_without_staging_it_as_valid(self):
        spec, *_ = fixture()
        text = b.render(spec)
        name = "      - name: Retain failed first ciphertext without promoting it\n"
        self.assertIn(name, text)
        partial = text[text.index(name):text.index("      - name: Retain the first bridge outcome")]
        self.assertIn("if: ${{ failure() }}", partial)
        self.assertIn("name: bridge-partial-", partial)
        self.assertIn("ciphertext/test-input.zip.age", partial)
        self.assertIn("if-no-files-found: ignore", partial)


if __name__ == "__main__":
    import unittest
    unittest.main()
