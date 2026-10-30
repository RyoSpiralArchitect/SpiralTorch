"""CPU ownership and workflow checks for the read-only input probe."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_probe as p
from test_byte_release_actions_input import records, zip_bytes, PAYLOAD
from test_byte_release_github_asset import TOKEN
from test_byte_release_transport_v2 import Offline


def fixture():
    raw = zip_bytes()
    pin, run, _artifact = records(raw)
    env = dict(GITHUB_REPOSITORY=p.t.REPO, GITHUB_SHA="a" * 40, EXPECTED_READER_HEAD="a" * 40,
        EXPECTED_PIN_SHA256=p.c.digest(pin), GITHUB_REF="refs/heads/" + p.BRANCH,
        GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="505", GH_TOKEN=TOKEN)
    context = dict(schema=p.SCHEMA, head="a" * 40, tree="b" * 40, pin_sha256=p.c.digest(pin),
        run_id=505, run_attempt=1, branch=p.BRANCH, workflow_id=303, fetch_seconds=300, release_ready=False)
    run.update(id=505, head_branch=p.BRANCH, status="in_progress", conclusion=None)
    return raw, pin, env, context, run


class ReadOnlyProbeOffline(Offline):
    def test_context_is_small_first_bound_clean_and_keyless(self):
        _raw, pin, env, context, _run = fixture()
        with patch.object(p.t, "read_record", return_value=pin), \
                patch.object(p.c, "source_identity", return_value=dict(commit="a" * 40, tree="b" * 40)):
            self.assertEqual(p.checked_context(env), (context, pin))
            for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main"),
                               ("GITHUB_EVENT_NAME", "push"), ("GITHUB_REPOSITORY", "fork/repo"),
                               ("GITHUB_SHA", "b" * 40), ("EXPECTED_PIN_SHA256", "0" * 64),
                               ("GITHUB_RUN_ID", "-1"), ("INPUT_IDENTITY", "not-a-real-key"),
                               ("INPUT_OBJECT_URL", "never-fetch"), ("GITHUB_TOKEN", TOKEN), ("GH_TOKEN", "")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    p.checked_context(dict(env, **{key: value}))
            pin["artifact"]["bytes"] = 65537
            with self.assertRaises(ValueError):
                p.checked_context(env)

    def test_unique_invocation_rejects_duplicates_forks_and_changed_attempts(self):
        _, pin, _, context, run = fixture()
        with patch.object(p.g, "_metadata", return_value=dict(total_count=1, workflow_runs=[run])):
            p.unique_invocation(context, pin, TOKEN)
        for listing in (dict(total_count=2, workflow_runs=[run, run]), dict(total_count=True, workflow_runs=[run]),
                        dict(total_count=1, workflow_runs=[dict(run, run_attempt=2)]),
                        dict(total_count=1, workflow_runs=[dict(run, head_repository=dict(id=101, full_name="fork/repo"))])):
            with patch.object(p.g, "_metadata", return_value=listing), self.assertRaises(ValueError):
                p.unique_invocation(context, pin, TOKEN)

    def test_owner_accepts_exact_proof_with_one_shared_deadline(self):
        raw, pin, _, context, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "read"

            def worker(command, **kwargs):
                (output / "input.zip").write_bytes(raw)
                (output / "input.age").write_bytes(PAYLOAD)
                p.c.save(output / "download.json", dict(context=context, transfer=p.expected_receipt(pin)))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(p, "checked_context", return_value=(context, pin)), \
                    patch.object(p.owned_process, "run", side_effect=worker) as child:
                result = p.run(output)
                self.assertEqual(result["status"], "verified_read_only_ciphertext")
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn(TOKEN, str(child.call_args))
                with self.assertRaises(FileExistsError):
                    p.run(output)
                self.assertEqual(child.call_count, 1)

    def test_owner_keeps_failed_exit_timeout_changed_bytes_and_missing_proof(self):
        raw, pin, _, context, _ = fixture()
        for mode in ("exit", "timeout", "bytes", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "read"

                def worker(command, **kwargs):
                    if mode == "timeout":
                        raise subprocess.TimeoutExpired(command, 300)
                    if mode == "exit":
                        return subprocess.CompletedProcess(command, 1)
                    (output / "input.zip").write_bytes(raw)
                    (output / "input.age").write_bytes(PAYLOAD + (b"x" if mode == "bytes" else b""))
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(p, "checked_context", return_value=(context, pin)), \
                        patch.object(p.owned_process, "run", side_effect=worker) as child:
                    with self.assertRaises((ValueError, FileNotFoundError, subprocess.TimeoutExpired)):
                        p.run(output)
                    self.assertEqual(p.t.read_record(output / "result.json")["status"], "failed")
                    with self.assertRaises(FileExistsError):
                        p.run(output)
                    self.assertEqual(child.call_count, 1)

    def test_worker_checks_unique_consumer_before_read_only_download(self):
        _, pin, _, context, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            p.c.save(output / "claim.json", context)
            with patch.object(p, "checked_context", return_value=(context, pin)), \
                    patch.dict(os.environ, dict(GH_TOKEN=TOKEN), clear=True), \
                    patch.object(p, "unique_invocation") as unique, \
                    patch.object(p.download, "download", return_value=p.expected_receipt(pin)) as fetch:
                p.fetch_worker(output)
                unique.assert_called_once_with(context, pin, TOKEN)
                fetch.assert_called_once_with(pin, TOKEN, output / "input.zip", output / "input.age")
                unique.side_effect = ValueError("already submitted")
                with self.assertRaises(ValueError):
                    p.fetch_worker(output)
                self.assertEqual(fetch.call_count, 1)

    def test_render_has_no_write_key_decrypt_or_gpu_command(self):
        text = p.render()
        self.assertIn("permissions:\n  contents: read\n  actions: read", text)
        self.assertEqual(text.count("permissions:"), 1)
        self.assertIn("runs-on: macos-26", text)
        self.assertIn("timeout-minutes: 15", text)
        self.assertIn("persist-credentials: false", text)
        self.assertEqual(text.count("GH_TOKEN:"), 1)
        for forbidden in ("contents: write", "actions: write", "secrets[", "--decrypt", "pip install", "cargo"):
            self.assertNotIn(forbidden, text)

    def test_renderer_retains_first_partial_archive_and_ciphertext(self):
        text = p.render()
        self.assertIn("${{ runner.temp }}/actions-input-probe/input.zip", text)
        self.assertIn("${{ runner.temp }}/actions-input-probe/input.age", text)
        self.assertIn("if: ${{ always() }}", text)


if __name__ == "__main__":
    import unittest
    unittest.main()
