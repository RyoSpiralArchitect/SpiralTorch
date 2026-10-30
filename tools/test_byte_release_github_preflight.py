"""CPU fixtures only; no real token, network, candidate or native imports."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_github_preflight as p
from test_byte_release_github_asset import github_records


def fixture():
    _amendment, descriptor = github_records()
    descriptor.update(admission_sha256=p.SYNTHETIC, source_commit=p.SOURCE,
                      plaintext=dict(bytes=len(p.TEXT), sha256=hashlib.sha256(p.TEXT).hexdigest()))
    env = dict(EXPECTED_PREFLIGHT_HEAD="a" * 40, GITHUB_SHA="a" * 40,
               EXPECTED_DESCRIPTOR_SHA256=p.c.digest(descriptor), GITHUB_REPOSITORY=p.t.REPO,
               GITHUB_REF="refs/heads/" + p.BRANCH, GITHUB_EVENT_NAME="workflow_dispatch",
               GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1", GH_TOKEN="fake-test-token")
    return descriptor, env


class PreflightOffline(unittest.TestCase):
    def context(self, descriptor, env):
        with patch.object(p.t, "read_record", return_value=descriptor), patch.object(
                p.c, "source_identity", return_value=dict(commit="a" * 40, tree="b" * 40)):
            return p.checked_context(env)

    def test_context_binds_first_attempt_branch_source_and_small_synthetic_input(self):
        descriptor, env = fixture()
        context, observed = self.context(descriptor, env)
        self.assertEqual(observed, descriptor)
        self.assertTrue(context["preflight_only"])
        self.assertFalse(context["release_ready"])
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_RUN_ID", "0"),
                           ("GITHUB_RUN_ID", "1e3"), ("GITHUB_EVENT_NAME", "push"),
                           ("GITHUB_REF", "refs/heads/main"), ("GITHUB_SHA", "c" * 40),
                           ("EXPECTED_PREFLIGHT_HEAD", "invalid"), ("EXPECTED_DESCRIPTOR_SHA256", "0" * 64),
                           ("GITHUB_REPOSITORY", "other/repo"), ("GH_TOKEN", ""),
                           ("GITHUB_TOKEN", "fake"), ("INPUT_IDENTITY", "fake"), ("INPUT_OBJECT_URL", "fake")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.context(descriptor, dict(env, **{key: value}))
        for key, value in (("admission_sha256", "f" * 64), ("source_commit", "f" * 40),
                           ("candidate", 2), ("plaintext", dict(bytes=1, sha256="0" * 64)),
                           ("ciphertext", dict(bytes=65537, sha256="0" * 64))):
            changed = dict(descriptor, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.context(changed, dict(env, EXPECTED_DESCRIPTOR_SHA256=p.c.digest(changed)))

    def test_checkout_must_be_the_pinned_head(self):
        descriptor, env = fixture()
        with patch.object(p.t, "read_record", return_value=descriptor), patch.object(
                p.c, "source_identity", return_value=dict(commit="c" * 40, tree="b" * 40)), self.assertRaises(ValueError):
            p.checked_context(env)

    def worker(self, directory, context, descriptor):
        (directory / "input.age").write_bytes(b"testonly")
        transfer = dict(status="verified_ciphertext", **descriptor["ciphertext"],
                        **{key: descriptor["object"][key] for key in ("repository_id", "release_id", "asset_id")},
                        path="single_cdn_redirect", release_ready=False)
        p.c.save(directory / "verified.json", dict(context=context, transfer=transfer))
        return subprocess.CompletedProcess([], 0)

    def test_success_is_once_owned_and_not_qualification(self):
        descriptor, env = fixture()
        context, _ = self.context(descriptor, env)
        with tempfile.TemporaryDirectory() as temporary, patch.object(p, "checked_context", return_value=(context, descriptor)):
            directory = Path(temporary).resolve() / "probe"
            with patch.object(p.owned_process, "run", side_effect=lambda *a, **k: self.worker(directory, context, descriptor)) as child:
                result = p.run(directory)
                self.assertEqual(result["status"], "verified_small_ciphertext")
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn("fake-test-token", str(child.call_args))
                self.assertNotIn("--execute", child.call_args.args[0])
                with self.assertRaises(FileExistsError):
                    p.run(directory)
                self.assertEqual(child.call_count, 1)
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual((directory / "worker.log").stat().st_mode & 0o777, 0o600)
            self.assertFalse(result["context"]["release_ready"])

    def test_first_failure_timeout_and_partial_bytes_are_retained(self):
        descriptor, env = fixture()
        context, _ = self.context(descriptor, env)
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary, patch.object(
                    p, "checked_context", return_value=(context, descriptor)):
                directory = Path(temporary).resolve() / "probe"
                def fail(*_args, **_kwargs):
                    (directory / "input.age").write_bytes(b"partial")
                    if timeout:
                        raise subprocess.TimeoutExpired(["safe-command"], 300)
                    return subprocess.CompletedProcess([], 1)
                with patch.object(p.owned_process, "run", side_effect=fail) as child:
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                        p.run(directory)
                    with self.assertRaises(FileExistsError):
                        p.run(directory)
                    self.assertEqual(child.call_count, 1)
                self.assertEqual((directory / "input.age").read_bytes(), b"partial")
                result = p.t.read_record(directory / "result.json")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["context"]["release_ready"])

    def test_zero_exit_cannot_hide_wrong_bytes_or_unbound_receipt(self):
        descriptor, env = fixture()
        context, _ = self.context(descriptor, env)
        for defect in ("bytes", "context", "typed_alias", "extra"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary, patch.object(
                    p, "checked_context", return_value=(context, descriptor)):
                directory = Path(temporary).resolve() / "probe"
                def wrong(*_args, **_kwargs):
                    receipt_context = dict(context, run_attempt=True) if defect == "context" else context
                    self.worker(directory, receipt_context, descriptor)
                    if defect == "bytes":
                        (directory / "input.age").write_bytes(b"wrong")
                    elif defect in ("typed_alias", "extra"):
                        receipt = p.t.read_record(directory / "verified.json")
                        if defect == "typed_alias":
                            receipt["transfer"]["asset_id"] = float(receipt["transfer"]["asset_id"])
                        else:
                            receipt["transfer"]["extra"] = "untrusted"
                        (directory / "verified.json").write_bytes(p.c.encoded(receipt))
                    return subprocess.CompletedProcess([], 0)
                with patch.object(p.owned_process, "run", side_effect=wrong), self.assertRaises(ValueError):
                    p.run(directory)

    def test_worker_uses_only_the_bound_read_token_and_requires_claim(self):
        descriptor, env = fixture()
        context, _ = self.context(descriptor, env)
        with tempfile.TemporaryDirectory() as temporary, patch.object(p, "checked_context", return_value=(context, descriptor)), patch.dict(os.environ, env, clear=True):
            directory = Path(temporary).resolve()
            with patch.object(p.github, "download", return_value={}) as fetch:
                with self.assertRaises(FileNotFoundError):
                    p.fetch_worker(directory)
                fetch.assert_not_called()
                p.c.save(directory / "claim.json", context)
                p.fetch_worker(directory)
                fetch.assert_called_once_with(descriptor, "fake-test-token", directory / "input.age")

    def test_errors_never_echo_secret_or_exception_body(self):
        stream = io.StringIO()
        with patch.object(p, "run", side_effect=ValueError("fake-secret-and-signed-url")), redirect_stderr(stream):
            self.assertEqual(p.main(["run", "--output", "unused"]), 1)
        self.assertNotIn("fake-secret", stream.getvalue())
        self.assertEqual(p.t.parse_record(stream.getvalue())["error_type"], "ValueError")

    def test_workflow_keeps_a_read_only_cpu_envelope(self):
        text = (p.ROOT / ".github/workflows/byte_release_qualification.yml").read_text()
        for required in ("contents: read", "actions: read", "runs-on: macos-26",
                         "timeout-minutes: 15", "python-version: '3.12.10'",
                         "persist-credentials: false", "GH_TOKEN: ${{ github.token }}",
                         "byte_release_github_preflight.py run", "worker.log"):
            self.assertIn(required, text)
        for forbidden in ("contents: write", "id-token:", "secrets.", "secrets[", "pip install",
                          "--execute", "hosted-metal", "input-key", "run_byte_release_qualification.py"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(text.count("GH_TOKEN:"), 1)
        self.assertNotIn("push:", text)


if __name__ == "__main__":
    unittest.main()
