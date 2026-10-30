"""Offline authenticated-download and credential-boundary checks."""
import copy
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_download as d
from test_byte_release_actions_input import records, zip_bytes, PAYLOAD
from test_byte_release_github_asset import Response, json_response, TOKEN
from test_byte_release_transport_v2 import Offline

CDN = "https://productionresultssa1.blob.core.windows.net/actions-results/opaque?sig=test-only-never-log"


class Script:
    def __init__(self, pin, run, artifact, raw, redirect=True):
        self.calls = []
        root = f"{d.g.API_ROOT}/repos/{d.a.t.REPO}/actions"
        self.run_url = root + f"/runs/{pin['bridge']['run_id']}"
        self.asset_url = root + f"/artifacts/{pin['artifact']['id']}"
        self.binary_url = self.asset_url + "/zip"
        self.responses = [json_response(run, self.run_url), json_response(artifact, self.asset_url)]
        if redirect:
            self.responses += [Response(b"", self.binary_url, 302, {"Location": CDN}),
                               Response(raw, CDN, headers={"Content-Length": str(len(raw))})]
        else:
            self.responses += [Response(raw, self.binary_url, headers={"Content-Length": str(len(raw))})]
        self.responses += [json_response(run, self.run_url), json_response(artifact, self.asset_url)]

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected retry")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class ActionsDownloadOffline(Offline):
    def test_direct_or_one_cdn_hop_verifies_whole_archive_and_ciphertext(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        for redirected in (False, True):
            with self.subTest(redirected=redirected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                script = Script(pin, run, artifact, raw, redirected)
                result = d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                self.assertEqual((root / "input.zip").read_bytes(), raw)
                self.assertEqual((root / "input.age").read_bytes(), PAYLOAD)
                self.assertFalse(result["release_ready"])
                self.assertEqual(len(script.calls), 6 if redirected else 5)
                for path in (root / "input.zip", root / "input.age"):
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                for request, timeout in script.calls:
                    self.assertEqual(request.get_method(), "GET")
                    self.assertEqual(timeout, 30)
                    self.assertIsNone(request.data)
                    if request.full_url == CDN:
                        self.assertIsNone(request.get_header("Authorization"))
                        self.assertIsNone(request.get_header("X-github-api-version"))
                    else:
                        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
                for secret in (TOKEN, CDN, "test-only-never-log"):
                    self.assertNotIn(secret, d.a.t.c.encoded(result).decode())

    def test_missing_token_existing_paths_and_links_stop_before_network(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = Script(pin, run, artifact, raw)
            for token in (None, "", "x\nsecret", "x" * 4097):
                with self.assertRaises(d.g.AssetAccessError):
                    d.download(pin, token, root / "input.zip", root / "input.age", script)
            for name in ("input.zip", "input.age"):
                path = root / name
                path.symlink_to(root / "absent")
                with self.assertRaises(FileExistsError):
                    d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                path.unlink()
            with self.assertRaises(ValueError):
                d.download(pin, TOKEN, root / "same", root / "same", script)
            self.assertEqual(script.calls, [])

    def test_default_opener_disables_proxy_and_automatic_redirects(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        script = Script(pin, run, artifact, raw)
        with tempfile.TemporaryDirectory() as temporary, patch.object(d.g.urllib.request, "build_opener", return_value=script) as build:
            root = Path(temporary).resolve()
            d.download(pin, TOKEN, root / "input.zip", root / "input.age")
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(build.call_args.args[1], d.g._NoRedirect)
        self.assertIsInstance(build.call_args.args[2], d.g._RawResponses)

    def test_redirect_rejects_foreign_authority_paths_and_credentials(self):
        self.assertEqual(d.validate_redirect(CDN), CDN)
        self.assertEqual(d.validate_redirect("https://results-receiver.actions.githubusercontent.com/a?sig=fake"),
                         "https://results-receiver.actions.githubusercontent.com/a?sig=fake")
        for url in (CDN.replace("https:", "http:"), CDN.replace(".net/", ".net:443/"),
                    CDN.replace(".net/", ".net.evil.invalid/"), CDN.replace("https://", "https://token@"),
                    CDN + "#frag", CDN.replace("/actions-results/", "/../"),
                    CDN.replace("/actions-results/", "/%2e%2e/"), CDN.replace("/opaque", "/%5copaq"),
                    "https://127.0.0.1/input", "https://blob.core.windows.net/input",
                    "https://release-assets.githubusercontent.com/input", CDN + "\n"):
            with self.subTest(url=url), self.assertRaises(d.g.AssetTransferError):
                d.validate_redirect(url)

    def test_access_and_server_failures_at_each_stage_stop_without_retry(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        for index in range(6):
            for status in (401, 403, 404, 429, 500, 307):
                with self.subTest(index=index, status=status), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    script = Script(pin, run, artifact, raw)
                    script.responses[index].status = status
                    with self.assertRaises(d.g.AssetTransferError):
                        d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                    self.assertEqual(len(script.calls), index + 1)
                    self.assertFalse((root / "input.age").exists())

    def test_changed_metadata_before_or_after_never_extracts_or_retries(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        for index in (0, 1, 4, 5):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                script = Script(pin, run, artifact, raw)
                changed = copy.deepcopy(run if index in (0, 4) else artifact)
                changed["run_attempt" if index in (0, 4) else "expired"] = 2 if index in (0, 4) else True
                script.responses[index] = json_response(changed, script.responses[index].url)
                with self.assertRaises(d.g.AssetTransferError):
                    d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                self.assertFalse((root / "input.age").exists())
                self.assertEqual(len(script.calls), 2 if index < 2 else 6)
                self.assertEqual((root / "input.zip").exists(), index >= 4)

    def test_truncation_overflow_or_wrong_digest_keeps_first_partial_archive(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        for payload in (raw[:-1], raw + b"x", b"x" * len(raw)):
            with self.subTest(length=len(payload)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                script = Script(pin, run, artifact, raw)
                script.responses[3] = Response(payload, CDN)
                with self.assertRaises(d.g.AssetTransferError):
                    d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                self.assertTrue((root / "input.zip").exists())
                self.assertFalse((root / "input.age").exists())
                with self.assertRaises(FileExistsError):
                    d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
                self.assertEqual(len(script.calls), 4)

    def test_exceptions_never_echo_private_values(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = Script(pin, run, artifact, raw)
            script.responses[3] = RuntimeError(TOKEN + CDN + " private body")
            with self.assertRaises(d.g.AssetTransferError) as caught:
                d.download(pin, TOKEN, root / "input.zip", root / "input.age", script)
            for secret in (TOKEN, CDN, "private body"):
                self.assertNotIn(secret, str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)


if __name__ == "__main__":
    import unittest
    unittest.main()
