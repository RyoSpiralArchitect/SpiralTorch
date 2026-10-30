"""Synthetic GET-only GitHub transport checks; no tokens, network or GPU use."""
import copy
from contextlib import redirect_stdout
import io
from email.message import Message
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.response
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_github_asset as g
import byte_release_hosted_v2 as h
import byte_release_transport_workflow_v2 as workflow
from test_byte_release_transport_v2 import Offline, records, controller_fixture, commit, ROOT
from test_byte_release_hosted_v2 import fixture, Remote, RECIPIENT
import test_byte_release_hosted_v2 as hosted_tests

t, c = g.t, g.t.c
TOKEN = "test-only-workflow-token"
SIGNED_URL = "https://release-assets.githubusercontent.com/github-production-release-asset/101/test-only?signature=never-log-this"


def github_records(amendment=None, descriptor=None):
    if amendment is None:
        amendment, descriptor = records()
    amendment, descriptor = copy.deepcopy(amendment), copy.deepcopy(descriptor)
    descriptor["schema"] = t.GITHUB_DESCRIPTOR_SCHEMA
    descriptor["object"] = dict(repository=t.REPO, repository_id=101, release_id=202, asset_id=303,
                                asset_name="test-input.zip.age", release_tag="transport-test-only")
    amendment["descriptor_sha256"] = c.digest(descriptor)
    return amendment, descriptor


def metadata(descriptor):
    locator, expected = descriptor["object"], descriptor["ciphertext"]
    root = g.API_ROOT + "/repos/" + t.REPO
    repository = dict(id=locator["repository_id"], full_name=t.REPO, archived=False)
    asset = dict(id=locator["asset_id"], name=locator["asset_name"], state="uploaded",
                 size=expected["bytes"], digest="sha256:" + expected["sha256"],
                 url=root + f"/releases/assets/{locator['asset_id']}")
    release = dict(id=locator["release_id"], draft=True, published_at=None,
                   tag_name=locator["release_tag"], url=root + f"/releases/{locator['release_id']}",
                   assets=[copy.deepcopy(asset)])
    return repository, release, asset


class Response(io.BytesIO):
    def __init__(self, body, url, status=200, headers=None):
        super().__init__(body)
        self.url, self.status = url, status
        self.headers = dict(headers or {})

    def geturl(self):
        return self.url


def json_response(value, url):
    body = c.encoded(value)
    return Response(body, url, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})


class Script:
    def __init__(self, descriptor, redirect=False):
        self.calls = []
        self.repository, self.release, self.asset = metadata(descriptor)
        root = g.API_ROOT + "/repos/" + t.REPO
        self.responses = [json_response(self.repository, root), json_response(self.release, self.release["url"]),
                          json_response(self.asset, self.asset["url"])]
        if redirect:
            self.responses += [Response(b"", self.asset["url"], 302, {"Location": SIGNED_URL}),
                               Response(b"testonly", SIGNED_URL, headers={"Content-Length": "8"})]
        else:
            self.responses.append(Response(b"testonly", self.asset["url"], headers={"Content-Length": "8"}))
        self.responses.append(json_response(self.release, self.release["url"]))

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected retry or extra request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GitHubReaderOffline(Offline):
    def test_descriptor_pins_the_route_and_disallows_names_or_id_aliases(self):
        amendment, descriptor = github_records()
        t.validate_amendment(amendment, c.digest(amendment), descriptor)
        for key, value in (("repository", "other/repository"), ("repository_id", True),
                           ("release_id", 202.0), ("asset_id", 0), ("asset_id", 2**63),
                           ("asset_name", "../input.age"), ("asset_name", "input.zip"),
                           ("release_tag", "../../tag"), ("url", SIGNED_URL)):
            changed = copy.deepcopy(descriptor)
            changed["object"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                t.validate_descriptor(changed)
        with self.assertRaisesRegex(ValueError, "direct-object"):
            t.validate_url(SIGNED_URL, descriptor)
        changed = copy.deepcopy(descriptor)
        changed["object"]["asset_id"] += 1
        with self.assertRaises(ValueError):
            t.validate_amendment(amendment, c.digest(amendment), changed)

    def test_direct_stream_and_one_redirect_preserve_bytes_and_get_only_scope(self):
        for redirected in (False, True):
            with self.subTest(redirected=redirected), tempfile.TemporaryDirectory() as temporary:
                _amendment, descriptor = github_records()
                script = Script(descriptor, redirect=redirected)
                output = Path(temporary).resolve() / "input.age"
                result = g.download(descriptor, TOKEN, output, script)
                self.assertEqual(output.read_bytes(), b"testonly")
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                self.assertEqual(result["path"], "single_cdn_redirect" if redirected else "direct_api")
                self.assertFalse(result["release_ready"])
                self.assertEqual(len(script.calls), 6 if redirected else 5)
                for request, timeout in script.calls:
                    self.assertEqual(request.get_method(), "GET")
                    self.assertIsNone(request.data)
                    self.assertEqual(timeout, 30)
                    if request.full_url == SIGNED_URL:
                        self.assertIsNone(request.get_header("Authorization"))
                        self.assertIsNone(request.get_header("X-github-api-version"))
                        self.assertNotIn(TOKEN, str(request.header_items()))
                    else:
                        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
                        self.assertTrue(request.full_url.startswith(g.API_ROOT + "/repos/" + t.REPO + "/")
                                        or request.full_url == g.API_ROOT + "/repos/" + t.REPO)
                self.assertNotIn(TOKEN, c.encoded(result).decode())
                self.assertNotIn("never-log-this", c.encoded(result).decode())
                with self.assertRaises(FileExistsError):
                    g.download(descriptor, TOKEN, output, script)
                self.assertEqual(len(script.calls), 6 if redirected else 5)

    def test_default_opener_disables_proxy_and_automatic_redirects(self):
        _amendment, descriptor = github_records()
        script = Script(descriptor, redirect=True)
        with tempfile.TemporaryDirectory() as temporary, patch.object(g.urllib.request, "build_opener", return_value=script) as build:
            g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age")
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], g._NoRedirect)
        self.assertIsInstance(handlers[2], g._RawResponses)
        with self.assertRaises(g.AssetTransferError):
            handlers[1].redirect_request(None, None, 302, "", {}, SIGNED_URL)

    def test_real_urllib_handler_chain_processes_redirect_only_in_our_reader(self):
        _amendment, descriptor = github_records()
        script = Script(descriptor, redirect=True)

        class SyntheticHTTPS(g.urllib.request.HTTPSHandler):
            def https_open(self, request):
                source = script.open(request, request.timeout)
                headers = Message()
                for name, value in source.headers.items():
                    headers[name] = value
                response = urllib.response.addinfourl(source, headers, source.geturl(), source.status)
                response.msg = "synthetic response"
                return response

        opener = g.urllib.request.build_opener(g.urllib.request.ProxyHandler({}), SyntheticHTTPS(),
                                               g._NoRedirect(), g._RawResponses())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "input.age"
            result = g.download(descriptor, TOKEN, output, opener)
            self.assertEqual(result["path"], "single_cdn_redirect")
            self.assertEqual(output.read_bytes(), b"testonly")
        self.assertEqual(len(script.calls), 6)
        self.assertIsNone(script.calls[4][0].get_header("Authorization"))

    def test_unexpected_status_or_transfer_encoding_stops_without_retry(self):
        _amendment, descriptor = github_records()
        for defect in ("throttled", "server_error", "other_redirect", "encoded_binary", "length", "followed"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor, redirect=True)
                response = script.responses[4]
                if defect in ("throttled", "server_error", "other_redirect"):
                    response.status = {"throttled": 429, "server_error": 500, "other_redirect": 307}[defect]
                elif defect == "encoded_binary":
                    response.headers["Content-Encoding"] = "gzip"
                elif defect == "length":
                    response.headers["Content-Length"] = "9"
                else:
                    response.url = "https://other.example.test/asset"
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                self.assertEqual(len(script.calls), 5)

    def test_missing_or_header_injecting_token_stops_before_any_request(self):
        _amendment, descriptor = github_records()
        for token in (None, "", "token\nsecret", "token\rsecret", " token", "x" * 4097):
            script = Script(descriptor)
            with self.subTest(token_type=type(token).__name__), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(g.AssetAccessError):
                    g.download(descriptor, token, Path(temporary).resolve() / "input.age", script)
            self.assertEqual(script.calls, [])

    def test_access_denial_at_any_stage_has_no_retry_or_credential_fallback(self):
        _amendment, descriptor = github_records()
        for index in range(6):
            for status in (401, 403, 404):
                with self.subTest(index=index, status=status), tempfile.TemporaryDirectory() as temporary:
                    script = Script(descriptor, redirect=True)
                    script.responses[index].status = status
                    with self.assertRaises(g.AssetAccessError):
                        g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                    self.assertEqual(len(script.calls), index + 1)

    def test_repository_release_and_asset_identity_changes_stop_before_binary(self):
        _amendment, descriptor = github_records()
        cases = ((0, "id", True), (0, "id", 999), (0, "full_name", "other/repo"), (0, "archived", True),
                 (1, "draft", False), (1, "draft", 1), (1, "published_at", "published"),
                 (1, "tag_name", "changed"), (1, "id", 202.0), (1, "assets", []),
                 (2, "id", True), (2, "size", 8.0), (2, "state", "starter"),
                 (2, "name", "different.age"), (2, "digest", "sha256:" + "0" * 64),
                 (2, "url", "https://other.example.test/asset"))
        for index, key, value in cases:
            with self.subTest(index=index, key=key), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor)
                row = copy.deepcopy((script.repository, script.release, script.asset)[index])
                row[key] = value
                original = script.responses[index]
                script.responses[index] = json_response(row, original.url)
                output = Path(temporary).resolve() / "input.age"
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, output, script)
                self.assertLessEqual(len(script.calls), 3)
                self.assertFalse(output.exists())

    def test_missing_publication_state_or_ambiguous_asset_roster_rejects(self):
        _amendment, descriptor = github_records()
        for defect in ("missing_publication", "duplicate", "id_alias", "list_digest"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor)
                row = copy.deepcopy(script.release)
                if defect == "missing_publication":
                    del row["published_at"]
                elif defect == "duplicate":
                    row["assets"] *= 2
                elif defect == "id_alias":
                    row["assets"][0]["id"] = 303.0
                else:
                    row["assets"][0]["digest"] = "sha256:" + "0" * 64
                script.responses[1] = json_response(row, row["url"])
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                self.assertEqual(len(script.calls), 2)

    def test_metadata_redirect_encoding_size_types_and_json_fail_closed(self):
        _amendment, descriptor = github_records()
        for defect in ("redirect", "encoding", "oversize", "actual_oversize", "length", "media", "duplicate", "list", "nan"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor)
                response = script.responses[0]
                if defect == "redirect":
                    response.status = 302
                    response.headers["Location"] = SIGNED_URL
                elif defect == "encoding":
                    response.headers["Content-Encoding"] = "gzip"
                elif defect == "oversize":
                    response.headers["Content-Length"] = str(t.MAX_RECORD + 1)
                elif defect == "length":
                    response.headers["Content-Length"] = "1"
                elif defect == "media":
                    response.headers["Content-Type"] = "text/html"
                else:
                    body = {"actual_oversize": b"x" * (t.MAX_RECORD + 1), "duplicate": b'{"id":1,"id":2}',
                            "list": b"[]", "nan": b'{"id":NaN}'}[defect]
                    script.responses[0] = Response(body, response.url, headers={"Content-Type": "application/json"})
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                self.assertEqual(len(script.calls), 1)

    def test_redirect_host_path_and_second_redirect_never_receive_authorization(self):
        _amendment, descriptor = github_records()
        bad = (None, SIGNED_URL.replace("https:", "http:"), SIGNED_URL.replace(g.ASSET_HOST, "other.example.test"),
               SIGNED_URL.replace(g.ASSET_HOST, g.ASSET_HOST + ".evil.test"),
               SIGNED_URL.replace(g.ASSET_HOST, g.ASSET_HOST + ":443"),
               SIGNED_URL.replace("https://", "https://user:password@"), SIGNED_URL + "#fragment",
               SIGNED_URL.replace("/test-only?", "/%2e%2e?"), SIGNED_URL.replace("/test-only?", "/%0a?"),
               "https://" + g.ASSET_HOST + "//outside")
        for location in bad:
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor, redirect=True)
                script.responses[3].headers["Location"] = location
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                self.assertEqual(len(script.calls), 4)
        with tempfile.TemporaryDirectory() as temporary:
            script = Script(descriptor, redirect=True)
            script.responses[4].status = 302
            script.responses[4].headers["Location"] = "https://other.example.test/asset"
            with self.assertRaises(g.AssetTransferError):
                g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
            self.assertEqual(len(script.calls), 5)
            self.assertIsNone(script.calls[-1][0].get_header("Authorization"))

    def test_incorrect_ciphertext_is_not_accepted_and_first_bytes_are_retained(self):
        _amendment, descriptor = github_records()
        for body in (b"test", b"changed!", b"testonly-excess"):
            with self.subTest(bytes=len(body)), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor)
                script.responses[3] = Response(body, script.asset["url"])
                output = Path(temporary).resolve() / "input.age"
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, output, script)
                self.assertTrue(output.exists())
                self.assertLessEqual(output.stat().st_size, descriptor["ciphertext"]["bytes"])
                self.assertEqual(len(script.calls), 4)

    def test_post_transfer_publication_or_replacement_keeps_bytes_but_never_accepts(self):
        _amendment, descriptor = github_records()
        for defect in ("published", "missing", "replaced"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor, redirect=True)
                row = copy.deepcopy(script.release)
                if defect == "published":
                    row["draft"] = False
                elif defect == "missing":
                    row["assets"] = []
                else:
                    row["assets"][0]["digest"] = "sha256:" + "0" * 64
                script.responses[-1] = json_response(row, row["url"])
                output = Path(temporary).resolve() / "input.age"
                with self.assertRaises(g.AssetTransferError):
                    g.download(descriptor, TOKEN, output, script)
                self.assertEqual(output.read_bytes(), b"testonly")
                self.assertEqual(len(script.calls), 6)

    def test_transport_exceptions_do_not_echo_token_signed_url_or_body(self):
        _amendment, descriptor = github_records()
        for index in (0, 4):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                script = Script(descriptor, redirect=True)
                script.responses[index] = RuntimeError(TOKEN + " " + SIGNED_URL + " private body")
                with self.assertRaises(g.AssetTransferError) as caught:
                    g.download(descriptor, TOKEN, Path(temporary).resolve() / "input.age", script)
                for secret in (TOKEN, SIGNED_URL, "private body"):
                    self.assertNotIn(secret, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertEqual(len(script.calls), index + 1)


def github_controller(root):
    source, controller, amendment, descriptor = controller_fixture(root)
    amendment, descriptor = github_records(amendment, descriptor)
    (controller / t.DESCRIPTOR).write_bytes(c.encoded(descriptor))
    (controller / t.GITHUB_RUNTIME_FILE).write_bytes((ROOT / t.GITHUB_RUNTIME_FILE).read_bytes())
    path = controller / ".github/workflows" / t.WORKFLOW
    path.write_text(workflow.render((source / ".github/workflows" / t.WORKFLOW).read_text(), github_asset=True))
    head, tree = commit(controller)
    amendment.update(controller_commit=head, controller_tree=tree)
    return source, controller, amendment, descriptor


class GitHubControllerOffline(Offline):
    def test_rendering_keeps_permissions_and_execution_but_isolates_the_read_token(self):
        baseline = (ROOT / ".github/workflows" / t.WORKFLOW).read_text()
        rendered = workflow.render(baseline, github_asset=True)
        start = "      - name: Install the same wheel offline without cached Python code\n"
        end = "      - name: Encrypt the first outcome including incomplete evidence\n"
        self.assertEqual(baseline[baseline.index(start):baseline.index(end)], rendered[rendered.index(start):rendered.index(end)])
        self.assertIn("permissions:\n  contents: read\n  actions: read\n", rendered)
        self.assertNotIn("contents: write", rendered)
        self.assertNotIn("INPUT_OBJECT_URL", rendered)
        self.assertNotIn("_INPUT_URL", rendered)
        self.assertEqual(rendered.count("GH_TOKEN: " + workflow.EXPRESSION + " github.token }}"), 2)
        self.assertLess(rendered.index("unset GH_TOKEN GITHUB_TOKEN"), rendered.index("prepare-input"))
        self.assertNotIn("Bearer", rendered)
        with self.assertRaises(ValueError):
            workflow.render(baseline, github_asset="yes")

    def test_actual_controller_requires_github_reader_and_exact_permission_envelope(self):
        for defect in (None, "missing_reader", "write_permission", "wrong_route"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                source, controller, amendment, descriptor = github_controller(Path(temporary).resolve())
                path = controller / ".github/workflows" / t.WORKFLOW
                if defect == "missing_reader":
                    (controller / t.GITHUB_RUNTIME_FILE).unlink()
                elif defect == "write_permission":
                    path.write_text(path.read_text().replace("contents: read", "contents: write"))
                elif defect == "wrong_route":
                    path.write_text(workflow.render((source / ".github/workflows" / t.WORKFLOW).read_text()))
                if defect:
                    head, tree = commit(controller)
                    amendment.update(controller_commit=head, controller_tree=tree)
                    with self.assertRaises(ValueError):
                        t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))
                else:
                    self.assertEqual(t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment)),
                                     amendment["controller_commit"])

    def test_dispatch_checks_only_input_identity_and_return_remains_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, admission, _archive, amendment, descriptor, controller = fixture(root)
            amendment, descriptor = github_records(amendment, descriptor)
            (controller / t.DESCRIPTOR).write_bytes(c.encoded(descriptor))
            remote = Remote(amendment)
            with patch.object(t, "validate_controller", return_value=amendment["controller_commit"]), \
                    patch.object(c, "source_identity", return_value=admission["source"]), patch.object(h, "api", remote):
                value = h.dispatch(ledger, controller, "synthetic", amendment, c.digest(amendment), RECIPIENT, execute=True)
                self.assertTrue(value["dispatched"])
                self.assertFalse(any("INPUT_URL" in path for path, _ in remote.calls))
                self.assertNotIn(TOKEN, c.encoded(value).decode())
                self.assertEqual(len(remote.posts), 1)
                remote.rows = [remote.run(status="completed", conclusion="success")]
                h.bind_run(ledger / "candidate-1")
                packet, receipt = hosted_tests.HostedOffline.return_packet(self, root, ledger, admission, amendment, descriptor)
                outcome = h.recover_result(ledger / "candidate-1", packet, receipt["archive_sha256"])
                self.assertEqual(outcome["status"], "verified_rounds")
                self.assertFalse(outcome["release_ready"])

    def test_fetch_owner_has_shared_deadline_no_url_fallback_and_no_token_in_argv(self):
        amendment, descriptor = github_records()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "input.age"
            env = dict(GH_TOKEN=TOKEN)

            def worker(command, **kwargs):
                output.write_bytes(b"testonly")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(h.owned_process, "run", side_effect=worker) as child:
                for invalid in ({}, dict(env, INPUT_OBJECT_URL=SIGNED_URL)):
                    with self.assertRaises(ValueError):
                        h.fetch_input(output, invalid)
                child.assert_not_called()
                self.assertEqual(h.fetch_input(output, env)["status"], "verified_ciphertext")
                self.assertEqual(child.call_args.kwargs["timeout"], 300)
                self.assertNotIn(TOKEN, str(child.call_args.args))
                with self.assertRaises(FileExistsError):
                    h.fetch_input(output, env)
                self.assertEqual(child.call_count, 1)

    def test_worker_routes_only_from_the_committed_descriptor(self):
        amendment, descriptor = github_records()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "input.age"
            with patch.object(h, "admitted_context", return_value=(amendment, descriptor, c.digest(amendment))), \
                    patch.object(g, "download", return_value={"release_ready": False}) as download, \
                    patch.object(h, "download_ciphertext", side_effect=AssertionError("direct fallback forbidden")) as direct, \
                    patch.dict(os.environ, {"GH_TOKEN": TOKEN}, clear=True), redirect_stdout(io.StringIO()):
                self.assertEqual(h.main(["fetch-worker", "--output", str(output)]), 0)
                download.assert_called_once_with(descriptor, TOKEN, output)
                direct.assert_not_called()
                with patch.dict(os.environ, {"INPUT_OBJECT_URL": SIGNED_URL}):
                    with self.assertRaises(ValueError):
                        h.main(["fetch-worker", "--output", str(output)])
                self.assertEqual(download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
