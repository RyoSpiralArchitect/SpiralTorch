"""Local server regressions with synthetic bytes, no GPU or real private inputs."""
import http.client
from contextlib import closing
import importlib.util
import json
import pathlib
import tempfile
import threading
import types
import unittest

spec = importlib.util.spec_from_file_location("qualify", pathlib.Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class QualificationServer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temporary.name)
        self.args = types.SimpleNamespace(repo=root / "repo", package=root / "package", output=root / "private", port=0)
        tests = self.args.repo / "bindings/st-wasm/tests"
        tests.mkdir(parents=True)
        self.args.package.mkdir()
        self.args.output.mkdir()
        for name in ("byte_lm_browser.html", "byte_lm_browser.mjs", "byte_lm_boundaries.mjs"):
            (tests / name).write_text("synthetic asset")
        (self.args.package / "spiraltorch_wasm.js").write_text("export {};")
        (self.args.output / "input.json").write_text('{"synthetic":true}')
        manifest = {"source": {str(p.relative_to(self.args.repo)): qualify.identity(p) for p in tests.iterdir()},
                    "package": {p.name: qualify.identity(p) for p in self.args.package.iterdir()},
                    "fixture": qualify.identity(self.args.output / "input.json")}
        (self.args.output / "inputs.json").write_text(json.dumps(manifest))
        self.server = qualify.create_server(self.args)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        with closing(http.client.HTTPConnection("127.0.0.1", self.server.server_port)) as conn:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read()

    def test_private_routes_reject_foreign_host(self):
        for route in ("/fixtures/input.json", "/results/training.json", "/"):
            self.assertEqual(self.request("GET", route, headers={"Host": "attacker.invalid"})[0], 403)
        self.assertEqual(self.request("GET", "/fixtures/input.json")[0], 200)
        self.assertEqual(self.request("GET", "/private/input.json")[0], 404)

    def test_assets_are_checked_then_frozen(self):
        original = (self.args.output / "input.json").read_bytes()
        (self.args.output / "input.json").write_text("altered after startup")
        self.assertEqual(self.request("GET", "/fixtures/input.json"), (200, original))
        with self.assertRaisesRegex(ValueError, "changed asset"):
            qualify.create_server(self.args)

    def test_result_writes_are_origin_bound_exclusive_and_frozen(self):
        route, body = "/results/training.json", b'{"synthetic":"checkpoint"}'
        self.assertEqual(self.request("POST", route, body, {"Origin": "http://attacker.invalid"})[0], 403)
        headers = {"Origin": f"http://127.0.0.1:{self.server.server_port}"}
        self.assertEqual(self.request("POST", route, body, headers)[0], 204)
        (self.args.output / "training.json").write_text("altered after POST")
        self.assertEqual(self.request("GET", route), (200, body))
        self.assertEqual(self.request("POST", route, body, headers)[0], 409)


if __name__ == "__main__":
    unittest.main()
