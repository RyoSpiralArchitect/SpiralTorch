"""Private fixture preparation and allowlisted loopback qualification server.

No training-policy edits, model-weight rewriting, or automatic publication.
Raw outputs remain private; publish only reviewed receipts and hashes.
"""
import argparse
import hashlib
import http.server
import json
import mimetypes
import pathlib
import subprocess
import urllib.parse

LIMIT = 64 * 1024 * 1024


def bytes_identity(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def identity(path):
    return bytes_identity(path.read_bytes())


def prepare(args):
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    request_text = args.request.read_text(encoding="utf-8")
    request = json.loads(request_text)
    batch = request["validation_batches"][0]
    steps = request["config"]["steps"]
    rows = [request["validation_documents"][doc][start:start + steps + 1] for doc, start in batch]
    fixture = {
        "request_json": request_text,
        "checkpoint_json": args.checkpoint.read_text(encoding="utf-8"),
        "case_names": [c["name"] for c in request["cases"]],
        "windows": [byte for row in rows for byte in row],
        "prompts": [byte for row in rows for byte in row[:-1]],
        "learning_request_json": args.learning_request.read_text(encoding="utf-8"),
        "serving": json.loads(args.serving.read_bytes()),
    }
    with (args.output / "input.json").open("x", encoding="utf-8") as handle:
        json.dump(fixture, handle, ensure_ascii=True)
    source_paths = [*sorted((args.repo / "bindings/st-wasm/src/byte_lm").glob("*.rs")),
                    args.repo / "bindings/st-wasm/src/byte_lm.rs",
                    args.repo / "bindings/st-wasm/src/lib.rs",
                    args.repo / "bindings/st-wasm/types/spiraltorch-wasm.d.ts",
                    *sorted((args.repo / "bindings/st-wasm/tests").glob("byte_lm*")),
                    pathlib.Path(__file__).resolve(), pathlib.Path(__file__).resolve().with_name("test_qualify.py"),
                    args.repo / "Cargo.lock"]
    manifest = {
        "schema": "spiraltorch.packaged_byte_mlops.inputs.v1",
        "parent_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "scope": "Packaged API qualification; fixed prior inputs; no new quality claim or gate relaxation",
        "inputs": {name: identity(getattr(args, name)) for name in ("request", "checkpoint", "learning_request", "serving")},
        "fixture": identity(args.output / "input.json"),
        "source": {str(p.relative_to(args.repo)): identity(p) for p in source_paths},
        "package": {str(p.relative_to(args.package)): identity(p) for p in sorted(args.package.rglob("*")) if p.is_file()},
        "serving_limits": "Reused unchanged from the pinned historical mechanics gate; not production qualification",
    }
    (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": manifest["fixture"], "cases": fixture["case_names"]}))


def create_server(args):
    tests = args.repo / "bindings/st-wasm/tests"
    reads = {"/": tests / "byte_lm_browser.html", "/fixtures/input.json": args.output / "input.json"}
    for name in ("byte_lm_browser.mjs", "byte_lm_boundaries.mjs"):
        reads["/tests/" + name] = tests / name
    for path in args.package.rglob("*"):
        if path.is_file() and path.suffix in (".js", ".wasm"):
            reads["/package/" + path.relative_to(args.package).as_posix()] = path
    writes = {"/results/" + name + ".json": args.output / (name + ".json")
              for name in ("training", "main", "resume", "failure-main", "failure-resume")}
    manifest = json.loads((args.output / "inputs.json").read_bytes())
    expected = {args.repo / path: value for path, value in manifest["source"].items()}
    expected.update({args.package / path: value for path, value in manifest["package"].items()})
    expected[args.output / "input.json"] = manifest["fixture"]
    assets = {}
    for route, path in reads.items():
        data = path.read_bytes()
        if bytes_identity(data) != expected.get(path):
            raise ValueError(f"unbound or changed asset: {route}")
        assets[route] = (data, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    # Serve only these frozen bytes, not files reread later while the run executes.
    results = {}
    training = args.output / "training.json"
    if training.is_file():
        results["/results/training.json"] = training.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                self.send_error(403)
                return
            route = urllib.parse.urlsplit(self.path).path
            asset = assets.get(route)
            if route == "/results/training.json" and route in results:
                asset = (results[route], "application/json")
            if asset is None:
                self.send_error(404)
                return
            data, mime_type = asset
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = writes.get(self.path)
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if not self.valid_host() or path is None or self.headers.get("Origin") != origin:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= LIMIT:
                    raise ValueError("invalid response size")
                body = self.rfile.read(size)
                json.loads(body)
                with path.open("xb") as handle:
                    handle.write(body)
                if self.path == "/results/training.json":
                    results[self.path] = body
            except FileExistsError:
                self.send_error(409)
                return
            except (ValueError, OSError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

    return http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)


def serve(args):
    with create_server(args) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


def verify(args):
    if not __debug__:
        raise RuntimeError("verification must run without Python -O")
    def load(name):
        return json.loads((args.output / name).read_bytes())

    manifest, browser, resumed = load("inputs.json"), load("main.json"), load("resume.json")
    for path, expected in manifest["source"].items():
        assert identity(args.repo / path) == expected, f"source changed: {path}"
    for path, expected in manifest["package"].items():
        assert identity(args.package / path) == expected, f"package changed: {path}"
    assert identity(args.output / "input.json") == manifest["fixture"]
    fixture, training = load("input.json"), load("training.json")
    request = json.loads(fixture["learning_request_json"])
    assert len(request["cases"]) > 0
    assert [c["name"] for c in browser["training"]["cases"]] == [c["name"] for c in request["cases"]]
    assert browser["training"]["completed_updates"] == len(request["train_batches"])
    assert json.loads(training["checkpoint_json"])["completed_updates"] == len(request["train_batches"])
    assert browser["schema"] == "spiraltorch.packaged_byte_mlops.browser.v1"
    assert browser["count"] == len(browser["checks"])
    assert browser["adapter_requests"] == 1
    assert browser["assessment"]["accepted"] is True
    assert browser["failed_assessment"]["accepted"] is False
    assert browser["assessment"]["adapter"]["backend"] == "BrowserWebGpu"
    assert [m["name"] for m in browser["models"]] == fixture["case_names"]
    assert len(set(m["logits_f32le_sha256"] for m in browser["models"])) == len(fixture["case_names"])
    assert resumed["exact_report"] is True and resumed["exact_checkpoint"] is True
    for key in ("report", "checkpoint"):
        actual = hashlib.sha256(training[key + "_json"].encode()).hexdigest()
        assert actual == resumed[key + "_sha256"] == browser["training"][key + "_sha256"]
    for feature in ("nn", "webgpu"):
        assert load("node-" + feature + ".json")["features"] == feature
        assert load("node-" + feature + ".json")["feature_matrix_passed"] is True
    # Receipts may include metrics/identities but never the private payloads.
    forbidden = {"model_json", "request_json", "checkpoint_json", "final_parameters",
                 "train_documents", "validation_documents", "generated_bytes", "optimizer_state"}

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    for value in (manifest, browser, resumed):
        inspect(value)
    print(json.dumps({"schema": "spiraltorch.packaged_byte_mlops.verification.v1",
                      "verified": True, "checks": browser["count"],
                      "cases": len(browser["training"]["cases"]),
                      "updates": browser["training"]["completed_updates"],
                      "exact_resume": True, "quality_claim": False,
                      "private_payloads_in_receipts": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "serve", "verify"))
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8812)
    for name in ("request", "checkpoint", "learning-request", "serving"):
        parser.add_argument("--" + name, type=pathlib.Path)
    args = parser.parse_args()
    for name in ("repo", "package", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.mode == "prepare":
        if any(getattr(args, name) is None for name in ("request", "checkpoint", "learning_request", "serving")):
            parser.error("prepare requires all four private input paths")
        prepare(args)
    elif args.mode == "serve":
        serve(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
