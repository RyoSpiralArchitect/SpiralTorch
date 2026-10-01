#!/usr/bin/env node
// Loopback-only fixture server. Browser control belongs to the caller; no profiles or automation here.
const fs = require("node:fs"), path = require("node:path"), http = require("node:http");
const crypto = require("node:crypto");
const [moduleDir, pythonFixture, outputDir, portText = "8768"] = process.argv.slice(2);
if (!moduleDir || !pythonFixture || !outputDir) throw Error("usage: serve_vision_trainer_fixture.cjs MODULE_DIR PYTHON_FIXTURE NEW_OUTPUT_DIR [PORT]");
const port = Number(portText);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error("invalid port");
fs.mkdirSync(outputDir);
const files = new Map([
  ["/", [path.resolve(__dirname, "../bindings/st-wasm/tests/vision_trainer_clients.html"), "text/html"]],
  ["/python.json", [path.resolve(pythonFixture), "application/json"]],
]);
const root = path.resolve(moduleDir);
function add(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) add(file);
    else if (entry.isFile() && /\.(js|wasm)$/.test(entry.name)) files.set(
      "/module/" + path.relative(root, file).split(path.sep).join("/"),
      [file, entry.name.endsWith(".wasm") ? "application/wasm" : "text/javascript"]);
  }
}
add(root);
const assets = Object.fromEntries([...files].map(([url, [file]]) =>
  [url, crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex")]));
const saved = new Map();
const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  response.setHeader("Cache-Control", "no-store");
  if (request.headers.host !== `127.0.0.1:${port}`) { response.writeHead(403); response.end(); return; }
  if (request.method === "GET") {
    if (saved.has(pathname)) {
      response.setHeader("Content-Type", "application/json"); response.end(saved.get(pathname)); return;
    }
    const file = files.get(pathname);
    if (!file) { response.writeHead(404); response.end(); return; }
    response.setHeader("Content-Type", file[1]); response.end(fs.readFileSync(file[0])); return;
  }
  const match = pathname.match(/^\/result\/(constant|cosine)-(control|prefix|resume|python)\.json$/);
  if (request.method !== "POST" || !match || request.headers.origin !== `http://127.0.0.1:${port}`) {
    response.writeHead(403); response.end(); return;
  }
  const chunks = []; let bytes = 0;
  request.on("data", chunk => {
    bytes += chunk.length;
    if (bytes > 2 * 1024 * 1024) { request.destroy(); return; }
    chunks.push(chunk);
  });
  request.on("end", () => {
    try {
      const raw = Buffer.concat(chunks).toString("utf8"), result = JSON.parse(raw);
      if (result.schema !== "vision_trainer_browser.v1" || result.schedule !== match[1] || result.mode !== match[2])
        throw Error("result identity mismatch");
      fs.writeFileSync(path.join(outputDir, `${match[1]}-${match[2]}.json`),
        JSON.stringify({assets, received_at: new Date().toISOString(), result}), {flag: "wx"});
      if (result.passed) saved.set(`/saved/${match[1]}-${match[2]}.json`, raw);
      console.log(JSON.stringify({schedule: match[1], mode: match[2], passed: result.passed, error: result.error}));
      response.writeHead(201); response.end("saved");
    } catch (error) { response.writeHead(409); response.end(String(error)); }
  });
});
server.listen(port, "127.0.0.1", () => console.log(`Fixture ready at http://127.0.0.1:${port}/?schedule=constant&mode=control`));
