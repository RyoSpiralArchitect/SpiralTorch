#!/usr/bin/env node
// Isolated headless test browser only: no user profile, cookies, or existing tabs.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const crypto = require("node:crypto");
const {chromium} = require("playwright");

async function main() {
  const [moduleDir, executablePath, outputPath, tiles, kernels, accumulations, shapes, fixture, baselineDir] = process.argv.slice(2);
  if (!moduleDir || !executablePath || !outputPath) {
    throw Error("usage: test_resident_browser.cjs MODULE_DIR CHROME_EXECUTABLE NEW_OUTPUT [TILES_MNK] [KERNELS] [ACCUMULATIONS] [SHAPES_MKN] [rank|rank-active-lanes|rank-tournament|rank-matched|rank-pruning-matched|rank-pair-lanes-matched|rank-prefix-matched|rank-count-matched|rank-profile|rank-adaptation|matmul|matmul-rank|tensor-mean|nn|nn-training|nd-tensor|nn-clients|nn-clients-cpu|nn-training-clients|nn-training-clients-cpu] [BASELINE_MODULE_DIR]");
  }
if(fixture && !["rank", "rank-active-lanes", "rank-tournament", "rank-matched", "rank-pruning-matched", "rank-pair-lanes-matched", "rank-prefix-matched", "rank-count-matched", "rank-profile", "rank-adaptation", "matmul", "matmul-rank", "tensor-mean", "nn", "nn-training", "nn-graph-training", "nn-graph-training-profile", "nn-graph-forward", "nn-forward-clients", "nn-forward-bench", "nd-tensor", "nn-clients", "nn-clients-cpu", "nn-training-clients", "nn-training-clients-cpu", "nn-graph-clients", "nn-graph-clients-cpu", "nn-fusion-clients", "nn-autograd-clients", "nn-learner-clients", "pointwise-clients", "nn-module-handoff"].includes(fixture)) throw Error("unknown fixture");
  const nnClientFixture = fixture === "nn-clients" || fixture === "nn-clients-cpu";
  const trainingClientFixture = fixture === "nn-training-clients" || fixture === "nn-training-clients-cpu";
  const graphClientFixture = fixture === "nn-graph-clients" || fixture === "nn-graph-clients-cpu";
  const matched=fixture === "rank-matched" || fixture === "rank-pruning-matched" || fixture === "rank-pair-lanes-matched" || fixture === "rank-prefix-matched" || fixture === "rank-count-matched";
  if(matched !== Boolean(baselineDir)) throw Error("matched rank fixtures require BASELINE_MODULE_DIR; other fixtures must omit it");
  const rankFixture = fixture === "rank" || fixture === "rank-active-lanes" || fixture === "rank-tournament";
  const fd = fs.openSync(outputPath, "wx");
  let report, browser, server, page;
  let profileFd, profileRows=0, profileBytes=0;
  const profileHash=crypto.createHash("sha256");
  let metadata = {}, pageErrors = [], consoleMessages = [];
  try {
    const files = new Map([
      ["/", [path.join(__dirname,"../bindings/st-wasm/tests/", nnClientFixture ? "resident_nn_clients.html" : fixture === "nn" ? "resident_nn.html" : fixture === "tensor-mean" ? "tensor_mean.html" : fixture === "rank-profile" ? "resident_rank_profile_webgpu.html" : matched ? "resident_rank_matched_webgpu.html" : fixture === "rank-adaptation" ? "resident_rank_adaptation_webgpu.html" : fixture === "matmul-rank" ? "resident_matmul_rank_webgpu.html" : rankFixture ? "resident_rank_webgpu.html" : "resident_webgpu.html"), "text/html"]],
      ["/module/spiraltorch_wasm.js", [path.join(moduleDir,"spiraltorch_wasm.js"), "text/javascript"]],
      ["/module/spiraltorch_wasm_bg.wasm", [path.join(moduleDir,"spiraltorch_wasm_bg.wasm"), "application/wasm"]],
    ]);
    const moduleRoot = path.resolve(moduleDir);
    if(fixture === "nn-training") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_training.html"), "text/html"]);
    if(fixture === "nn-graph-training") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_training.html"), "text/html"]);
    if(fixture === "nn-fusion-clients") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_pointwise_fusion_clients.html"), "text/html"]);
    if(fixture === "nn-autograd-clients") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_autograd_clients.html"), "text/html"]);
    if(fixture === "nn-learner-clients") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_learner_clients.html"), "text/html"]);
    if(fixture === "pointwise-clients") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_pointwise_clients.html"), "text/html"]);
    if(fixture === "nn-module-handoff") {
      files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_module_handoff.html"), "text/html"]);
      files.set("/fixture.json", [path.join(moduleRoot, "handoff-fixture.json"), "application/json"]);
    }
    if(fixture === "nn-graph-training-profile") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_training_profile.html"), "text/html"]);
    if(fixture === "nn-graph-forward") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_forward.html"), "text/html"]);
    if(fixture === "nn-forward-clients") {
      files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_forward_clients.html"), "text/html"]);
      files.set("/fixture.json", [path.join(moduleRoot, "forward-fixture.json"), "application/json"]);
    }
    if(fixture === "nn-forward-bench") {
      files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_forward_bench.html"), "text/html"]);
      files.set("/fixture.json", [path.join(moduleRoot, "forward-bench-fixture.json"), "application/json"]);
    }
    if(fixture === "nd-tensor") files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_nd_tensor.html"), "text/html"]);
    if(nnClientFixture) files.set("/fixture.json", [path.join(moduleRoot,"nn-fixture.json"), "application/json"]);
    if(trainingClientFixture) {
      files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_training_clients.html"), "text/html"]);
      files.set("/fixture.json", [path.join(moduleRoot, "training-fixture.json"), "application/json"]);
    }
    if(graphClientFixture) {
      files.set("/", [path.join(__dirname, "../bindings/st-wasm/tests/resident_graph_clients.html"), "text/html"]);
      files.set("/fixture.json", [path.join(moduleRoot, "graph-fixture.json"), "application/json"]);
    }
    function addGeneratedAssets(dir, root=moduleRoot, prefix="/module/") {
      for(const entry of fs.readdirSync(dir,{withFileTypes:true})) {
        const file=path.join(dir,entry.name);
        if(entry.isDirectory()) addGeneratedAssets(file,root,prefix);
        else if(entry.isFile() && /\.(js|wasm)$/.test(entry.name)) {
          files.set(prefix+path.relative(root,file).split(path.sep).join("/"),
                    [file,entry.name.endsWith(".wasm") ? "application/wasm" : "text/javascript"]);
        }
      }
    }
    addGeneratedAssets(moduleRoot);
    if(baselineDir) {
      const baselineRoot=path.resolve(baselineDir);
      addGeneratedAssets(baselineRoot,baselineRoot,"/baseline/");
      for(const asset of ["spiraltorch_wasm.js","spiraltorch_wasm_bg.wasm"]) {
        if(!files.has("/baseline/"+asset)) throw Error("missing baseline asset "+asset);
      }
    }
    const wasm = fs.readFileSync(files.get("/module/spiraltorch_wasm_bg.wasm")[0]);
    metadata = {
      wasm_sha256: crypto.createHash("sha256").update(wasm).digest("hex"),
      page_sha256: crypto.createHash("sha256").update(fs.readFileSync(files.get("/")[0])).digest("hex"),
      launch_flags: ["--enable-unsafe-webgpu"],
      tiles_mnk_request: tiles ?? null,
      kernels_request: kernels ?? null,
      accumulations_request: accumulations ?? null,
      shapes_request: shapes ?? null,
      fixture_request: fixture ?? "matmul",
      asset_sha256: Object.fromEntries([...files].map(([url,[file]])=>[
        url,crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"),
      ])),
    };
    server = http.createServer((req,res)=>{
      const file=files.get(new URL(req.url,"http://127.0.0.1").pathname);
      if(!file) { res.writeHead(404); res.end(); return; }
      res.setHeader("Content-Type",file[1]); res.end(fs.readFileSync(file[0]));
    });
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
    browser = await chromium.launch({executablePath,headless:true,args:["--enable-unsafe-webgpu"]});
    metadata.browser_version = browser.version();
    page = await browser.newPage();
    if(fixture === "nn-graph-training-profile") {
      profileFd=fs.openSync(outputPath+".cases.jsonl","wx");
      await page.exposeFunction("publishResidentProfile",raw=>{
        if(typeof raw !== "string" || Buffer.byteLength(raw)>64*1024*1024)
          throw Error("profile state capture exceeds its per-case budget");
        const data=Buffer.from(raw+"\n");
        fs.writeFileSync(profileFd,data);
        profileHash.update(data); profileBytes+=data.length; profileRows++;
        return {line:profileRows,sha256:crypto.createHash("sha256").update(raw).digest("hex")};
      });
    }
    page.on("console", message=>{ if(consoleMessages.length<100) consoleMessages.push({type:message.type(),text:message.text()}); });
    let rejectPageError;
    const fatal = new Promise((_,reject)=>{ rejectPageError=reject; });
    fatal.catch(()=>{});
    page.on("pageerror", error=>{ pageErrors.push(String(error)); rejectPageError(error); });
    page.on("crash",()=>{ const error=Error("owned test browser page crashed");pageErrors.push(String(error));rejectPageError(error); });
    page.on("response", response=>{
      if(response.status() >= 400 && /^\/(module|baseline)\//.test(new URL(response.url()).pathname)) {
        const error = Error("generated module asset failed: "+response.status()+" "+response.url());
        pageErrors.push(String(error)); rejectPageError(error);
      }
    });
    const params = new URLSearchParams();
    if(fixture === "nn-clients-cpu" || fixture === "nn-training-clients-cpu" || fixture === "nn-graph-clients-cpu") params.set("cpu_only","1");
    if(tiles) params.set("tiles",tiles);
    if(kernels) params.set("kernels",kernels);
    if(accumulations) params.set("accumulations",accumulations);
    if(shapes) params.set("shapes",shapes);
    if(fixture === "rank-active-lanes") params.set("suite","active-lanes");
    if(fixture === "rank-tournament") params.set("suite","tournament");
    if(fixture === "rank-pruning-matched") params.set("suite","pruning");
    if(fixture === "rank-pair-lanes-matched") params.set("suite","pair-lanes");
    if(fixture === "rank-prefix-matched") params.set("suite","prefix");
    if(fixture === "rank-count-matched") params.set("suite","count-bounds");
    const query = "?"+params.toString();
    await page.goto(`http://127.0.0.1:${server.address().port}/${query}`);
    await Promise.race([fatal, page.locator("#result:not([data-status='running'])").waitFor({timeout:fixture === "rank-prefix-matched" || fixture === "rank-count-matched" ? 600000 : 300000})]);
    report = fixture === "nn-graph-training-profile"
      ? await page.evaluate(()=>window.residentProfileReport)
      : JSON.parse(await page.locator("#result").textContent());
    if(fixture === "nn-graph-training-profile" && (!report || report.cases?.length !== profileRows || profileRows !== 21))
      throw Error("incomplete streamed profiling matrix");
    if(pageErrors.length) report.status="error";
    if((fixture === "rank-profile" || fixture === "nn-graph-training-profile") && consoleMessages.some(m => /Invalid QuerySet|Invalid CommandBuffer|Cannot allocate sample buffer/.test(m.text))) {
      report.status="error";
      report.error="uncaptured WebGPU timestamp validation/allocation failure";
    }
  } catch(error) {
    report = {status:"error",error:String(error.stack||error)};
    if(page) report.last_page_result = await page.locator("#result").textContent({timeout:2000}).catch(()=>null);
  } finally {
    if(browser) await browser.close();
    if(server) await new Promise(resolve=>server.close(resolve));
    if(profileFd !== undefined) {
      fs.closeSync(profileFd);
      metadata.profile_state_artifacts={path:path.basename(outputPath)+".cases.jsonl",
        rows:profileRows,bytes:profileBytes,sha256:profileHash.digest("hex")};
    }
    Object.assign(report,metadata,{page_errors:pageErrors,console_messages:consoleMessages});
    fs.writeFileSync(fd,JSON.stringify(report,null,2)+"\n");
    fs.closeSync(fd);
  }
  console.log(JSON.stringify({status:report.status,cases:report.cases?.length,error:report.error,output:outputPath}));
  process.exitCode=report.status === "passed" ? 0 : 1;
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
