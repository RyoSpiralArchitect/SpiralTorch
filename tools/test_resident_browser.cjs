#!/usr/bin/env node
// Isolated headless test browser only: no user profile, cookies, or existing tabs.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const crypto = require("node:crypto");
const {chromium} = require("playwright");

async function main() {
  const [moduleDir, executablePath, outputPath, tiles, kernels, accumulations, shapes] = process.argv.slice(2);
  if (!moduleDir || !executablePath || !outputPath) {
    throw Error("usage: test_resident_browser.cjs MODULE_DIR CHROME_EXECUTABLE NEW_OUTPUT [TILES_MNK] [KERNELS] [ACCUMULATIONS] [SHAPES_MKN]");
  }
  const fd = fs.openSync(outputPath, "wx");
  let report, browser, server, page;
  let metadata = {}, pageErrors = [];
  try {
    const files = new Map([
      ["/", [path.join(__dirname,"../bindings/st-wasm/tests/resident_webgpu.html"), "text/html"]],
      ["/module/spiraltorch_wasm.js", [path.join(moduleDir,"spiraltorch_wasm.js"), "text/javascript"]],
      ["/module/spiraltorch_wasm_bg.wasm", [path.join(moduleDir,"spiraltorch_wasm_bg.wasm"), "application/wasm"]],
    ]);
    const moduleRoot = path.resolve(moduleDir);
    function addGeneratedAssets(dir) {
      for(const entry of fs.readdirSync(dir,{withFileTypes:true})) {
        const file=path.join(dir,entry.name);
        if(entry.isDirectory()) addGeneratedAssets(file);
        else if(entry.isFile() && /\.(js|wasm)$/.test(entry.name)) {
          files.set("/module/"+path.relative(moduleRoot,file).split(path.sep).join("/"),
                    [file,entry.name.endsWith(".wasm") ? "application/wasm" : "text/javascript"]);
        }
      }
    }
    addGeneratedAssets(moduleRoot);
    const wasm = fs.readFileSync(files.get("/module/spiraltorch_wasm_bg.wasm")[0]);
    metadata = {
      wasm_sha256: crypto.createHash("sha256").update(wasm).digest("hex"),
      page_sha256: crypto.createHash("sha256").update(fs.readFileSync(files.get("/")[0])).digest("hex"),
      launch_flags: ["--enable-unsafe-webgpu"],
      tiles_mnk_request: tiles ?? null,
      kernels_request: kernels ?? null,
      accumulations_request: accumulations ?? null,
      shapes_request: shapes ?? null,
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
    let rejectPageError;
    const fatal = new Promise((_,reject)=>{ rejectPageError=reject; });
    fatal.catch(()=>{});
    page.on("pageerror", error=>{ pageErrors.push(String(error)); rejectPageError(error); });
    page.on("response", response=>{
      if(response.status() >= 400 && new URL(response.url()).pathname.startsWith("/module/")) {
        const error = Error("generated module asset failed: "+response.status()+" "+response.url());
        pageErrors.push(String(error)); rejectPageError(error);
      }
    });
    const params = new URLSearchParams();
    if(tiles) params.set("tiles",tiles);
    if(kernels) params.set("kernels",kernels);
    if(accumulations) params.set("accumulations",accumulations);
    if(shapes) params.set("shapes",shapes);
    const query = "?"+params.toString();
    await page.goto(`http://127.0.0.1:${server.address().port}/${query}`);
    await Promise.race([fatal, page.locator("#result:not([data-status='running'])").waitFor({timeout:300000})]);
    report = JSON.parse(await page.locator("#result").textContent());
    if(pageErrors.length) report.status="error";
  } catch(error) {
    report = {status:"error",error:String(error.stack||error)};
    if(page) report.last_page_result = await page.locator("#result").textContent({timeout:2000}).catch(()=>null);
  } finally {
    if(browser) await browser.close();
    if(server) await new Promise(resolve=>server.close(resolve));
    Object.assign(report,metadata,{page_errors:pageErrors});
    fs.writeFileSync(fd,JSON.stringify(report,null,2)+"\n");
    fs.closeSync(fd);
  }
  console.log(JSON.stringify({status:report.status,cases:report.cases?.length,error:report.error,output:outputPath}));
  process.exitCode=report.status === "passed" ? 0 : 1;
}
main().catch(error=>{ console.error(error); process.exitCode=1; });
