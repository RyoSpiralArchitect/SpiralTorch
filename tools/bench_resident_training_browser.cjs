#!/usr/bin/env node
// Separate headless profile; never attaches to the user's browser or GPU tasks.
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto"),http=require("node:http");
const {chromium}=require("playwright");
function digest(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
async function main() {
  const [baseline,candidate,chrome,output]=process.argv.slice(2);
  if(!baseline||!candidate||!chrome||!output) throw Error("usage: BASELINE_MODULE CANDIDATE_MODULE CHROME NEW_OUTPUT");
  const fd=fs.openSync(output,"wx");
  let server,browser,report={status:"error"},metadata={},errors=[],consoleMessages=[];
  try {
    const files=new Map([["/",[path.join(__dirname,"../bindings/st-wasm/tests/resident_training_bench.html"),"text/html"]]]);
    function assets(dir,root,prefix) {
      for(const entry of fs.readdirSync(dir,{withFileTypes:true})) {
        const file=path.join(dir,entry.name);
        if(entry.isDirectory()) assets(file,root,prefix);
        else if(entry.isFile()&&/\.(js|wasm)$/.test(file)) files.set(prefix+path.relative(root,file).split(path.sep).join("/"),[file,file.endsWith("wasm")?"application/wasm":"text/javascript"]);
      }
    }
    for(const [dir,prefix] of [[baseline,"/baseline/"],[candidate,"/candidate/"]]) {
      const root=path.resolve(dir); assets(root,root,prefix);
      for(const name of ["spiraltorch_wasm.js","spiraltorch_wasm_bg.wasm"])
        if(!files.has(prefix+name)) throw Error("missing generated module asset");
    }
    metadata.asset_sha256=Object.fromEntries([...files].map(([url,[file]])=>[url,digest(file)]));
    server=http.createServer((request,response)=>{
      const file=files.get(new URL(request.url,"http://127.0.0.1").pathname);
      if(!file) {response.writeHead(404);response.end();return;}
      response.setHeader("Content-Type",file[1]);response.end(fs.readFileSync(file[0]));
    });
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
    metadata.launch_flags=["--enable-unsafe-webgpu"];
    browser=await chromium.launch({executablePath:chrome,headless:true,args:metadata.launch_flags});
    metadata.browser_version=browser.version();
    const page=await browser.newPage();
    let fail;
    const fatal=new Promise((_,reject)=>{fail=reject;});fatal.catch(()=>{});
    page.on("pageerror",error=>{errors.push(String(error));fail(error);});
    page.on("console",message=>{if(consoleMessages.length<100)consoleMessages.push({type:message.type(),text:message.text()});});
    await page.goto("http://127.0.0.1:"+server.address().port);
    await Promise.race([fatal,page.locator("#result:not([data-status='running'])").waitFor({timeout:600000})]);
    report=JSON.parse(await page.locator("#result").textContent());
    for(const [url,[file]] of files) if(digest(file)!==metadata.asset_sha256[url]) throw Error("served asset changed");
    if(errors.length) report.status="error";
  } catch(error) { report.status="error";report.error=String(error.stack||error); }
  finally {
    if(browser) await browser.close();
    if(server) await new Promise(resolve=>server.close(resolve));
    fs.writeFileSync(fd,JSON.stringify({...report,...metadata,page_errors:errors,console_messages:consoleMessages},null,2)+"\n");
    fs.closeSync(fd);
  }
  console.log(JSON.stringify({status:report.status,cases:report.cases?.length,error:report.error}));
  process.exitCode=report.status==="passed"?0:1;
}
main().catch(error=>{console.error(error);process.exitCode=1;});
