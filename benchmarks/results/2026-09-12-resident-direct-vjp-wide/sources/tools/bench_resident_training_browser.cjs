#!/usr/bin/env node
// Separate headless profile; never attaches to the user's browser or GPU tasks.
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto"),http=require("node:http");
const {chromium}=require("playwright");
function digest(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
async function main() {
  const [baseline,candidate,chrome,output,workload="dense",matrix="standard",optimization="none",optimizer="none"]=process.argv.slice(2);
  if(!baseline||!candidate||!chrome||!output||!["dense","graph","learner"].includes(workload)||
     !["standard","wide"].includes(matrix)||(matrix==="wide"&&workload==="dense")||
     !["none","fuse-pointwise","fuse-learner-seeds"].includes(optimization)||(optimization==="fuse-pointwise"&&workload!=="graph")||
     (optimization==="fuse-learner-seeds"&&workload!=="learner")||
     !["none","topos_ema","clipped_topos_ema"].includes(optimizer)||(optimizer!=="none"&&workload!=="learner"))
    throw Error("usage: BASELINE_MODULE CANDIDATE_MODULE CHROME NEW_OUTPUT [dense|graph|learner] [standard|wide] [none|fuse-pointwise|fuse-learner-seeds] [none|topos_ema|clipped_topos_ema]");
  const fd=fs.openSync(output,"wx");
  let server,browser,page,progressFd,casesFd,report={status:"error"},metadata={},errors=[],consoleMessages=[];
  const cases=[];
  try {
    progressFd=fs.openSync(output+".progress.jsonl","wx");
    casesFd=fs.openSync(output+".cases.jsonl","wx");
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
    page=await browser.newPage();
    await page.exposeFunction("recordTrainingProgress",value=>{
      metadata.last_progress=value;
      fs.writeSync(progressFd,JSON.stringify(value)+"\n");
      if(value.stage==="create_workspaces"||value.stage==="case_finished") console.log(JSON.stringify(value));
    });
    await page.exposeFunction("recordTrainingCase",value=>{
      const row=JSON.parse(value);
      if(cases.length>=9||!row.config||row.samples?.length!==20) throw Error("invalid completed case");
      fs.writeSync(casesFd,value+"\n");
      cases.push(row);
    });
    let fail;
    const fatal=new Promise((_,reject)=>{fail=reject;});fatal.catch(()=>{});
    page.on("pageerror",error=>{errors.push(String(error));fail(error);});
    page.on("console",message=>{if(consoleMessages.length<100)consoleMessages.push({type:message.type(),text:message.text()});});
    await page.goto("http://127.0.0.1:"+server.address().port+"/?workload="+workload+"&matrix="+matrix+"&optimization="+optimization+"&optimizer="+optimizer);
    await Promise.race([fatal,page.locator("#result:not([data-status='running'])").waitFor({timeout:600000})]);
    report=JSON.parse(await page.locator("#result").textContent());
    if(report.status==="passed"&&(report.cases?.length!==9||cases.length!==9)) throw Error("missing completed cases");
    for(const [url,[file]] of files) if(digest(file)!==metadata.asset_sha256[url]) throw Error("served asset changed");
    if(errors.length) report.status="error";
  } catch(error) {
    report.status="error";report.error=String(error.stack||error);
    if(page) metadata.last_page_result=await page.locator("#result").textContent({timeout:2000}).catch(()=>null);
  }
  finally {
    if(browser) await browser.close();
    if(server) await new Promise(resolve=>server.close(resolve));
    fs.writeFileSync(fd,JSON.stringify({...report,cases,...metadata,page_errors:errors,console_messages:consoleMessages})+"\n");
    fs.closeSync(fd);
    if(progressFd!==undefined) fs.closeSync(progressFd);
    if(casesFd!==undefined) fs.closeSync(casesFd);
  }
  console.log(JSON.stringify({status:report.status,cases:report.cases?.length,error:report.error}));
  process.exitCode=report.status==="passed"?0:1;
}
main().catch(error=>{console.error(error);process.exitCode=1;});
