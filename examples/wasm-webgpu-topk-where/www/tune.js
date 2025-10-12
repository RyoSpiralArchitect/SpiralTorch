async function initDevice() {
  if (!('gpu' in navigator)) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  return {adapter, device};
}
// WGSL portable kernel (1CE); tuner still valid
const WGSL_TOPK_1CE = `
struct Meta { rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32, cand_cols:u32 };
@group(0) @binding(0) var<storage, read>  X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUTV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUTI: array<i32>;
@group(0) @binding(3) var<uniform> meta: Meta;
var<workgroup> cand_vals: array<f32, 256u*32u>;
var<workgroup> cand_idxs: array<i32, 256u*32u>;
fn row_base(r:u32, cols:u32) -> u32 { return r * cols; }
fn scan_and_keep(tid:u32, stride:u32, base:u32, start_col:u32, end_col:u32, k_lane:u32, inout_vals: ptr<function, array<f32,32>>, inout_idxs: ptr<function, array<i32,32>>) {
  var c:u32 = start_col + tid;
  while (c < end_col) {
    let v = X[base + c];
    var minv:f32 = (*inout_vals)[0];
    var minp:u32 = 0u;
    for (var j:u32=1u; j<k_lane; j=j+1u) {
      if ((*inout_vals)[j] < minv) { minv = (*inout_vals)[j]; minp = j; }
    }
    if (v > minv) {
      (*inout_vals)[minp] = v;
      (*inout_idxs)[minp] = i32(c);
    }
    c = c + stride;
  }
}
fn bitonic_desc(total:u32, tid:u32) {
  var size:u32 = 2u;
  while (size <= total) {
    var stride2:u32 = size / 2u;
    while (stride2 > 0u) {
      var i:u32 = tid;
      while (i < total) {
        let j = i ^ stride2;
        if (j > i) {
          let up = ((i & size) == 0u);
          let vi = cand_vals[i];
          let vj = cand_vals[j];
          if ((up && vi < vj) || (!up && vi > vj)) {
            let t = cand_vals[i]; cand_vals[i] = cand_vals[j]; cand_vals[j] = t;
            let ti = cand_idxs[i]; cand_idxs[i] = cand_idxs[j]; cand_idxs[j] = ti;
          }
        }
        i = i + 256u;
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    size = size * 2u;
  }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rows = meta.rows, cols = meta.cols, k = meta.k, k_lane = meta.k_lane;
  let row  = wid.x; let tid  = lid.x;
  if (row >= rows) { return; }
  let base  = row * cols;
  var local_vals: array<f32, 32>; var local_idxs: array<i32, 32>;
  for (var i:u32=0u;i<k_lane;i=i+1u){ local_vals[i] = -1.0/0.0; local_idxs[i] = -1; }
  let chunk = select(cols, meta.chunk_cols, meta.chunk_cols!=0u);
  var pos:u32 = 0u;
  loop {
    let start_c = pos * chunk;
    if (start_c >= cols) { break; }
    let end_c   = min(start_c + chunk, cols);
    scan_and_keep(tid, 256u, base, start_c, end_c, k_lane, &local_vals, &local_idxs);
    pos = pos + 1u;
    if (meta.chunk_cols==0u) { break; }
  }
  let offset = tid*k_lane;
  for (var i:u32=0u;i<k_lane;i=i+1u){ cand_vals[offset+i]=local_vals[i]; cand_idxs[offset+i]=local_idxs[i]; }
  workgroupBarrier();
  bitonic_desc(256u*k_lane, tid);
  if (tid < k){
    OUTV[row*k + tid] = cand_vals[tid];
    OUTI[row*k + tid] = cand_idxs[tid];
  }
}`;

async function runOnce(device, rows, cols, k, k_lane, chunk_cols){
  const n = rows*cols;
  const xs = new Float32Array(n);
  for (let i=0;i<n;i++) xs[i] = Math.sin(i*0.001) + Math.random()*1e-6;
  const outv = new Float32Array(rows*k);
  const outi = new Int32Array(rows*k);
  const bX = device.createBuffer({ size: xs.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
  const bV = device.createBuffer({ size: outv.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
  const bI = device.createBuffer({ size: outi.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
  const meta = new Uint32Array([rows, cols, k, k_lane, chunk_cols, 256*k_lane]);
  const bM = device.createBuffer({ size: meta.byteLength, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bX, 0, xs);
  device.queue.writeBuffer(bM, 0, meta);

  const mod = device.createShaderModule({code:WGSL_TOPK_1CE});
  const pl = device.createComputePipeline({layout:'auto', compute:{module:mod, entryPoint:'main'}});
  const bind = device.createBindGroup({layout:pl.getBindGroupLayout(0), entries:[
    {binding:0, resource:{buffer:bX}}, {binding:1, resource:{buffer:bV}}, {binding:2, resource:{buffer:bI}}, {binding:3, resource:{buffer:bM}},
  ]});

  const e = device.createCommandEncoder();
  const p = e.beginComputePass();
  p.setPipeline(pl); p.setBindGroup(0, bind);
  p.dispatchWorkgroups(rows);
  p.end();
  const t0 = performance.now();
  device.queue.submit([e.finish()]);
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();
  return t1 - t0;
}

function parseCsvInts(id){ return document.getElementById(id).value.split(',').map(s=>parseInt(s.trim(),10)); }

function kmeans(points, k, iters=25){
  const n = points.length; if (n==0) return [];
  let centers = [];
  for (let i=0;i<k;i++) centers.push({...points[Math.floor(Math.random()*n)]});
  for (let it=0; it<iters; it++){
    const buckets = Array.from({length:k}, ()=>[]);
    for (const p of points){
      let best=0, bd=1e100;
      for (let i=0;i<k;i++){
        const c = centers[i];
        const dx=p.x-c.x, dy=p.y-c.y, dz=p.z-c.z;
        const d=dx*dx+dy*dy+dz*dz;
        if (d<bd){bd=d;best=i;}
      }
      buckets[best].push(p);
    }
    for (let i=0;i<k;i++){
      const b=buckets[i]; if (b.length==0) continue;
      const sx=b.reduce((a,p)=>a+p.x,0), sy=b.reduce((a,p)=>a+p.y,0), sz=b.reduce((a,p)=>a+p.z,0);
      const skl=b.reduce((a,p)=>a+p.kl,0), swg=b.reduce((a,p)=>a+p.wg,0), sch=b.reduce((a,p)=>a+p.ch,0);
      const sms=b.reduce((a,p)=>a+p.ms,0);
      centers[i]={x:sx/b.length, y:sy/b.length, z:sz/b.length, kl:Math.round(skl/b.length), wg:Math.round(swg/b.length), ch:Math.round(sch/b.length), ms:sms/b.length};
    }
  }
  return centers;
}

function synthHeuristicsTable(centers){
  const lines=[];
  lines.push("pub fn choose(rows:u32, cols:u32, k:u32, _subgroup: bool) -> Option<(bool,u32,u32,u32)> {");
  lines.push("  // generated by WASM tuner (k-means nearest-center table)");
  lines.push("  let lr = (rows as f64).log2(); let lc = (cols as f64).log2(); let lk = (k as f64).log2();");
  lines.push("  let mut bestd = f64::INFINITY; let mut best: Option<(bool,u32,u32,u32)> = None;");
  for (const c of centers){
    lines.push(`  { let dx=lr-${c.z}; let dy=lc-${c.x}; let dz=lk-${c.y}; let d=dx*dx+dy*dy+dz*dz;`);
    lines.push(`    if d < bestd { bestd=d; best=Some(( ${c.ch>0}, ${c.wg}u32, ${c.kl}u32, ${c.ch}u32 )); } }`);
  }
  lines.push("  best");
  lines.push("}");
  return lines.join("\n");
}

async function scan(){
  const out = document.getElementById('out');
  out.textContent = "Preparing...\n";
  const {device, adapter} = await initDevice();
  const rowsArr=parseCsvInts('rows'), colsArr=parseCsvInts('cols'), kArr=parseCsvInts('ks');
  const configs=[];
  for (const rows of rowsArr) for (const cols of colsArr) for (const k of kArr){
    for (const k_lane of [8,16,32]) for (const chunk_cols of [0,8192]){
      configs.push({rows, cols, k, k_lane, chunk_cols});
    }
  }
  const results=[];
  let i=0;
  for (const c of configs){
    const ms = await runOnce(device, c.rows, c.cols, c.k, c.k_lane, c.chunk_cols);
    results.push({x:Math.log2(c.cols), y:Math.log2(c.k), z:Math.log2(c.rows), ms, kl:c.k_lane, wg:(c.cols<4096?128:256), ch:c.chunk_cols});
    out.textContent = `Run ${++i}/${configs.length}: rows=${c.rows} cols=${c.cols} k=${c.k} k_lane=${c.k_lane} -> ${ms.toFixed(3)} ms`;
  }
  window.__spiral_scan = results;
  out.textContent = `Done. ${results.length} configs measured.`;
}

function exportCenters(centers){
  const rs = synthHeuristicsTable(centers);
  const blob = new Blob([rs], {type:'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'wgpu_heuristics.rs';
  a.click();
}

document.getElementById('run').onclick = scan;
document.getElementById('cluster').onclick = () => {
  const out = document.getElementById('out');
  const meas = window.__spiral_scan || [];
  if (!meas.length) { out.textContent = "No measurements. Run Grid Scan first."; return; }
  const k = Math.min(8, Math.max(2, Math.floor(Math.log2(meas.length)))); // heuristic cluster count
  const centers = kmeans(meas, k);
  window.__spiral_centers = centers;
  out.textContent = "Centers:\n" + centers.map((c,i)=>`#${i} lr=${c.z.toFixed(2)} lc=${c.x.toFixed(2)} lk=${c.y.toFixed(2)} | wg=${c.wg} kl=${c.kl} ch=${c.ch} ms~${c.ms.toFixed(3)}`).join("\n");
};
document.getElementById('export').onclick = () => {
  const centers = window.__spiral_centers || [];
  if (!centers.length) { alert("Run k-means first"); return; }
  exportCenters(centers);
};
