// Compact elements that lie within [low, high]; produce (vals, idx) compacted per row.
// For large rows, dispatch twice (scan + apply); here we show a 1CE small-row version.
struct InV { data: array<f32>, }
struct InI { data: array<i32>, }
struct OutV { data: array<f32>, }
struct OutI { data: array<i32>, }
struct Param { rows:u32, cols:u32, low:f32, high:f32, stride:u32, }

@group(0) @binding(0) var<storage, read>   vin  : InV;
@group(0) @binding(1) var<storage, read>   iin  : InI;
@group(0) @binding(2) var<storage, read_write> vout : OutV;
@group(0) @binding(3) var<storage, read_write> iout : OutI;
@group(0) @binding(4) var<uniform> P : Param;

var<workgroup> flags : array<u32, 256>;
var<workgroup> scanp : array<u32, 256>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>)
{
  let rows  = P.rows;
  let cols  = P.cols;
  let low   = P.low;
  let high  = P.high;
  let r = wid.x;
  if (r >= rows) { return; }

  // phase 1: predicate flags for this block-chunk of the row
  let tid = lid.x;
  let stride = 256u;
  let row_off = r * cols;

  // Each block compacts its chunk; multi-block rows need multi-CE or a second pass (not shown).
  // Here assume cols <= 256 for demo (or you can tile).
  var f:u32 = 0u;
  if (tid < cols) {
    let v = vin.data[row_off + tid];
    if (v >= low && v <= high) { f = 1u; }
  }
  flags[tid] = f;
  workgroupBarrier();

  // Blelloch scan (exclusive)
  var offset:u32 = 1u;
  // up-sweep
  var d:u32 = 256u;
  loop {
    if (d <= 1u) { break; }
    d = d >> 1u;
    if (tid < d) {
      let ai = offset*(2u*tid+1u) - 1u;
      let bi = offset*(2u*tid+2u) - 1u;
      flags[bi] = flags[bi] + flags[ai];
    }
    offset = offset << 1u;
    workgroupBarrier();
  }
  if (tid == 0u) { flags[256u-1u] = 0u; }
  workgroupBarrier();
  // down-sweep
  var d2:u32 = 1u;
  loop {
    if (d2 >= 256u) { break; }
    offset = offset >> 1u;
    if (tid < d2) {
      let ai = offset*(2u*tid+1u) - 1u;
      let bi = offset*(2u*tid+2u) - 1u;
      let t = flags[ai];
      flags[ai] = flags[bi];
      flags[bi] = flags[bi] + t;
    }
    d2 = d2 << 1u;
    workgroupBarrier();
  }

  // phase 2: scatter
  if (tid < cols) {
    let v = vin.data[row_off + tid];
    let keep = (v >= low && v <= high);
    if (keep) {
      let pos = flags[tid];
      vout.data[row_off + pos] = v;
      iout.data[row_off + pos] = iin.data[row_off + tid];
    }
  }
}
