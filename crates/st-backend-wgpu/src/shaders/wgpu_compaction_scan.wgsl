// Scan Pass: compute exclusive prefix sums of flags over row tiles.
// Writes per-tile counts to a side buffer for a second pass to compute block offsets.
struct InV { data: array<f32>; }
struct Param { rows:u32, cols:u32, low:f32, high:f32, tiles_per_row:u32 }
struct OutFlags { data: array<u32>; }     // flags per element (0/1), same shape as input (optional)
struct OutTileCnt { data: array<u32>; }   // per tile count (rows * tiles_per_row)
@group(0) @binding(0) var<storage, read> vin  : InV;
@group(0) @binding(1) var<storage, read_write> flags: OutFlags;
@group(0) @binding(2) var<storage, read_write> tilecnt: OutTileCnt;
@group(0) @binding(3) var<uniform> P : Param;

var<workgroup> sflags: array<u32, 256>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>)
{
  let rows = P.rows;
  let cols = P.cols;
  let low  = P.low;
  let high = P.high;
  let tpr  = P.tiles_per_row; // tiles per row
  let r = wid.x / tpr;
  let tile = wid.x % tpr;
  if (r >= rows) { return; }
  let tile_start = tile * 256u;
  let tid = lid.x;
  let g = r*cols + tile_start + tid;

  var f = 0u;
  if (tile_start + tid < cols) {
    let v = vin.data[g];
    if (v >= low && v <= high) { f = 1u; }
  }
  sflags[tid] = f;
  workgroupBarrier();

  // Exclusive scan in shared (Blelloch)
  var offset:u32 = 1u;
  for (var d=256u>>1; d>0u; d>>=1u) {
    if (tid < d) {
      let ai = offset*(2u*tid+1u)-1u;
      let bi = offset*(2u*tid+2u)-1u;
      sflags[bi] = sflags[bi] + sflags[ai];
    }
    offset <<= 1u;
    workgroupBarrier();
  }
  if (tid==0u) { sflags[255u] = 0u; }
  workgroupBarrier();
  for (var d=1u; d<256u; d<<=1u) {
    offset >>= 1u;
    if (tid < d) {
      let ai = offset*(2u*tid+1u)-1u;
      let bi = offset*(2u*tid+2u)-1u;
      let t = sflags[ai];
      sflags[ai] = sflags[bi];
      sflags[bi] = sflags[bi] + t;
    }
    workgroupBarrier();
  }

  if (tile_start + tid < cols) {
    flags.data[g] = sflags[tid]; // optional; used by apply
  }

  // tile count into tilecnt[r * tpr + tile] (last position + last flag)
  if (tid == 255u) {
    let last = sflags[255u] + f;
    tilecnt.data[r * tpr + tile] = last;
  }
}
