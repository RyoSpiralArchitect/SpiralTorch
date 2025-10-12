// Apply Pass: computes row block offsets from tilecnt, then scatters kept items.
struct InV { data: array<f32>; }
struct InI { data: array<i32>; }
struct Param { rows:u32, cols:u32, low:f32, high:f32, tiles_per_row:u32 }
struct Flags { data: array<u32>; }
struct TileCnt { data: array<u32>; }
struct OutV { data: array<f32>; }
struct OutI { data: array<i32>; }
@group(0) @binding(0) var<storage, read>   vin   : InV;
@group(0) @binding(1) var<storage, read>   iin   : InI;
@group(0) @binding(2) var<storage, read>   flags : Flags;
@group(0) @binding(3) var<storage, read>   tilecnt : TileCnt;
@group(0) @binding(4) var<storage, read_write> vout : OutV;
@group(0) @binding(5) var<storage, read_write> iout : OutI;
@group(0) @binding(6) var<uniform> P : Param;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>)
{
  let rows = P.rows;
  let cols = P.cols;
  let low  = P.low;
  let high = P.high;
  let tpr  = P.tiles_per_row;

  let r = wid.x / tpr;
  let tile = wid.x % tpr;
  if (r >= rows) { return; }
  let tid = lid.x;
  let tile_start = tile * 256u;

  // compute row offset by prefix of tilecnt
  var row_off:u32 = 0u;
  for (var t=0u; t<tile; t++) {
    row_off += tilecnt.data[r*tpr + t];
  }

  if (tile_start + tid < cols) {
    let g = r*cols + tile_start + tid;
    let v = vin.data[g];
    let keep = (v >= low && v <= high);
    if (keep) {
      let pos = flags.data[g] + row_off;
      vout.data[r*cols + pos] = v;
      iout.data[r*cols + pos] = iin.data[g];
    }
  }
}
