// Scan pass: compute per-element exclusive positions for kept elements within [low, high].
// One WG (256) per row; tiles across columns.
struct InV { data: array<f32>; }
struct Pos { data: array<u32>; }
struct Param { rows:u32, cols:u32, low:f32, high:f32, tile:u32 }

@group(0) @binding(0) var<storage, read>   vin : InV;
@group(0) @binding(1) var<storage, read_write> pos: Pos;
@group(0) @binding(2) var<uniform> P : Param;

var<workgroup> flags : array<u32, 256>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id)       wid: vec3<u32>)
{
  let rows  = P.rows;
  let cols  = P.cols;
  let low   = P.low;
  let high  = P.high;
  let TILE  = max(P.tile, 256u);
  let r = wid.x;
  if (r >= rows) { return; }
  let tid = lid.x;
  let row_off = r * cols;

  var base:u32 = 0u; // running base across tiles (exclusive)
  var c0 = 0u;
  loop {
    if (c0 >= cols) { break; }
    let cend = min(c0+TILE, cols);
    // 1) flags for this tile
    var keep:u32 = 0u;
    var c = c0 + tid;
    if (c < cend) {
      let v = vin.data[row_off + c];
      if (v >= low && v <= high) { keep = 1u; }
    }
    flags[tid] = keep;
    workgroupBarrier();

    // 2) in-tile exclusive scan (Blelloch)
    var offset:u32 = 1u;
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
    if (tid == 0u) { flags[255u] = 0u; }
    workgroupBarrier();
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

    // 3) write positions for kept elements in this tile
    if (c < cend) {
      if (keep == 1u) {
        pos.data[row_off + c] = base + flags[tid];
      } else {
        pos.data[row_off + c] = 0xffffffffu; // sentinel for non-keep
      }
    }
    // 4) advance base by number of kept in tile (flags[255] now holds inclusive sum; fix up)
    if (tid == 255u) {
      // inclusive sum at last index + 1 (since last was shifted to exclusive root)
      // Here we compute tile_kept as previous flags[255] + original keep of tid==255.
      // For simplicity, recompute last keep:
      let last_keep = select(0u, 1u,
        (cend>c0) && ((cend-1u) >= c0) &&
        (vin.data[row_off + (cend-1u)] >= low && vin.data[row_off + (cend-1u)] <= high));
      let tile_kept = flags[255u] + last_keep;
      // broadcast via shared slot 255
      flags[255u] = tile_kept;
    }
    workgroupBarrier();
    base = base + flags[255u];
    c0 = c0 + TILE;
    workgroupBarrier();
  }
}
