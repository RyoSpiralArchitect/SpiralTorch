// Apply pass: scatter kept elements to compacted output using positions from scan_pass.
struct InV { data: array<f32>, }
struct InI { data: array<i32>, }
struct Pos { data: array<u32>, }
struct OutV { data: array<f32>, }
struct OutI { data: array<i32>, }
struct Param { rows:u32, cols:u32, low:f32, high:f32, }

@group(0) @binding(0) var<storage, read>   vin : InV;
@group(0) @binding(1) var<storage, read>   iin : InI;
@group(0) @binding(2) var<storage, read>   pos : Pos;
@group(0) @binding(3) var<storage, read_write> vout: OutV;
@group(0) @binding(4) var<storage, read_write> iout: OutI;
@group(0) @binding(5) var<uniform> P : Param;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id)       wid: vec3<u32>)
{
  let rows = P.rows;
  let cols = P.cols;
  let low  = P.low;
  let high = P.high;
  let r = wid.x;
  if (r >= rows) { return; }
  let tid = lid.x;
  let row_off = r * cols;
  var c = tid;
  loop {
    if (c >= cols) { break; }
    let v = vin.data[row_off + c];
    if (v >= low && v <= high) {
      let p = pos.data[row_off + c];
      if (p != 0xffffffffu) {
        vout.data[row_off + p] = v;
        iout.data[row_off + p] = iin.data[row_off + c];
      }
    }
    c += 256u;
  }
}
