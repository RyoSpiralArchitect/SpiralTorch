struct Meta { rows:u32; cols:u32; stride:u32; }
@group(0) @binding(0) var<uniform> U : Meta;
@group(0) @binding(1) var<storage, read>  MASK : array<u32>;
@group(0) @binding(2) var<storage, read>  X : array<f32>;
@group(0) @binding(3) var<storage, read_write> PREFIX : array<u32>;
@group(0) @binding(4) var<storage, read_write> Y : array<f32>;
@group(0) @binding(5) var<storage, read_write> IDX : array<i32>;

fn idx(r:u32,c:u32)->u32{ return r*U.stride + c; }

@compute @workgroup_size(256)
fn exclusive_scan_rows(@builtin(local_invocation_id)  lid: vec3<u32>,
                       @builtin(global_invocation_id) gid: vec3<u32>,
                       @builtin(workgroup_id) wid: vec3<u32>) {
  let r = wid.x;
  if (r>=U.rows) { return; }
  var c = lid.x;
  loop {
    if (c < U.cols) {
      PREFIX[idx(r,c)] = MASK[idx(r,c)];
      c += 256u;
    } else { break; }
  }
  workgroupBarrier();
  var offset:u32 = 1u;
  loop {
    if (offset >= 256u) { break; }
    var c2 = lid.x;
    loop {
      if (c2 < U.cols) {
        let prev = i32(c2) - i32(offset);
        if (prev >= 0) {
          PREFIX[idx(r,c2)] = PREFIX[idx(r,c2)] + PREFIX[idx(r,u32(prev))];
        }
        c2 += 256u;
      } else { break; }
    }
    workgroupBarrier();
    offset = offset * 2u;
  }
  var c3 = lid.x;
  loop {
    if (c3 < U.cols) {
      let inc = PREFIX[idx(r,c3)];
      let ex = select(0u, inc - MASK[idx(r,c3)], inc>0u || MASK[idx(r,c3)]>0u);
      PREFIX[idx(r,c3)] = ex;
      c3 += 256u;
    } else { break; }
  }
}

@compute @workgroup_size(256)
fn scatter_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.y; let c = gid.x;
  if (r>=U.rows || c>=U.cols){ return; }
  if (MASK[idx(r,c)] != 0u) {
    let dst = PREFIX[idx(r,c)];
    Y[idx(r,dst)] = X[idx(r,c)];
    IDX[idx(r,dst)] = i32(c);
  }
}
