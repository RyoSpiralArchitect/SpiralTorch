struct RC { nd:u32, n:u32 };
@group(92) @binding(0) var<storage, read>  cond: array<u32>;
@group(92) @binding(1) var<storage, read>  x: array<f32>;
@group(92) @binding(2) var<storage, read>  y: array<f32>;
@group(92) @binding(3) var<storage, read_write> outv: array<f32>;
@group(92) @binding(4) var<storage, read>  shape: array<u32>;
@group(92) @binding(5) var<storage, read>  strides_out: array<u32>;
@group(92) @binding(6) var<storage, read>  strides_c: array<u32>;
@group(92) @binding(7) var<storage, read>  strides_x: array<u32>;
@group(92) @binding(8) var<storage, read>  strides_y: array<u32>;
@group(92) @binding(9) var<uniform> rc: RC;
fn coord_at(i:u32, d:u32) -> u32 { let s=strides_out[d]; let sh=shape[d]; return (i / s) % sh; }
@compute @workgroup_size(256)
fn where_nd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if(i>=rc.n){return;}
  var oc:u32=0u; var ox:u32=0u; var oy:u32=0u;
  for(var d:u32=0u; d<rc.nd; d=d+1u){ let cd=coord_at(i,d); oc+=cd*strides_c[d]; ox+=cd*strides_x[d]; oy+=cd*strides_y[d]; }
  let c=cond[oc]!=0u; let vx=x[ox]; let vy=y[oy]; outv[i]=select(vy,vx,c);
}
