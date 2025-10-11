struct RC { nd:u32, n:u32 };
@group(93) @binding(0) var<storage, read>  cond: array<u32>;
@group(93) @binding(1) var<storage, read>  x: array<f32>;
@group(93) @binding(2) var<storage, read>  y: array<f32>;
@group(93) @binding(3) var<storage, read_write> outv: array<f32>;
@group(93) @binding(4) var<storage, read>  out_shape: array<u32>;
@group(93) @binding(5) var<storage, read>  out_strides: array<u32>;
@group(93) @binding(6) var<storage, read>  c_shape: array<u32>;
@group(93) @binding(7) var<storage, read>  c_strides: array<u32>;
@group(93) @binding(8) var<storage, read>  x_shape: array<u32>;
@group(93) @binding(9) var<storage, read>  x_strides: array<u32>;
@group(93) @binding(10) var<storage, read>  y_shape: array<u32>;
@group(93) @binding(11) var<storage, read>  y_strides: array<u32>;
@group(93) @binding(12) var<uniform> rc: RC;
fn coord_at(i:u32, d:u32)->u32{ (i / out_strides[d]) % out_shape[d] }
@compute @workgroup_size(256)
fn where_nd_strided(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if(i>=rc.n){return;}
  var oc:u32=0u; var ox:u32=0u; var oy:u32=0u;
  for(var d:u32=0u; d<rc.nd; d=d+1u){
    let cd=coord_at(i,d);
    let ic=select(0u,cd,c_shape[d]>1u);
    let ix=select(0u,cd,x_shape[d]>1u);
    let iy=select(0u,cd,y_shape[d]>1u);
    oc += ic * c_strides[d];
    ox += ix * x_strides[d];
    oy += iy * y_strides[d];
  }
  let c=cond[oc]!=0u; let vx=x[ox]; let vy=y[oy]; outv[i]=select(vy,vx,c);
}
