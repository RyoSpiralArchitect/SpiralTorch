fn main(){
println!("cargo:warning=SpiralTorch: build stub for wgpu_heuristics_generated.rs");
use std::{fs, path::Path};
let g = Path::new("crates/st-core/src/backend/wgpu_heuristics_generated.rs");
if !g.exists() {
    fs::write(g, "pub fn choose(_:u32,_:u32,_:u32,_:bool)->Option<super::heuristics::Choice>{None}\n").ok();
}

}
