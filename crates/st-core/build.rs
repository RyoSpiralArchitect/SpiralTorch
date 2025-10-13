use std::{fs, path::Path};
fn main(){
    let gen = Path::new("src/backend/wgpu_heuristics_generated.rs");
    if !gen.exists(){
        let stub = r#"
#[allow(unused)]
pub fn choose(_rows: usize, _cols: usize, _k: usize, _subgroup: bool) -> Option<super::Choice> {
    None
}
"#;
        fs::create_dir_all("src/backend").ok();
        fs::write(gen, stub).expect("write stub heuristics_generated");
        println!("cargo:warning=st-core: wrote stub backend/wgpu_heuristics_generated.rs");
    }
}
