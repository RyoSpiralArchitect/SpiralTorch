use std::{fs, path::Path};

fn main() {
    let ascii = r#"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    d8,                    d8b                                      d8b      
                   `8P                     88P        d8P                           ?88      
                                          d88      d888888P                          88b     
 .d888b,?88,.d88b,  88b  88bd88b d888b8b  888        ?88'   d8888b   88bd88b d8888b  888888b 
 ?8b,   `?88'  ?88  88P  88P'  `d8P' ?88  ?88        88P   d8P' ?88  88P'  `d8P' `P  88P `?8b
   `?8b   88b  d8P d88  d88     88b  ,88b  88b       88b   88b  d88 d88     88b     d88   88P
`?888P'   888888P'd88' d88'     `?88P'`88b  88b      `?8b  `?8888P'd88'     `?888P'd88'   88b
          88P'                                                                               
         d88                                                                                 
         ?8P                                                                                 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"#;
    for line in ascii.lines() { println!("cargo:warning={}", line); }

    let gen = Path::new("src/backend/wgpu_heuristics_generated.rs");
    if !gen.exists() {
        let stub = r#"// auto-generated stub (build.rs)
// return None => fallback path
pub(super) fn choose(_rows: u32, _cols: u32, _k: u32, _subgroup: bool) -> Option<super::Choice> { None }
"#;
        fs::write(gen, stub).expect("write heuristics stub");
        println!("cargo:warning=st-core: wrote stub backend/wgpu_heuristics_generated.rs");
    }
}
