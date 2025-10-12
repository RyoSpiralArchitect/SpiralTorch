use std::{env, process::Command, path::PathBuf};

fn main(){
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    if env::var("CARGO_FEATURE_HIP_REAL").is_ok() {
        if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
            println!("cargo:rustc-link-search=native={}/lib", rocm);
            println!("cargo:rustc-link-search=native={}/lib64", rocm);
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rccl");

        let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
        let src = "src/hip_kernels/topk_pass1.cu";
        let obj = out.join("topk_pass1.o");
        let status = Command::new(&hipcc)
            .args(["-O3","--std=c++17","-ffast-math","-fPIC","-c",src,"-o"])
            .arg(&obj)
            .status().expect("failed to run hipcc");
        if !status.success(){
            println!("cargo:warning=hipcc failed; HIP kernels will be unavailable");
        } else {
            let lib = out.join("libsthipkernels.a");
            let _ = Command::new("ar").args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()]).status();
            println!("cargo:rustc-link-lib=static=sthipkernels");
            println!("cargo:rustc-link-search=native={}", out.display());
        }
    }
}
