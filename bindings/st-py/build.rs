use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=SPIRALTORCH_PY_LINK_MODE");

    let python = std::env::var("PYTHON_SYS_EXECUTABLE").unwrap_or_else(|_| "python3".to_string());

    let probe = Command::new(&python)
        .arg("-c")
        .arg(
            r#"import sysconfig
libdir = sysconfig.get_config_var('LIBDIR') or ''
libname = sysconfig.get_config_var('LDLIBRARY') or ''
print(libdir)
print(libname)
"#,
        )
        .output();

    let mut configured = false;

    if let Ok(output) = probe {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines();
            if let (Some(libdir), Some(libname)) = (lines.next(), lines.next()) {
                let libdir = libdir.trim();
                let libname = libname.trim();
                if !libdir.is_empty() && !libname.is_empty() {
                    let libpath = PathBuf::from(libdir).join(libname);
                    if libpath.exists() {
                        println!("cargo:rustc-link-search=native={}", libdir);
                        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{}", libdir);
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir);

                        let mut lib = libname.trim_start_matches("lib").to_string();
                        for suffix in [".so", ".a", ".dylib", ".dll"] {
                            if let Some(stripped) = lib.strip_suffix(suffix) {
                                lib = stripped.to_string();
                                break;
                            }
                        }

                        if !lib.is_empty() {
                            println!("cargo:rustc-link-lib=dylib={}", lib);
                            configured = true;
                        }
                    }
                }
            }
        }
    }

    if !configured {
        println!(
            "cargo:warning=spiraltorch-py: falling back to system linker search paths for Python (PYTHON_SYS_EXECUTABLE={}).",
            python
        );
    }

    pyo3_build_config::add_extension_module_link_args();
}
