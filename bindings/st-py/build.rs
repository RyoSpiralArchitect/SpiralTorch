use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn sanitize_library_name(raw: &str) -> (String, bool) {
    let trimmed = raw.trim();
    let mut name = trimmed.trim_start_matches("lib").to_string();
    let mut is_static = false;
    let mut matched_suffix = false;

    for (suffix, static_flag) in [
        (".so", false),
        (".dylib", false),
        (".dll", false),
        (".lib", false),
        (".a", true),
    ] {
        if name.ends_with(suffix) {
            matched_suffix = true;
            if static_flag {
                is_static = true;
            }
            name.truncate(name.len() - suffix.len());
            break;
        }
    }

    if !matched_suffix {
        if let Some(index) = name.find(".so.") {
            name.truncate(index);
        }
    }

    (name, is_static)
}

fn probe_python(python: &str) -> Option<PythonLinkMetadata> {
    let script = r#"
import sysconfig

def cfg(name):
    value = sysconfig.get_config_var(name)
    return value or ''

def unique(seq):
    seen = set()
    out = []
    for item in seq:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out

libdirs = unique([
    cfg('LIBDIR'),
    cfg('LIBPL'),
    cfg('LIBDEST'),
    cfg('installed_base'),
    cfg('base'),
    cfg('platbase'),
])

libnames = unique([
    cfg('LDLIBRARY'),
    cfg('LIBRARY'),
    cfg('LIBPYTHON'),
])

framework = cfg('PYTHONFRAMEWORK')
frameworkdir = cfg('PYTHONFRAMEWORKPREFIX')
linkfor_shared = cfg('LINKFORSHARED')

print('LIBDIRS=' + ';'.join(libdirs))
print('LIBNAMES=' + ';'.join(libnames))
print('FRAMEWORK=' + framework)
print('FRAMEWORKDIR=' + frameworkdir)
print('LINKFORSHARED=' + linkfor_shared)
"#;

    let output = Command::new(python).arg("-c").arg(script).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut libdirs = Vec::new();
    let mut libnames = Vec::new();
    let mut framework = None;
    let mut frameworkdir = None;
    let mut linkfor_shared = String::new();

    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("LIBDIRS=") {
            libdirs = rest
                .split(';')
                .filter_map(|dir| {
                    let trimmed = dir.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect();
        } else if let Some(rest) = line.strip_prefix("LIBNAMES=") {
            libnames = rest
                .split(';')
                .filter_map(|name| {
                    let trimmed = name.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect();
        } else if let Some(rest) = line.strip_prefix("FRAMEWORK=") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                framework = Some(trimmed.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("FRAMEWORKDIR=") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                frameworkdir = Some(trimmed.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("LINKFORSHARED=") {
            linkfor_shared = rest.trim().to_string();
        }
    }

    Some(PythonLinkMetadata {
        libdirs,
        libnames,
        framework,
        frameworkdir,
        linkfor_shared,
    })
}

struct PythonLinkMetadata {
    libdirs: Vec<String>,
    libnames: Vec<String>,
    framework: Option<String>,
    frameworkdir: Option<String>,
    linkfor_shared: String,
}

fn find_python_library(metadata: &PythonLinkMetadata) -> Option<(String, bool, PathBuf)> {
    for dir in &metadata.libdirs {
        let base = Path::new(dir);
        for name in &metadata.libnames {
            let candidate = base.join(name);
            if candidate.exists() {
                let (lib, is_static) = sanitize_library_name(name);
                return Some((lib, is_static, base.to_path_buf()));
            }
        }
    }

    for dir in &metadata.libdirs {
        let base = Path::new(dir);
        if let Ok(entries) = base.read_dir() {
            let mut dynamic_choice: Option<(String, PathBuf)> = None;
            let mut static_choice: Option<(String, PathBuf)> = None;

            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(file_name) = path.file_name().and_then(|f| f.to_str()) {
                    if !file_name.starts_with("libpython") {
                        continue;
                    }

                    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
                        match ext {
                            "so" | "dylib" | "dll" => {
                                dynamic_choice = Some((file_name.to_string(), base.to_path_buf()));
                            }
                            "a" => {
                                static_choice = Some((file_name.to_string(), base.to_path_buf()));
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some((name, dir)) = dynamic_choice.or(static_choice) {
                let (lib, is_static) = sanitize_library_name(&name);
                return Some((lib, is_static, dir));
            }
        }
    }

    None
}

fn apply_linkfor_shared(
    args: &str,
    emitted_searches: &mut HashSet<String>,
    emitted_frameworks: &mut HashSet<String>,
) -> bool {
    if args.is_empty() {
        return false;
    }

    let mut configured = false;
    let mut tokens = args.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token.starts_with("-L") {
            let path = token.trim_start_matches("-L");
            if !path.is_empty() && emitted_searches.insert(format!("native:{path}")) {
                println!("cargo:rustc-link-search=native={}", path);
            }
        } else if token.starts_with("-F") {
            let path = token.trim_start_matches("-F");
            if !path.is_empty() && emitted_searches.insert(format!("framework:{path}")) {
                println!("cargo:rustc-link-search=framework={}", path);
            }
        } else if token == "-framework" {
            if let Some(name) = tokens.next() {
                if !name.is_empty() && emitted_frameworks.insert(name.to_string()) {
                    println!("cargo:rustc-link-lib=framework={}", name);
                    configured = true;
                }
            }
        } else if let Some(rest) = token.strip_prefix("-framework=") {
            if !rest.is_empty() && emitted_frameworks.insert(rest.to_string()) {
                println!("cargo:rustc-link-lib=framework={}", rest);
                configured = true;
            }
        } else if token.starts_with("-l") {
            let lib = token.trim_start_matches("-l");
            if !lib.is_empty() {
                println!("cargo:rustc-link-lib={}", lib);
                configured = true;
            }
        } else {
            println!("cargo:rustc-link-arg={}", token);
        }
    }

    configured
}

fn main() {
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=SPIRALTORCH_PY_LINK_MODE");

    let python = std::env::var("PYTHON_SYS_EXECUTABLE").unwrap_or_else(|_| "python3".to_string());

    let mut configured = false;
    let mut emitted_searches: HashSet<String> = HashSet::new();
    let mut emitted_frameworks: HashSet<String> = HashSet::new();

    if let Some(metadata) = probe_python(&python) {
        if let Some((lib, is_static, dir)) = find_python_library(&metadata) {
            let dir_str = dir.display().to_string();
            if emitted_searches.insert(format!("native:{dir_str}")) {
                println!("cargo:rustc-link-search=native={}", dir_str);
            }

            if !is_static {
                if emitted_searches.insert(format!("rpath:{dir_str}")) {
                    println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{}", dir_str);
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir_str);
                }
                println!("cargo:rustc-link-lib=dylib={}", lib);
            } else {
                println!("cargo:rustc-link-lib=static={}", lib);
            }
            configured = true;
        }

        if let Some(dir) = &metadata.frameworkdir {
            if let Some(framework) = &metadata.framework {
                if !dir.is_empty() && !framework.is_empty() {
                    if emitted_searches.insert(format!("framework:{}", dir)) {
                        println!("cargo:rustc-link-search=framework={}", dir);
                    }
                    if emitted_frameworks.insert(framework.clone()) {
                        println!("cargo:rustc-link-lib=framework={}", framework);
                        configured = true;
                    }
                }
            }
        }

        if apply_linkfor_shared(
            &metadata.linkfor_shared,
            &mut emitted_searches,
            &mut emitted_frameworks,
        ) {
            configured = true;
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
