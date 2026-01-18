// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let mut doc: Option<PathBuf> = None;
    let mut check = false;
    let mut write = false;
    let mut stdout = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--doc" => {
                let Some(path) = args.next() else {
                    eprintln!("backend_matrix_md: --doc requires a path");
                    std::process::exit(2);
                };
                doc = Some(PathBuf::from(path));
            }
            "--check" => check = true,
            "--write" => write = true,
            "--stdout" => stdout = true,
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("backend_matrix_md: unknown argument '{other}'");
                print_help();
                std::process::exit(2);
            }
        }
    }

    if stdout && (check || write) {
        eprintln!("backend_matrix_md: --stdout cannot be combined with --check/--write");
        std::process::exit(2);
    }

    if stdout || (doc.is_none() && !check && !write) {
        print!("{}", st_bench::backend_matrix::backend_matrix_autogen_block());
        return;
    }

    let doc_path = doc.unwrap_or_else(|| PathBuf::from("docs/backend_matrix.md"));
    let original = fs::read_to_string(&doc_path).unwrap_or_else(|err| {
        eprintln!("backend_matrix_md: failed to read {}: {err}", doc_path.display());
        std::process::exit(2);
    });

    let updated = st_bench::backend_matrix::sync_backend_matrix_markdown(&original)
        .unwrap_or_else(|err| {
            eprintln!("backend_matrix_md: {err}");
            std::process::exit(2);
        });

    if check {
        if updated != original {
            eprintln!(
                "backend_matrix_md: {} is out of date; run with --write",
                doc_path.display()
            );
            std::process::exit(1);
        }
        return;
    }

    if !write {
        eprintln!("backend_matrix_md: pass --write to update {}", doc_path.display());
        std::process::exit(2);
    }

    fs::write(&doc_path, updated).unwrap_or_else(|err| {
        eprintln!(
            "backend_matrix_md: failed to write {}: {err}",
            doc_path.display()
        );
        std::process::exit(2);
    });
}

fn print_help() {
    println!(
        "Usage: backend_matrix_md [--doc <path>] [--check|--write|--stdout]\n\n\
        --stdout         Print the autogen block to stdout (default when no mode is selected)\n\
        --doc <path>     Path to docs/backend_matrix.md (default: docs/backend_matrix.md)\n\
        --check          Exit non-zero if the doc differs from the generated table\n\
        --write          Rewrite the doc in-place with the generated table\n"
    );
}

