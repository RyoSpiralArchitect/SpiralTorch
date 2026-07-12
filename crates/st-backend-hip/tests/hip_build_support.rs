// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[path = "../build_support.rs"]
mod build_support;

use std::ffi::OsString;

use build_support::{archive_kernels_with, compile_kernels_with, CommandReport, HIP_KERNELS};
use tempfile::tempdir;

fn success() -> CommandReport {
    CommandReport {
        success: true,
        code: Some(0),
        stdout: String::new(),
        stderr: String::new(),
    }
}

#[test]
fn configured_kernels_have_unique_object_names() {
    let stems = HIP_KERNELS
        .iter()
        .map(|source| {
            std::path::Path::new(source)
                .file_stem()
                .expect("configured kernel has no file stem")
                .to_os_string()
        })
        .collect::<std::collections::HashSet<_>>();

    assert!(!HIP_KERNELS.is_empty());
    assert_eq!(stems.len(), HIP_KERNELS.len());
}

#[test]
fn compile_requires_every_configured_kernel_to_succeed() {
    let out = tempdir().expect("temporary output directory");
    let kernels = ["src/a.cu", "src/b.cu", "src/c.cu"];
    let mut calls = Vec::new();

    let err = compile_kernels_with(
        "fake-hipcc".as_ref(),
        out.path(),
        &kernels,
        |program, args| {
            calls.push((program.to_os_string(), args.to_vec()));
            if calls.len() == 2 {
                Ok(CommandReport {
                    success: false,
                    code: Some(23),
                    stdout: "compiler output".to_string(),
                    stderr: "kernel rejected".to_string(),
                })
            } else {
                Ok(success())
            }
        },
    )
    .expect_err("a failed kernel must fail the native build");

    assert_eq!(calls.len(), 2, "compilation must stop at first failure");
    assert!(err.contains("src/b.cu"));
    assert!(err.contains("exit status 23"));
    assert!(err.contains("compiler output"));
    assert!(err.contains("kernel rejected"));
}

#[test]
fn compile_builds_one_unique_object_per_kernel() {
    let out = tempdir().expect("temporary output directory");
    let kernels = ["src/alpha.cu", "src/beta.cu"];
    let mut calls = Vec::new();

    let objects = compile_kernels_with(
        "fake-hipcc".as_ref(),
        out.path(),
        &kernels,
        |program, args| {
            calls.push((program.to_os_string(), args.to_vec()));
            Ok(success())
        },
    )
    .expect("all fake kernel compilations should succeed");

    assert_eq!(
        objects,
        vec![out.path().join("alpha.o"), out.path().join("beta.o")]
    );
    assert_eq!(calls.len(), kernels.len());
    for (index, (program, args)) in calls.iter().enumerate() {
        assert_eq!(program, &OsString::from("fake-hipcc"));
        assert!(args.contains(&OsString::from("-O3")));
        assert!(args.contains(&OsString::from("-c")));
        assert!(args.contains(&OsString::from(kernels[index])));
        assert!(args.contains(&objects[index].as_os_str().to_os_string()));
    }
}

#[test]
fn compile_reports_launch_failure_with_kernel_context() {
    let out = tempdir().expect("temporary output directory");

    let err = compile_kernels_with(
        "missing-hipcc".as_ref(),
        out.path(),
        &["src/kernel.cu"],
        |_, _| Err("executable not found".to_string()),
    )
    .expect_err("an unavailable HIP compiler must fail the native build");

    assert!(err.contains("missing-hipcc"));
    assert!(err.contains("src/kernel.cu"));
    assert!(err.contains("executable not found"));
}

#[test]
fn archive_rejects_empty_object_sets() {
    let out = tempdir().expect("temporary output directory");
    let archive = out.path().join("libsthipkernels.a");
    let mut invoked = false;

    let err = archive_kernels_with("ar".as_ref(), &archive, &[], |_, _| {
        invoked = true;
        Ok(success())
    })
    .expect_err("an empty object set must not produce a native archive");

    assert!(!invoked);
    assert!(err.contains("empty HIP kernel archive"));
}

#[test]
fn archive_failure_is_not_downgraded_to_a_warning() {
    let out = tempdir().expect("temporary output directory");
    let archive = out.path().join("libsthipkernels.a");
    let objects = vec![out.path().join("a.o"), out.path().join("b.o")];

    let err = archive_kernels_with("fake-ar".as_ref(), &archive, &objects, |_, _| {
        Ok(CommandReport {
            success: false,
            code: Some(7),
            stdout: String::new(),
            stderr: "archive write failed".to_string(),
        })
    })
    .expect_err("an archive failure must fail the native build");

    assert!(err.contains("fake-ar"));
    assert!(err.contains("exit status 7"));
    assert!(err.contains("archive write failed"));
}

#[test]
fn archive_includes_every_compiled_object() {
    let out = tempdir().expect("temporary output directory");
    let archive = out.path().join("libsthipkernels.a");
    let objects = vec![out.path().join("a.o"), out.path().join("b.o")];
    let mut captured = None;

    archive_kernels_with("fake-ar".as_ref(), &archive, &objects, |program, args| {
        captured = Some((program.to_os_string(), args.to_vec()));
        Ok(success())
    })
    .expect("fake archive should succeed");

    let (program, args) = captured.expect("archive command was not invoked");
    assert_eq!(program, OsString::from("fake-ar"));
    assert_eq!(args[0], OsString::from("crus"));
    assert_eq!(args[1], archive.as_os_str());
    assert_eq!(args[2], objects[0].as_os_str());
    assert_eq!(args[3], objects[1].as_os_str());
}
