use std::fs;
use std::path::PathBuf;

#[test]
fn backend_matrix_markdown_is_in_sync() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace layout: crates/st-bench");
    let doc_path = repo_root.join("docs").join("backend_matrix.md");
    let doc = fs::read_to_string(&doc_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", doc_path.display()));
    let updated = st_bench::backend_matrix::sync_backend_matrix_markdown(&doc)
        .unwrap_or_else(|err| panic!("failed to sync backend matrix markdown: {err}"));
    assert_eq!(
        doc, updated,
        "docs/backend_matrix.md is out of date; run `cargo run -p st-bench --bin backend_matrix_md -- --write --doc docs/backend_matrix.md`"
    );
}

