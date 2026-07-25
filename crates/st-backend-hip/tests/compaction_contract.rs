use st_backend_hip::compaction_contract::{compact_rows_reference_f32, CompactionShape};

#[test]
fn public_reference_contract_exposes_stable_rows_and_counts() {
    let output = compact_rows_reference_f32(
        &[0.3, 0.8, 0.5, 1.2],
        &[7, 8, 9, 10],
        CompactionShape {
            rows: 1,
            cols: 4,
            low: 0.5,
            high: 1.0,
        },
    )
    .unwrap();

    assert_eq!(output.row_counts(), &[2]);
    assert_eq!(output.row_values(0), Some(&[0.8, 0.5][..]));
    assert_eq!(output.row_indices(0), Some(&[8, 9][..]));
    assert_eq!(output.values_storage(), &[0.8, 0.5, 0.0, 0.0]);
    assert_eq!(output.indices_storage(), &[8, 9, 0, 0]);
}
