use positional_geometry_fixture::{reference, EncodingCase};
use serde_json::json;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let contracts: serde_json::Value = serde_json::from_str(&positional_geometry_fixture::contract_report()).unwrap();
    let mut cases = Vec::new();
    for rows in [1, 32, 1024] {
        for cols in [3, 6] {
            for bands in [0, 4, 10] {
                for residual in [false, true] {
                    let case = EncodingCase::new(rows, cols, bands, residual, 0).unwrap();
                    let expected = reference(rows, cols, bands, residual);
                    let run = || black_box(case.run().unwrap());
                    for _ in 0..3 { run(); }
                    let mut elapsed_ns = Vec::new();
                    for _ in 0..15 {
                        let start = Instant::now();
                        for _ in 0..8 { run(); }
                        elapsed_ns.push(start.elapsed().as_nanos() as f64 / 8.0);
                    }
                    let output = run();
                    assert_eq!(output.data().len(), expected.len());
                    assert!(output.data().iter().zip(&expected).all(|(&a, &b)| a.is_finite() && (f64::from(a)-b).abs() <= 2e-6));
                    cases.push(json!({"rows": rows, "cols": cols, "bands": bands, "residual": residual,
                        "valid": true, "elapsed_ns": elapsed_ns,
                        "output_bits": output.data().iter().map(|x|x.to_bits()).collect::<Vec<_>>() }));
                }
            }
        }
    }
    println!("{}",json!({"schema":"spiraltorch.positional_encoding.v1", "cases":cases,"contracts":contracts,
        "warmups":3,"intervals":15,"repetitions":8,
        "boundary":"Actual st-vision row-major host encoding, output allocation/free included; setup and export excluded; no field/training/GPU claim"}));
}
