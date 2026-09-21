use nerf_ray_integral_fixture::{contract_report, NerfCase};
use std::time::Instant;

fn main() {
    let mut cases = Vec::new();
    for batch in [1, 32, 256] {
        for samples in [1, 8, 64] {
            for varying in [false, true] {
                let mut case = NerfCase::new(batch, samples, varying, false);
                for _ in 0..5 {
                    std::hint::black_box(case.run());
                }
                let mut elapsed_ns = Vec::new();
                for _ in 0..9 {
                    let start = Instant::now();
                    for _ in 0..4 {
                        std::hint::black_box(case.run());
                    }
                    elapsed_ns.push(start.elapsed().as_nanos() as f64 / 4.0);
                }
                cases.push(serde_json::json!({"metadata": serde_json::from_str::<serde_json::Value>(&case.metadata()).unwrap(),
                    "values": case.run().values(), "elapsed_ns": elapsed_ns}));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"runtime": "native-rust-cpu", "warmups": 5, "intervals": 9,
        "repetitions": 4, "contract": serde_json::from_str::<serde_json::Value>(&contract_report()).unwrap(), "cases": cases})
    );
}
