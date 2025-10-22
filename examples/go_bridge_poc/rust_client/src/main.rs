use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Send numeric payloads to the Go bridge service"
)]
struct Args {
    /// HTTP endpoint to call, e.g. http://127.0.0.1:8080/predict
    #[arg(short, long, default_value = "http://127.0.0.1:8080/predict")]
    endpoint: String,

    /// List of floating point values to send
    #[arg(required = true)]
    values: Vec<f64>,
}

#[derive(Serialize)]
struct PredictionRequest {
    input: Vec<f64>,
}

#[derive(Deserialize, Debug)]
struct PredictionResponse {
    sum: f64,
    count: usize,
    average: f64,
}

fn main() {
    let args = Args::parse();

    let payload = PredictionRequest { input: args.values };

    let response: PredictionResponse = ureq::post(&args.endpoint)
        .set("Content-Type", "application/json")
        .send_json(ureq::serde_json::to_value(payload).expect("serialize payload"))
        .map_err(|err| {
            eprintln!("request failed: {err}");
            err
        })
        .unwrap()
        .into_json()
        .expect("invalid JSON response");

    println!(
        "Go service responded with: sum={:.3}, count={}, avg={:.3}",
        response.sum, response.count, response.average
    );
}
