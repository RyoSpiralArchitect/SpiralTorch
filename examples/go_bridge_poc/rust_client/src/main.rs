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
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
}

fn main() {
    let args = Args::parse();

    let payload = PredictionRequest { input: args.values };

    let http_response = ureq::post(&args.endpoint)
        .set("Content-Type", "application/json")
        .send_json(ureq::serde_json::to_value(payload).expect("serialize payload"))
        .unwrap_or_else(|err| {
            match err {
                ureq::Error::Status(code, response) => {
                    if let Ok(body) = response.into_string() {
                        eprintln!("request failed with status {code}: {body}");
                    } else {
                        eprintln!("request failed with status {code}");
                    }
                }
                ureq::Error::Transport(transport) => {
                    eprintln!("transport error: {transport}");
                }
            }
            std::process::exit(1);
        });

    let response: PredictionResponse = http_response.into_json().expect("invalid JSON response");

    println!(
        "Go service responded with: sum={:.3}, count={}, avg={:.3}, min={:?}, max={:?}",
        response.sum, response.count, response.average, response.min, response.max
    );
}
