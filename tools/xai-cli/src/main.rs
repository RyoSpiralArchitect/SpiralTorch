use serde::{Deserialize, Serialize};
use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor};
use st_vision::{
    models::run_integrated_gradients,
    xai::{AttributionOutput, GradCam, GradCamConfig, IntegratedGradientsConfig},
};
use std::env;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiskTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DiskTensor {
    fn into_tensor(self) -> Result<Tensor, Box<dyn std::error::Error>> {
        Tensor::from_vec(self.rows, self.cols, self.data)
            .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
    }
}

enum Algorithm {
    GradCam,
    IntegratedGradients,
}

impl Algorithm {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "grad-cam" => Ok(Self::GradCam),
            "integrated-gradients" => Ok(Self::IntegratedGradients),
            other => Err(Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                format!("unknown algorithm: {other}"),
            ))),
        }
    }
}

struct CliArgs {
    algorithm: Algorithm,
    activations: Option<PathBuf>,
    gradients: Option<PathBuf>,
    height: Option<usize>,
    width: Option<usize>,
    input: Option<PathBuf>,
    baseline: Option<PathBuf>,
    steps: Option<usize>,
    target: Option<usize>,
    target_label: Option<String>,
    layer: Option<String>,
    weights: Option<PathBuf>,
    epsilon: Option<f32>,
    apply_relu: bool,
    normalise: bool,
    output: PathBuf,
}

impl CliArgs {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = env::args().skip(1);
        let mut algorithm = None;
        let mut activations = None;
        let mut gradients = None;
        let mut height = None;
        let mut width = None;
        let mut input = None;
        let mut baseline = None;
        let mut steps = None;
        let mut target = None;
        let mut target_label = None;
        let mut layer = None;
        let mut weights = None;
        let mut epsilon = None;
        let mut apply_relu = true;
        let mut normalise = true;
        let mut output = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--algorithm" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--algorithm requires a value")
                    })?;
                    algorithm = Some(Algorithm::parse(&value)?);
                }
                "--activations" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--activations requires a path")
                    })?;
                    activations = Some(PathBuf::from(value));
                }
                "--gradients" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--gradients requires a path")
                    })?;
                    gradients = Some(PathBuf::from(value));
                }
                "--height" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--height requires a value")
                    })?;
                    height = Some(value.parse().map_err(|_| {
                        io::Error::new(ErrorKind::InvalidInput, "--height expects an integer")
                    })?);
                }
                "--width" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--width requires a value")
                    })?;
                    width = Some(value.parse().map_err(|_| {
                        io::Error::new(ErrorKind::InvalidInput, "--width expects an integer")
                    })?);
                }
                "--input" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--input requires a path")
                    })?;
                    input = Some(PathBuf::from(value));
                }
                "--baseline" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--baseline requires a path")
                    })?;
                    baseline = Some(PathBuf::from(value));
                }
                "--steps" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--steps requires a value")
                    })?;
                    steps = Some(value.parse().map_err(|_| {
                        io::Error::new(ErrorKind::InvalidInput, "--steps expects an integer")
                    })?);
                }
                "--target" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--target requires a value")
                    })?;
                    target = Some(value.parse().map_err(|_| {
                        io::Error::new(ErrorKind::InvalidInput, "--target expects an integer")
                    })?);
                }
                "--target-label" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--target-label requires a value")
                    })?;
                    target_label = Some(value);
                }
                "--layer" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--layer requires a value")
                    })?;
                    layer = Some(value);
                }
                "--weights" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--weights requires a path")
                    })?;
                    weights = Some(PathBuf::from(value));
                }
                "--epsilon" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--epsilon requires a value")
                    })?;
                    epsilon = Some(value.parse().map_err(|_| {
                        io::Error::new(
                            ErrorKind::InvalidInput,
                            "--epsilon expects a floating value",
                        )
                    })?);
                }
                "--no-relu" => {
                    apply_relu = false;
                }
                "--raw-heatmap" => {
                    normalise = false;
                }
                "--output" => {
                    let value = args.next().ok_or_else(|| {
                        io::Error::new(ErrorKind::InvalidInput, "--output requires a path")
                    })?;
                    output = Some(PathBuf::from(value));
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                unknown => {
                    return Err(Box::new(io::Error::new(
                        ErrorKind::InvalidInput,
                        format!("unknown argument: {unknown}"),
                    )));
                }
            }
        }

        let algorithm = algorithm.ok_or_else(|| {
            Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "--algorithm is required",
            )) as Box<dyn std::error::Error>
        })?;
        let output = output.ok_or_else(|| {
            Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "--output is required",
            )) as Box<dyn std::error::Error>
        })?;

        Ok(Self {
            algorithm,
            activations,
            gradients,
            height,
            width,
            input,
            baseline,
            steps,
            target,
            target_label,
            layer,
            weights,
            epsilon,
            apply_relu,
            normalise,
            output,
        })
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse()?;
    match args.algorithm {
        Algorithm::GradCam => run_grad_cam(&args),
        Algorithm::IntegratedGradients => run_integrated_gradients_cli(&args),
    }
}

fn run_grad_cam(args: &CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    let activations_path = required_path(&args.activations, "--activations for grad-cam")?;
    let gradients_path = required_path(&args.gradients, "--gradients for grad-cam")?;
    let height = required_usize(args.height, "--height for grad-cam")?;
    let width = required_usize(args.width, "--width for grad-cam")?;
    let epsilon = args.epsilon.unwrap_or(1e-6);

    let activations = read_tensor(activations_path)?;
    let gradients = read_tensor(gradients_path)?;
    let config = GradCamConfig {
        height,
        width,
        apply_relu: args.apply_relu,
        epsilon,
        normalise: args.normalise,
    };
    let heatmap = GradCam::attribute(&activations, &gradients, &config)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    let mut metadata = AttributionMetadata::for_algorithm("grad-cam");
    if let Some(layer) = &args.layer {
        metadata.layer = Some(layer.clone());
    }
    metadata.insert_extra_number("height", height as f64);
    metadata.insert_extra_number("width", width as f64);
    metadata.insert_extra_flag("apply_relu", config.apply_relu);
    metadata.insert_extra_number("epsilon", epsilon as f64);
    metadata.insert_extra_flag("normalise", config.normalise);
    let output = AttributionOutput::new(heatmap, metadata);
    write_report(&output.to_report(), &args.output)
}

fn run_integrated_gradients_cli(args: &CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = required_path(&args.input, "--input for integrated-gradients")?;
    let baseline_path = required_path(&args.baseline, "--baseline for integrated-gradients")?;
    let steps = required_usize(args.steps, "--steps for integrated-gradients")?;
    let target = required_usize(args.target, "--target for integrated-gradients")?;

    let input = read_tensor(input_path)?;
    let baseline = read_tensor(baseline_path)?;
    let weights = match &args.weights {
        Some(path) => Some(read_tensor(path)?),
        None => None,
    };
    let mut model = CliModel::from_weights(weights);
    let config = IntegratedGradientsConfig::new(steps, target);
    let mut attribution = run_integrated_gradients(
        &mut model,
        &input,
        &baseline,
        config,
        args.target_label.as_deref(),
    )
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    if let Some(layer) = &args.layer {
        attribution.metadata.layer = Some(layer.clone());
    }
    let model_kind = if args.weights.is_some() {
        "linear"
    } else {
        "identity"
    };
    attribution
        .metadata
        .insert_extra_text("cli_model", model_kind);
    write_report(&attribution.to_report(), &args.output)
}

fn read_tensor(path: &Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let disk: DiskTensor = serde_json::from_str(&contents)?;
    disk.into_tensor()
}

fn write_report(report: &AttributionReport, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let payload = serde_json::to_string_pretty(report)?;
    fs::write(path, payload)?;
    Ok(())
}

fn required_path<'a>(
    value: &'a Option<PathBuf>,
    label: &str,
) -> Result<&'a PathBuf, Box<dyn std::error::Error>> {
    value.as_ref().ok_or_else(|| {
        Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            format!("{label} is required"),
        )) as Box<dyn std::error::Error>
    })
}

fn required_usize(value: Option<usize>, label: &str) -> Result<usize, Box<dyn std::error::Error>> {
    value.ok_or_else(|| {
        Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            format!("{label} is required"),
        )) as Box<dyn std::error::Error>
    })
}

fn print_usage() {
    println!(
        "Usage: st-xai-cli --algorithm <grad-cam|integrated-gradients> [options]\n\
         \n\
         Grad-CAM:\n\
         --activations <file>\n\
         --gradients <file>\n\
         --height <pixels>\n\
         --width <pixels>\n\
         [--epsilon <float>] [--no-relu] [--raw-heatmap]\n\
         \n\
         Integrated Gradients:\n\
         --input <file> --baseline <file>\n\
         --steps <n> --target <index>\n\
         [--weights <file>] [--target-label <name>]\n\
         \n\
         Shared options:\n\
         [--layer <name>] --output <file>"
    );
}

enum CliModel {
    Identity,
    Linear { weights: Tensor },
}

impl CliModel {
    fn from_weights(weights: Option<Tensor>) -> Self {
        match weights {
            Some(weights) => Self::Linear { weights },
            None => Self::Identity,
        }
    }
}

impl Module for CliModel {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        match self {
            CliModel::Identity => Ok(input.clone()),
            CliModel::Linear { weights } => weights.matmul(input),
        }
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        match self {
            CliModel::Identity => Ok(grad_output.clone()),
            CliModel::Linear { weights } => weights.transpose().matmul(grad_output),
        }
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&st_nn::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut st_nn::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}
