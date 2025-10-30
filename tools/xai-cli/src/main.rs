use clap::{ArgAction, Args, Parser, Subcommand, ValueHint};
use serde::{Deserialize, Serialize};
use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use st_nn::module::Module;
use st_tensor::{PureResult, Tensor};
use st_vision::{
    models::run_integrated_gradients,
    xai::{
        AttributionOutput, AttributionStatistics, GradCam, GradCamConfig, IntegratedGradientsConfig,
    },
};
use std::error::Error;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

type DynError = Box<dyn Error>;

type Result<T> = std::result::Result<T, DynError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiskTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DiskTensor {
    fn into_tensor(self) -> Result<Tensor> {
        if self.rows == 0 || self.cols == 0 {
            return Err(Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "tensor must have non-zero dimensions",
            )));
        }
        if self.data.len() != self.rows * self.cols {
            return Err(Box::new(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "tensor payload has {} values but expected {} (rows * cols)",
                    self.data.len(),
                    self.rows * self.cols
                ),
            )));
        }
        Tensor::from_vec(self.rows, self.cols, self.data).map_err(|err| Box::new(err) as DynError)
    }
}

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Explainability driver for SpiralTorch vision models"
)]
struct Cli {
    /// Attach basic statistics (min/max/mean/entropy) to the attribution metadata output
    #[arg(long, global = true)]
    include_stats: bool,

    /// Print attribution statistics to STDOUT after generation
    #[arg(long, global = true)]
    print_stats: bool,

    /// Emit the metadata component to a separate JSON file in addition to the main report
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    metadata_out: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a Grad-CAM heatmap from saved activations and gradients
    GradCam(GradCamArgs),

    /// Run Integrated Gradients using an optional linear probe
    IntegratedGradients(IntegratedGradientsArgs),
}

#[derive(Args)]
struct GradCamArgs {
    /// Path to the saved activation tensor (JSON, DiskTensor format)
    #[arg(long, value_hint = ValueHint::FilePath)]
    activations: PathBuf,

    /// Path to the saved gradient tensor (JSON, DiskTensor format)
    #[arg(long, value_hint = ValueHint::FilePath)]
    gradients: PathBuf,

    /// Height of the original input in pixels
    #[arg(long)]
    height: usize,

    /// Width of the original input in pixels
    #[arg(long)]
    width: usize,

    /// Optional epsilon used while normalising the Grad-CAM heatmap
    #[arg(long, default_value_t = 1e-6)]
    epsilon: f32,

    /// Disable the ReLU post-processing stage applied to the weighted activations
    #[arg(long = "no-relu", action = ArgAction::SetFalse, default_value_t = true)]
    apply_relu: bool,

    /// Preserve the raw heatmap values without unit-interval normalisation
    #[arg(long = "raw-heatmap", action = ArgAction::SetFalse, default_value_t = true)]
    normalise: bool,

    /// Optional logical layer label to embed in the metadata
    #[arg(long)]
    layer: Option<String>,

    /// Destination for the attribution report JSON payload
    #[arg(long, value_hint = ValueHint::FilePath)]
    output: PathBuf,
}

#[derive(Args)]
struct IntegratedGradientsArgs {
    /// Path to the input tensor (JSON, DiskTensor format)
    #[arg(long, value_hint = ValueHint::FilePath)]
    input: PathBuf,

    /// Path to the baseline tensor (JSON, DiskTensor format)
    #[arg(long, value_hint = ValueHint::FilePath)]
    baseline: PathBuf,

    /// Number of integration steps to take
    #[arg(long)]
    steps: usize,

    /// Index within the model output to explain
    #[arg(long, value_parser = clap::value_parser!(usize))]
    target: usize,

    /// Optional label to record alongside the target index
    #[arg(long)]
    target_label: Option<String>,

    /// Optional linear probe weights applied before attribution
    #[arg(long, value_hint = ValueHint::FilePath)]
    weights: Option<PathBuf>,

    /// Optional logical layer label to embed in the metadata
    #[arg(long)]
    layer: Option<String>,

    /// Destination for the attribution report JSON payload
    #[arg(long, value_hint = ValueHint::FilePath)]
    output: PathBuf,
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::GradCam(args) => {
            let output = run_grad_cam(args)?;
            finalise_output(&cli, output, &args.output)
        }
        Command::IntegratedGradients(args) => {
            let output = run_integrated_gradients_cli(args)?;
            finalise_output(&cli, output, &args.output)
        }
    }
}

fn finalise_output(cli: &Cli, mut output: AttributionOutput, destination: &Path) -> Result<()> {
    let statistics = if cli.include_stats || cli.print_stats {
        Some(output.statistics())
    } else {
        None
    };

    if let Some(stats) = statistics.as_ref() {
        if cli.include_stats {
            attach_statistics(&mut output.metadata, stats);
        }
        if cli.print_stats {
            print_statistics(stats, &output.metadata);
        }
    }

    let report = output.to_report();
    write_report(&report, destination)?;

    if let Some(path) = cli.metadata_out.as_ref() {
        write_metadata(&output.metadata, path)?;
    }

    Ok(())
}

fn run_grad_cam(args: &GradCamArgs) -> Result<AttributionOutput> {
    if args.height == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--height must be greater than zero",
        )));
    }
    if args.width == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--width must be greater than zero",
        )));
    }
    let activations = read_tensor(&args.activations)?;
    let gradients = read_tensor(&args.gradients)?;
    let config = GradCamConfig {
        height: args.height,
        width: args.width,
        apply_relu: args.apply_relu,
        epsilon: args.epsilon,
        normalise: args.normalise,
    };
    let heatmap = GradCam::attribute(&activations, &gradients, &config)
        .map_err(|err| Box::new(err) as DynError)?;
    let mut metadata = AttributionMetadata::for_algorithm("grad-cam");
    if let Some(layer) = &args.layer {
        metadata.layer = Some(layer.clone());
    }
    metadata.insert_extra_number("height", args.height as f64);
    metadata.insert_extra_number("width", args.width as f64);
    metadata.insert_extra_flag("apply_relu", config.apply_relu);
    metadata.insert_extra_number("epsilon", config.epsilon as f64);
    metadata.insert_extra_flag("normalise", config.normalise);
    Ok(AttributionOutput::new(heatmap, metadata))
}

fn run_integrated_gradients_cli(args: &IntegratedGradientsArgs) -> Result<AttributionOutput> {
    if args.steps == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--steps must be greater than zero",
        )));
    }
    let input = read_tensor(&args.input)?;
    let baseline = read_tensor(&args.baseline)?;
    let weights = match &args.weights {
        Some(path) => Some(read_tensor(path)?),
        None => None,
    };
    let mut model = CliModel::from_weights(weights)?;
    let config = IntegratedGradientsConfig::new(args.steps, args.target);
    let mut attribution = run_integrated_gradients(
        &mut model,
        &input,
        &baseline,
        config,
        args.target_label.as_deref(),
    )
    .map_err(|err| Box::new(err) as DynError)?;

    if let Some(layer) = &args.layer {
        attribution.metadata.layer = Some(layer.clone());
    }
    attribution.metadata.insert_extra_text(
        "cli_model",
        if args.weights.is_some() {
            "linear"
        } else {
            "identity"
        },
    );

    Ok(attribution)
}

fn attach_statistics(metadata: &mut AttributionMetadata, stats: &AttributionStatistics) {
    metadata.insert_extra_number("heatmap_min", stats.min as f64);
    metadata.insert_extra_number("heatmap_max", stats.max as f64);
    metadata.insert_extra_number("heatmap_mean", stats.mean as f64);
    metadata.insert_extra_number("heatmap_entropy", stats.entropy as f64);
}

fn print_statistics(stats: &AttributionStatistics, metadata: &AttributionMetadata) {
    let mut context = format!("{}", metadata.algorithm);
    if let Some(layer) = &metadata.layer {
        context.push_str(&format!("/{layer}"));
    }
    if let Some(target) = &metadata.target {
        context.push_str(&format!(" -> {target}"));
    }
    println!(
        "[{context}] min: {min:.6}, max: {max:.6}, mean: {mean:.6}, entropy: {entropy:.6}",
        min = stats.min,
        max = stats.max,
        mean = stats.mean,
        entropy = stats.entropy
    );
}

fn read_tensor(path: &Path) -> Result<Tensor> {
    let contents = fs::read_to_string(path)?;
    let disk: DiskTensor = serde_json::from_str(&contents)?;
    disk.into_tensor()
}

fn write_report(report: &AttributionReport, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let payload = serde_json::to_string_pretty(report)?;
    fs::write(path, payload)?;
    Ok(())
}

fn write_metadata(metadata: &AttributionMetadata, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let payload = serde_json::to_string_pretty(metadata)?;
    fs::write(path, payload)?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

enum CliModel {
    Identity,
    Linear { weights: Tensor },
}

impl CliModel {
    fn from_weights(weights: Option<Tensor>) -> Result<Self> {
        match weights {
            Some(weights) => Ok(Self::Linear { weights }),
            None => Ok(Self::Identity),
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
