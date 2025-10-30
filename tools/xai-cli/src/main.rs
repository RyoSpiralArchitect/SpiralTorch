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

    fn from_tensor(tensor: &Tensor) -> Self {
        let (rows, cols) = tensor.shape();
        Self {
            rows,
            cols,
            data: tensor.data().to_vec(),
        }
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

    /// Write the processed heatmap tensor as a DiskTensor JSON payload
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    heatmap_out: Option<PathBuf>,

    /// Emit attribution statistics to a standalone JSON document for downstream tooling
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    stats_out: Option<PathBuf>,

    /// Base tensor used when constructing overlay artefacts from the generated heatmap
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    overlay_base: Option<PathBuf>,

    /// Alpha applied when blending the generated heatmap with the provided base tensor
    #[arg(long, global = true, default_value_t = 0.35)]
    overlay_alpha: f32,

    /// Destination for the blended heatmap overlay artefact
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    overlay_out: Option<PathBuf>,

    /// Destination for the threshold-gated overlay artefact
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    gated_overlay_out: Option<PathBuf>,

    /// Apply an odd-sized box blur to smooth the final heatmap before emitting it
    #[arg(long, global = true, value_hint = ValueHint::Other)]
    smooth_kernel: Option<usize>,

    /// Normalise the heatmap values to the unit interval prior to writing them out
    #[arg(long, global = true)]
    normalise_output: bool,

    /// Threshold used when emitting a binary focus mask; stored as metadata even without output
    #[arg(long, global = true, value_hint = ValueHint::Other)]
    focus_threshold: Option<f32>,

    /// Destination file for a binary focus mask derived from the generated heatmap
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    focus_mask_out: Option<PathBuf>,

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
    output = apply_post_processing(cli, output)?;

    let overlay_requested = cli.overlay_out.is_some() || cli.gated_overlay_out.is_some();

    if overlay_requested && cli.overlay_base.is_none() {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--overlay-base is required when emitting overlay artefacts",
        )));
    }

    if (overlay_requested || cli.overlay_base.is_some())
        && (!(0.0..=1.0).contains(&cli.overlay_alpha) || !cli.overlay_alpha.is_finite())
    {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--overlay-alpha must be between 0.0 and 1.0",
        )));
    }

    let statistics = if cli.include_stats || cli.print_stats || cli.stats_out.is_some() {
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

    if let Some(path) = cli.heatmap_out.as_ref() {
        write_tensor_json(&output.map, path)?;
    }

    let overlay_base = if overlay_requested {
        Some(read_tensor(
            cli.overlay_base
                .as_ref()
                .expect("overlay base checked above"),
        )?)
    } else {
        None
    };

    if let (Some(base), Some(path)) = (overlay_base.as_ref(), cli.overlay_out.as_ref()) {
        let overlay = output
            .overlay(base, cli.overlay_alpha)
            .map_err(|err| Box::new(err) as DynError)?;
        write_tensor_json(&overlay, path)?;
    }

    if let (Some(base), Some(path)) = (overlay_base.as_ref(), cli.gated_overlay_out.as_ref()) {
        let threshold = resolve_focus_threshold(cli).unwrap_or(0.5);
        let overlay = output
            .gated_overlay(base, threshold, cli.overlay_alpha)
            .map_err(|err| Box::new(err) as DynError)?;
        write_tensor_json(&overlay, path)?;
    }

    if let Some(path) = cli.stats_out.as_ref() {
        if let Some(stats) = statistics.as_ref() {
            write_statistics(stats, path)?;
        }
    }

    if let Some(mask_path) = cli.focus_mask_out.as_ref() {
        let threshold = resolve_focus_threshold(cli).unwrap_or(0.5);
        let mask = output
            .focus_mask(threshold)
            .map_err(|err| Box::new(err) as DynError)?;
        write_tensor_json(&mask, mask_path)?;
    }

    Ok(())
}

fn apply_post_processing(cli: &Cli, output: AttributionOutput) -> Result<AttributionOutput> {
    if let Some(threshold) = resolve_focus_threshold(cli) {
        if !threshold.is_finite() {
            return Err(Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "--focus-threshold must be a finite number",
            )));
        }
    }

    let mut output = output;

    if let Some(kernel) = cli.smooth_kernel {
        if kernel == 0 || kernel % 2 == 0 {
            return Err(Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "--smooth-kernel must be an odd, positive integer",
            )));
        }
        let mut smoothed = output
            .smoothed(kernel)
            .map_err(|err| Box::new(err) as DynError)?;
        smoothed
            .metadata
            .insert_extra_number("smooth_kernel", kernel as f64);
        output = smoothed;
    }

    if cli.normalise_output {
        let mut normalised = output
            .normalised()
            .map_err(|err| Box::new(err) as DynError)?;
        normalised
            .metadata
            .insert_extra_flag("normalised_output", true);
        output = normalised;
    }

    if let Some(threshold) = resolve_focus_threshold(cli) {
        output
            .metadata
            .insert_extra_number("focus_threshold", threshold as f64);
    }

    Ok(output)
}

fn resolve_focus_threshold(cli: &Cli) -> Option<f32> {
    cli.focus_threshold.or_else(|| {
        if cli.focus_mask_out.is_some() || cli.gated_overlay_out.is_some() {
            Some(0.5)
        } else {
            None
        }
    })
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

fn write_tensor_json(tensor: &Tensor, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let disk = DiskTensor::from_tensor(tensor);
    let payload = serde_json::to_string_pretty(&disk)?;
    fs::write(path, payload)?;
    Ok(())
}

#[derive(Serialize)]
struct StatisticsFile {
    min: f32,
    max: f32,
    mean: f32,
    entropy: f32,
}

impl From<&AttributionStatistics> for StatisticsFile {
    fn from(stats: &AttributionStatistics) -> Self {
        Self {
            min: stats.min,
            max: stats.max,
            mean: stats.mean,
            entropy: stats.entropy,
        }
    }
}

fn write_statistics(stats: &AttributionStatistics, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let payload = serde_json::to_string_pretty(&StatisticsFile::from(stats))?;
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
