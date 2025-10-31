mod audit;

use audit::{introspect_bundles, AuditBundle, AuditIntrospection, AuditTrail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueHint};
use serde::{Deserialize, Serialize};
use serde_json::json;
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

    /// Write a structured audit trail describing the CLI execution plan
    #[arg(long, global = true, value_hint = ValueHint::FilePath)]
    audit_out: Option<PathBuf>,

    /// Embed an audit summary and self-check results into the attribution metadata
    #[arg(long, global = true)]
    embed_audit_summary: bool,

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

    /// Review an existing audit bundle and verify its self-checks
    AuditReview(AuditReviewArgs),

    /// Introspect one or more audit bundles and surface structural anomalies
    AuditIntrospect(AuditIntrospectArgs),
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

#[derive(Args)]
struct AuditReviewArgs {
    /// Path to the audit bundle JSON file to review
    #[arg(long, value_hint = ValueHint::FilePath)]
    input: PathBuf,

    /// Optional destination to write the review report as JSON (defaults to STDOUT)
    #[arg(long, value_hint = ValueHint::FilePath)]
    output: Option<PathBuf>,

    /// Pretty-print the review JSON when writing to STDOUT or disk
    #[arg(long)]
    pretty: bool,
}

#[derive(Args)]
struct AuditIntrospectArgs {
    /// Path(s) to the audit bundle JSON files to introspect
    #[arg(long, value_hint = ValueHint::FilePath, num_args = 1..)]
    input: Vec<PathBuf>,

    /// Optional destination to write the introspection report as JSON (defaults to STDOUT)
    #[arg(long, value_hint = ValueHint::FilePath)]
    output: Option<PathBuf>,

    /// Pretty-print the introspection JSON when writing to STDOUT or disk
    #[arg(long)]
    pretty: bool,

    /// Include per-bundle introspection entries in the output alongside the aggregate view
    #[arg(long)]
    per_bundle: bool,
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let cli = Cli::parse();
    let mut audit = AuditTrail::new();
    audit.record_with_value(
        "cli.parsed",
        json!({
            "include_stats": cli.include_stats,
            "print_stats": cli.print_stats,
            "metadata_out": cli.metadata_out.is_some(),
            "heatmap_out": cli.heatmap_out.is_some(),
            "stats_out": cli.stats_out.is_some(),
            "overlay_base": cli.overlay_base.is_some(),
            "overlay_out": cli.overlay_out.is_some(),
            "gated_overlay_out": cli.gated_overlay_out.is_some(),
            "focus_mask_out": cli.focus_mask_out.is_some(),
            "smooth_kernel": cli.smooth_kernel,
            "normalise_output": cli.normalise_output,
            "focus_threshold": cli.focus_threshold,
            "audit_out": cli.audit_out.is_some(),
            "embed_audit_summary": cli.embed_audit_summary,
        }),
    );

    if let Some(path) = cli.audit_out.as_ref() {
        audit.record_with_value("io.write.audit", json!({ "path": path }));
    }

    match &cli.command {
        Command::GradCam(args) => {
            audit.record_with_value(
                "cli.command",
                json!({
                    "name": "grad-cam",
                    "layer": args.layer,
                    "height": args.height,
                    "width": args.width,
                }),
            );
            let output = run_grad_cam(args, &mut audit)?;
            finalise_output(&cli, output, &args.output, &mut audit)?;
        }
        Command::IntegratedGradients(args) => {
            audit.record_with_value(
                "cli.command",
                json!({
                    "name": "integrated-gradients",
                    "layer": args.layer,
                    "steps": args.steps,
                    "target": args.target,
                }),
            );
            let output = run_integrated_gradients_cli(args, &mut audit)?;
            finalise_output(&cli, output, &args.output, &mut audit)?;
        }
        Command::AuditReview(args) => {
            audit.record_with_value(
                "cli.command",
                json!({
                    "name": "audit-review",
                    "input": args.input,
                    "output": args.output,
                    "pretty": args.pretty,
                }),
            );
            run_audit_review(args, &mut audit)?;
        }
        Command::AuditIntrospect(args) => {
            audit.record_with_value(
                "cli.command",
                json!({
                    "name": "audit-introspect",
                    "inputs": args.input,
                    "output": args.output,
                    "pretty": args.pretty,
                    "per_bundle": args.per_bundle,
                }),
            );
            run_audit_introspect(args, &mut audit)?;
        }
    }

    let bundle = audit.finish();

    if let Some(path) = cli.audit_out.as_ref() {
        write_audit_report(&bundle, path)?;
    }

    Ok(())
}

fn run_audit_review(args: &AuditReviewArgs, audit: &mut AuditTrail) -> Result<()> {
    let bundle = read_audit_bundle(&args.input, audit, "audit.review.bundle_loaded")?;
    let review = audit::review_bundle(&bundle);
    audit.record_with_value(
        "audit.review.summary",
        json!({
            "matches": review.summary_matches,
            "issues": review.issues.len(),
        }),
    );

    let payload = if args.pretty {
        serde_json::to_string_pretty(&review)?
    } else {
        serde_json::to_string(&review)?
    };

    if let Some(path) = args.output.as_ref() {
        ensure_parent_dir(path)?;
        fs::write(path, &payload)?;
        audit.record_with_value("io.write.audit_review", json!({ "path": path }));
    } else {
        println!("{payload}");
        audit.record("audit.review.stdout_emitted");
    }

    Ok(())
}

fn run_audit_introspect(args: &AuditIntrospectArgs, audit: &mut AuditTrail) -> Result<()> {
    let mut bundles = Vec::new();
    let mut per_bundle = Vec::new();

    for path in &args.input {
        let bundle = read_audit_bundle(path, audit, "audit.introspect.bundle_loaded")?;
        let introspection = audit::introspect_bundle(&bundle);
        audit.record_with_value(
            "audit.introspect.bundle",
            json!({
                "path": path,
                "events": introspection.total_events,
                "unique_stages": introspection.unique_stages,
                "anomalies": introspection.anomalies.len(),
            }),
        );
        per_bundle.push((path.display().to_string(), introspection.clone()));
        bundles.push(bundle);
    }

    let aggregated = introspect_bundles(&bundles);
    audit.record_with_value(
        "audit.introspect.summary",
        json!({
            "bundles": bundles.len(),
            "unique_stages": aggregated.unique_stages,
            "entropy": aggregated.entropy,
            "anomalies": aggregated.anomalies.len(),
        }),
    );

    let mut report = AuditIntrospectReportFile {
        bundles: bundles.len(),
        aggregated,
        per_bundle: Vec::new(),
    };

    if args.per_bundle {
        report.per_bundle = per_bundle
            .into_iter()
            .map(|(label, introspection)| AuditIntrospectEntry {
                label,
                introspection,
            })
            .collect();
    }

    let payload = if args.pretty {
        serde_json::to_string_pretty(&report)?
    } else {
        serde_json::to_string(&report)?
    };

    if let Some(path) = args.output.as_ref() {
        ensure_parent_dir(path)?;
        fs::write(path, &payload)?;
        audit.record_with_value("io.write.audit_introspect", json!({ "path": path }));
    } else {
        println!("{payload}");
        audit.record("audit.introspect.stdout_emitted");
    }

    Ok(())
}

fn finalise_output(
    cli: &Cli,
    mut output: AttributionOutput,
    destination: &Path,
    audit: &mut AuditTrail,
) -> Result<()> {
    output = apply_post_processing(cli, output, audit)?;

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
        let stats = output.statistics();
        audit.record_with_value(
            "statistics.computed",
            json!({
                "min": stats.min,
                "max": stats.max,
                "mean": stats.mean,
                "entropy": stats.entropy,
            }),
        );
        Some(stats)
    } else {
        None
    };

    if let Some(stats) = statistics.as_ref() {
        if cli.include_stats {
            attach_statistics(&mut output.metadata, stats);
            audit.record("metadata.statistics_embedded");
        }
        if cli.print_stats {
            print_statistics(stats, &output.metadata);
            audit.record("statistics.printed");
        }
    }

    let overlay_base = if overlay_requested {
        audit.record("overlay.requested");
        Some(read_tensor(
            audit,
            "io.read.overlay_base",
            cli.overlay_base
                .as_ref()
                .expect("overlay base checked above"),
        )?)
    } else {
        None
    };

    let overlay_tensor =
        if let (Some(base), Some(path)) = (overlay_base.as_ref(), cli.overlay_out.as_ref()) {
            let overlay = output
                .overlay(base, cli.overlay_alpha)
                .map_err(|err| Box::new(err) as DynError)?;
            audit.record_with_value(
                "overlay.generated",
                json!({ "path": path, "alpha": cli.overlay_alpha }),
            );
            Some((overlay, path.clone()))
        } else {
            None
        };

    let gated_overlay_tensor =
        if let (Some(base), Some(path)) = (overlay_base.as_ref(), cli.gated_overlay_out.as_ref()) {
            let threshold = resolve_focus_threshold(cli).unwrap_or(0.5);
            let overlay = output
                .gated_overlay(base, threshold, cli.overlay_alpha)
                .map_err(|err| Box::new(err) as DynError)?;
            audit.record_with_value(
                "overlay.gated_generated",
                json!({
                    "path": path,
                    "alpha": cli.overlay_alpha,
                    "threshold": threshold,
                }),
            );
            Some((overlay, path.clone()))
        } else {
            None
        };

    let focus_mask = if let Some(mask_path) = cli.focus_mask_out.as_ref() {
        let threshold = resolve_focus_threshold(cli).unwrap_or(0.5);
        audit.record("focus_mask.requested");
        let mask = output
            .focus_mask(threshold)
            .map_err(|err| Box::new(err) as DynError)?;
        audit.record_with_value(
            "focus_mask.generated",
            json!({ "threshold": threshold, "path": mask_path }),
        );
        Some((mask, mask_path.clone()))
    } else {
        None
    };

    if statistics.is_some() {
        if let Some(path) = cli.stats_out.as_ref() {
            audit.record_with_value("io.write.stats", json!({ "path": path }));
        }
    }

    if let Some(path) = cli.heatmap_out.as_ref() {
        audit.record_with_value("io.write.heatmap", json!({ "path": path }));
    }

    if let Some(path) = cli.metadata_out.as_ref() {
        audit.record_with_value("io.write.metadata", json!({ "path": path }));
    }

    if let Some((_, path)) = overlay_tensor.as_ref() {
        audit.record_with_value("io.write.overlay", json!({ "path": path }));
    }

    if let Some((_, path)) = gated_overlay_tensor.as_ref() {
        audit.record_with_value("io.write.gated_overlay", json!({ "path": path }));
    }

    if let Some((_, path)) = focus_mask.as_ref() {
        audit.record_with_value("io.write.focus_mask", json!({ "path": path }));
    }

    audit.record_with_value("io.write.report", json!({ "path": destination }));

    let review = audit.review();
    if cli.embed_audit_summary {
        output
            .metadata
            .insert_extra("audit_summary", serde_json::to_value(&review.summary)?);
        output.metadata.insert_extra(
            "audit_self_checks",
            serde_json::to_value(&review.self_checks)?,
        );
        audit.record_with_value(
            "metadata.audit_summary_embedded",
            json!({
                "total_events": review.summary.total_events,
                "checks": review.self_checks.len(),
            }),
        );
    }

    let report = output.to_report();
    write_report(&report, destination)?;

    if let Some(path) = cli.metadata_out.as_ref() {
        write_metadata(&output.metadata, path)?;
    }

    if let Some(path) = cli.heatmap_out.as_ref() {
        write_tensor_json(&output.map, path)?;
    }

    if let Some((overlay, path)) = overlay_tensor.as_ref() {
        write_tensor_json(overlay, path)?;
    }

    if let Some((overlay, path)) = gated_overlay_tensor.as_ref() {
        write_tensor_json(overlay, path)?;
    }

    if let Some(stats) = statistics.as_ref() {
        if let Some(path) = cli.stats_out.as_ref() {
            write_statistics(stats, path)?;
        }
    }

    if let Some((mask, path)) = focus_mask.as_ref() {
        write_tensor_json(mask, path)?;
    }

    audit.record("finalise.completed");

    Ok(())
}

fn apply_post_processing(
    cli: &Cli,
    output: AttributionOutput,
    audit: &mut AuditTrail,
) -> Result<AttributionOutput> {
    if let Some(threshold) = resolve_focus_threshold(cli) {
        if !threshold.is_finite() {
            return Err(Box::new(io::Error::new(
                ErrorKind::InvalidInput,
                "--focus-threshold must be a finite number",
            )));
        }
        audit.record_with_value("postprocess.focus_threshold", json!({ "value": threshold }));
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
        audit.record_with_value("postprocess.smoothed", json!({ "kernel": kernel }));
    }

    if cli.normalise_output {
        let mut normalised = output
            .normalised()
            .map_err(|err| Box::new(err) as DynError)?;
        normalised
            .metadata
            .insert_extra_flag("normalised_output", true);
        output = normalised;
        audit.record("postprocess.normalised");
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

fn run_grad_cam(args: &GradCamArgs, audit: &mut AuditTrail) -> Result<AttributionOutput> {
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
    let activations = read_tensor(audit, "io.read.activations", &args.activations)?;
    let gradients = read_tensor(audit, "io.read.gradients", &args.gradients)?;
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
    audit.record_with_value(
        "grad_cam.completed",
        json!({
            "height": args.height,
            "width": args.width,
            "apply_relu": config.apply_relu,
        }),
    );
    Ok(AttributionOutput::new(heatmap, metadata))
}

fn run_integrated_gradients_cli(
    args: &IntegratedGradientsArgs,
    audit: &mut AuditTrail,
) -> Result<AttributionOutput> {
    if args.steps == 0 {
        return Err(Box::new(io::Error::new(
            ErrorKind::InvalidInput,
            "--steps must be greater than zero",
        )));
    }
    let input = read_tensor(audit, "io.read.input", &args.input)?;
    let baseline = read_tensor(audit, "io.read.baseline", &args.baseline)?;
    let weights = match &args.weights {
        Some(path) => Some(read_tensor(audit, "io.read.weights", path)?),
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

    audit.record_with_value(
        "integrated_gradients.completed",
        json!({
            "steps": args.steps,
            "target": args.target,
            "target_label": args.target_label,
        }),
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

fn read_tensor(audit: &mut AuditTrail, stage: &str, path: &Path) -> Result<Tensor> {
    let contents = fs::read_to_string(path)?;
    let disk: DiskTensor = serde_json::from_str(&contents)?;
    let tensor = disk.into_tensor()?;
    let (rows, cols) = tensor.shape();
    audit.record_with_value(
        stage,
        json!({
            "path": path,
            "rows": rows,
            "cols": cols,
        }),
    );
    Ok(tensor)
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

#[derive(Serialize)]
struct AuditIntrospectEntry {
    label: String,
    introspection: AuditIntrospection,
}

#[derive(Serialize)]
struct AuditIntrospectReportFile {
    bundles: usize,
    aggregated: AuditIntrospection,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    per_bundle: Vec<AuditIntrospectEntry>,
}

fn write_statistics(stats: &AttributionStatistics, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let payload = serde_json::to_string_pretty(&StatisticsFile::from(stats))?;
    fs::write(path, payload)?;
    Ok(())
}

fn write_audit_report(bundle: &AuditBundle, path: &Path) -> Result<()> {
    ensure_parent_dir(path)?;
    let payload = serde_json::to_string_pretty(bundle)?;
    fs::write(path, payload)?;
    Ok(())
}

fn read_audit_bundle(path: &Path, audit: &mut AuditTrail, stage: &str) -> Result<AuditBundle> {
    audit.record_with_value("io.read.audit", json!({ "path": path }));
    let contents = fs::read_to_string(path)?;
    let bundle: AuditBundle = serde_json::from_str(&contents)?;
    audit.record_with_value(
        stage,
        json!({
            "events": bundle.events.len(),
            "recorded_self_checks": bundle.self_checks.len(),
        }),
    );
    Ok(bundle)
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
