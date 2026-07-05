import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = Path("/tmp/spiraltorch-profile-smoke")
BYTE_LM_VOCAB = 256
BYTE_LM_HIDDEN = 24
TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS = {
    "transformers",
    "torch-transformers",
    "hf-runtime",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny local byte-LM profile/promotion smoke: write a "
            "byte-compatible HF state dict, sweep it, emit a source profile, "
            "run the profile lane, and compare/promote the resulting summary."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for checkpoint, summary, aggregate, profile, and promotion JSONL files.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child example scripts.",
    )
    parser.add_argument(
        "--source-label",
        default="local-smoke",
        help="checkpoint_source_label used throughout the smoke.",
    )
    parser.add_argument(
        "--hf-state-dict",
        type=Path,
        default=None,
        help=(
            "Optional local HF/PyTorch state dict file or directory to use as "
            "the smoke source instead of writing a tiny checkpoint fixture."
        ),
    )
    parser.add_argument(
        "--key-preset",
        default="llama",
        help=(
            "HF key preset passed to checkpoint preflight and sweep.py. Use "
            "'auto' with --hf-state-dict to infer the checkpoint layout."
        ),
    )
    parser.add_argument(
        "--include-extra-key",
        dest="include_extra_keys",
        action="append",
        default=[],
        help="Additional external HF state-dict key to include in checkpoint audit rows.",
    )
    parser.add_argument(
        "--no-synthesize-missing-biases",
        action="store_true",
        help="Require embed/head bias tensors to exist in the HF checkpoint.",
    )
    parser.add_argument(
        "--allow-overlap-resize",
        action="store_true",
        help=(
            "Allow overlap-copy/zero-fill adaptation from arbitrary HF shapes "
            "into the bounded byte-LM smoke shape."
        ),
    )
    parser.add_argument(
        "--checkpoint-projection",
        choices=["none", "zspace"],
        default="none",
        help="Optional checkpoint projection policy shared by preflight and sweep.",
    )
    parser.add_argument(
        "--checkpoint-projection-preset",
        choices=["healthy"],
        default=None,
        help="Shortcut projection preset shared by preflight and sweep.",
    )
    parser.add_argument(
        "--checkpoint-projection-strength",
        type=float,
        default=None,
        help="Projection strength forwarded when --checkpoint-projection zspace is used.",
    )
    parser.add_argument(
        "--checkpoint-projection-curvature",
        type=float,
        default=None,
        help="Projection curvature forwarded when --checkpoint-projection zspace is used.",
    )
    parser.add_argument(
        "--checkpoint-projection-frequency",
        type=float,
        default=None,
        help="Projection language-wave frequency forwarded when projection is used.",
    )
    parser.add_argument(
        "--checkpoint-source-gain",
        type=float,
        default=None,
        help="Optional positive checkpoint source gain shared by preflight and sweep.",
    )
    parser.add_argument(
        "--transformers-audit",
        action="store_true",
        help=(
            "Forward an optional Transformers config/tokenizer runtime audit to "
            "checkpoint_preflight.py without making Transformers a hard dependency."
        ),
    )
    parser.add_argument(
        "--transformers-model-path",
        type=Path,
        default=None,
        help=(
            "Local Transformers model/config/tokenizer directory for the audit. "
            "Defaults to the HF state-dict directory or parent."
        ),
    )
    parser.add_argument(
        "--transformers-revision",
        default=None,
        help="Optional Transformers revision forwarded to checkpoint_preflight.py.",
    )
    parser.add_argument(
        "--allow-transformers-remote",
        action="store_true",
        help="Allow the Transformers audit to resolve non-local files.",
    )
    parser.add_argument(
        "--transformers-trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to the Transformers audit.",
    )
    parser.add_argument(
        "--skip-transformers-tokenizer",
        action="store_true",
        help="Only audit AutoConfig metadata when --transformers-audit is set.",
    )
    parser.add_argument(
        "--transformers-load-model",
        action="store_true",
        help="Also instantiate AutoModelForCausalLM during the optional audit.",
    )
    parser.add_argument(
        "--require-transformers-audit",
        action="store_true",
        help="Fail checkpoint preflight when the optional Transformers audit is not clean.",
    )
    parser.add_argument(
        "--transformers-trace",
        action="store_true",
        help=(
            "Run byte_lm_transformers_trace.py against the same local "
            "Transformers runtime before the FT smoke."
        ),
    )
    parser.add_argument(
        "--transformers-trace-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for the Transformers prompt trace. Defaults "
            "to --out-dir/transformers-trace.jsonl when --transformers-trace is set."
        ),
    )
    parser.add_argument(
        "--compare-transformers-trace-jsonl",
        type=Path,
        default=None,
        help="Optional baseline Transformers prompt trace JSONL to compare.",
    )
    parser.add_argument(
        "--transformers-trace-compare-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for Transformers trace comparison/gate rows. "
            "Defaults to --out-dir/transformers-trace-compare.jsonl when needed."
        ),
    )
    parser.add_argument(
        "--transformers-trace-prompt",
        dest="transformers_trace_prompts",
        action="append",
        default=[],
        help=(
            "Prompt forwarded to byte_lm_transformers_trace.py. May be repeated; "
            "the trace script supplies a default when omitted."
        ),
    )
    parser.add_argument(
        "--transformers-trace-prompt-file",
        type=Path,
        default=None,
        help="Optional prompt file forwarded to byte_lm_transformers_trace.py.",
    )
    parser.add_argument(
        "--transformers-trace-top-k",
        type=int,
        default=5,
        help="Top-k next-token candidates kept by byte_lm_transformers_trace.py.",
    )
    parser.add_argument(
        "--transformers-trace-zspace-project",
        action="store_true",
        help="Ask byte_lm_transformers_trace.py to attach Z-space projection metrics.",
    )
    parser.add_argument(
        "--transformers-trace-zspace-source",
        choices=["hidden", "top_logits"],
        default="hidden",
        help="Trace vector source used when --transformers-trace-zspace-project is set.",
    )
    parser.add_argument(
        "--transformers-trace-runtime-import",
        dest="transformers_trace_runtime_imports",
        action="append",
        default=[],
        help=(
            "Additional Python module imported by byte_lm_transformers_trace.py "
            "while SpiralTorch and Transformers are loaded. May be repeated."
        ),
    )
    parser.add_argument(
        "--transformers-trace-runtime-import-preset",
        dest="transformers_trace_runtime_import_presets",
        action="append",
        choices=sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Named byte_lm_transformers_trace.py runtime import bundle to "
            "probe. 'torch-transformers' probes both modules; 'hf-runtime' "
            "also probes tokenizers. May be repeated."
        ),
    )
    parser.add_argument(
        "--require-transformers-trace-runtime-imports",
        action="store_true",
        help="Fail the Transformers trace when any runtime import probe fails.",
    )
    parser.add_argument(
        "--require-transformers-trace-match",
        action="store_true",
        help="Fail when current Transformers trace scope differs from the baseline.",
    )
    parser.add_argument(
        "--require-transformers-trace-runtime-metadata-match",
        action="store_true",
        help=(
            "Fail when current Transformers trace runtime metadata differs "
            "from the baseline trace manifest."
        ),
    )
    parser.add_argument(
        "--require-transformers-trace-top-token-match",
        action="store_true",
        help="Fail when a traced prompt's top token differs from the baseline.",
    )
    parser.add_argument(
        "--transformers-trace-max-top-logit-regression",
        type=float,
        default=None,
        help="Maximum allowed top-1 logit regression versus the baseline trace.",
    )
    parser.add_argument(
        "--transformers-trace-max-top-probability-regression",
        type=float,
        default=None,
        help="Maximum allowed top-1 probability regression versus the baseline trace.",
    )
    parser.add_argument(
        "--transformers-trace-max-logit-l2-change",
        type=float,
        default=None,
        help="Maximum allowed prompt logit L2 change versus the baseline trace.",
    )
    parser.add_argument(
        "--transformers-trace-max-hidden-state-l2-change",
        type=float,
        default=None,
        help="Maximum allowed hidden-state L2 change versus the baseline trace.",
    )
    parser.add_argument(
        "--transformers-trace-require-zspace-status",
        default=None,
        help="Required zspace_projection_status for current prompt trace rows.",
    )
    parser.add_argument(
        "--skip-checkpoint-shape-audit",
        action="store_true",
        help="Skip the checkpoint_preflight.py --shape-only audit before sweep.",
    )
    parser.add_argument(
        "--checkpoint-shape-audit-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for the shape-only checkpoint audit row.",
    )
    parser.add_argument(
        "--shape-audit-require-exact-shape-match",
        action="store_true",
        help="Require exact HF shape match during the shape-only audit.",
    )
    parser.add_argument(
        "--shape-audit-require-detected-key-preset",
        default=None,
        help="Require the shape-only auto-detected key preset to match this value.",
    )
    parser.add_argument(
        "--skip-checkpoint-preflight",
        action="store_true",
        help="Skip the normal checkpoint preflight/load audit before sweep.",
    )
    parser.add_argument(
        "--checkpoint-preflight-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for normal checkpoint preflight rows.",
    )
    parser.add_argument(
        "--compare-checkpoint-preflight-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional baseline checkpoint preflight JSONL to compare before "
            "the sweep and promoted ladder run."
        ),
    )
    parser.add_argument(
        "--require-checkpoint-preflight-match",
        action="store_true",
        help=(
            "Fail before sweep when --compare-checkpoint-preflight-jsonl differs "
            "from the current checkpoint preflight rows."
        ),
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=[],
        help="Case label to run. Defaults to adapter_ja. May be repeated.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="LoRA config to run. Defaults to r12_a64_lr4. May be repeated.",
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        default=[],
        help="Profile lane to emit/run. Defaults to strong_effect. May be repeated.",
    )
    parser.add_argument(
        "--ft-epochs",
        type=int,
        default=1,
        help="Sparse FT epochs for the diagnostic smoke. Defaults to 1.",
    )
    parser.add_argument(
        "--promotion-metric",
        default="target_loss_delta_mean",
        help="Profile-run metric used for smoke promotion ranking.",
    )
    parser.add_argument(
        "--skip-promoted-follow-up",
        action="store_true",
        help=(
            "Stop after the first profile run and promotion compare. By default "
            "the smoke feeds the promotion JSONL back into profile_runner and "
            "runs the promoted lane once more."
        ),
    )
    parser.add_argument(
        "--promoted-output-prefix",
        default="profile-smoke-promoted",
        help="Filename prefix for the promoted follow-up profile run outputs.",
    )
    parser.add_argument(
        "--promoted-ft-epochs",
        type=int,
        default=None,
        help=(
            "FT epochs for the first promoted follow-up rung. Defaults to one "
            "more than --ft-epochs."
        ),
    )
    parser.add_argument(
        "--promoted-rungs",
        type=int,
        default=1,
        help=(
            "Number of promoted follow-up rungs to run. Defaults to 1; use 0 "
            "or --skip-promoted-follow-up to stop after the first profile run."
        ),
    )
    parser.add_argument(
        "--promoted-ft-epochs-step",
        type=int,
        default=1,
        help=(
            "FT epoch increment between promoted rungs. Defaults to 1, so the "
            "default ladder is --ft-epochs+1, +2, ..."
        ),
    )
    parser.add_argument(
        "--promoted-rungs-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL manifest for promoted rung artifacts. Defaults to "
            "--out-dir/promoted-rungs.jsonl when promoted rungs run."
        ),
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional top-level JSONL manifest for all profile smoke artifacts. "
            "Defaults to --out-dir/profile-smoke-manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--continue-manifest-jsonl",
        type=Path,
        default=None,
        help=(
            "Continue promoted rungs from a previous profile-smoke-manifest.jsonl "
            "without rerunning checkpoint preflight, sweep, or source compare."
        ),
    )
    parser.add_argument(
        "--validate-manifest-jsonl",
        type=Path,
        default=None,
        help=(
            "Validate a previous profile-smoke-manifest.jsonl plus its "
            "promoted-rungs chain without launching any new commands."
        ),
    )
    parser.add_argument(
        "--manifest-validation-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL report path for --validate-manifest-jsonl or "
            "--validate-produced-manifest. Writes one "
            "profile_smoke_manifest_validation row after validation passes."
        ),
    )
    parser.add_argument(
        "--validate-produced-manifest",
        action="store_true",
        help=(
            "After a normal profile smoke run or manifest continuation, validate "
            "the manifest produced by this invocation and apply any manifest trace "
            "validation gates."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace",
        action="store_true",
        help="Fail --validate-manifest-jsonl when the manifest has no Transformers trace artifact.",
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-compare-pass",
        action="store_true",
        help=(
            "Fail --validate-manifest-jsonl when the manifest's Transformers "
            "trace compare summary is missing or did not pass."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-runtime-metadata-match",
        action="store_true",
        help=(
            "Fail --validate-manifest-jsonl when the manifest's Transformers "
            "trace compare summary reports config/tokenizer/model metadata drift."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-coimport",
        action="store_true",
        help=(
            "Fail --validate-manifest-jsonl when the current Transformers trace "
            "manifest does not report a clean SpiralTorch/Transformers co-import."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-runtime-imports",
        action="store_true",
        help=(
            "Fail --validate-manifest-jsonl when the current Transformers trace "
            "manifest has missing or failed runtime import probes."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-runtime-import",
        dest="require_manifest_transformers_trace_runtime_import",
        action="append",
        default=[],
        help=(
            "Fail --validate-manifest-jsonl unless this module appears in the "
            "Transformers trace manifest's successful runtime import probes. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--require-manifest-transformers-trace-runtime-import-preset",
        dest="require_manifest_transformers_trace_runtime_import_preset",
        action="append",
        choices=sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Fail --validate-manifest-jsonl unless this named runtime import "
            "preset appears in the Transformers trace manifest. May be repeated."
        ),
    )
    parser.add_argument(
        "--max-manifest-transformers-trace-top-token-changed-rows",
        type=int,
        default=None,
        help=(
            "Fail --validate-manifest-jsonl when the trace compare summary "
            "reports more top-token drift rows than this value."
        ),
    )
    parser.add_argument(
        "--max-manifest-transformers-trace-top-probability-regression",
        type=float,
        default=None,
        help=(
            "Fail --validate-manifest-jsonl when observed max top-probability "
            "regression exceeds this value."
        ),
    )
    parser.add_argument(
        "--continue-manifest-output-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional manifest output path for --continue-manifest-jsonl. "
            "Defaults to updating the input manifest path."
        ),
    )
    parser.add_argument(
        "--continue-plan-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL plan for --continue-manifest-jsonl. Writes the "
            "planned promoted rung artifact/epoch chain before child commands run."
        ),
    )
    parser.add_argument(
        "--continue-rungs",
        type=int,
        default=1,
        help="Number of additional promoted rungs to run from --continue-manifest-jsonl.",
    )
    parser.add_argument(
        "--continue-ft-epochs",
        type=int,
        default=None,
        help="FT epochs for the first continued promoted rung.",
    )
    parser.add_argument(
        "--continue-ft-epochs-step",
        type=int,
        default=None,
        help="FT epoch increment between continued promoted rungs.",
    )
    parser.add_argument(
        "--continue-promoted-output-prefix",
        default=None,
        help="Override promoted output prefix while continuing from a manifest.",
    )
    parser.add_argument(
        "--strict-aggregate-gates",
        action="store_true",
        help=(
            "Keep profile_runner's default aggregate accepted/movement gates. "
            "By default this smoke disables them so guard-rejected diagnostic "
            "runs can still exercise run-summary and promotion gates."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing output files in --out-dir instead of cleaning this smoke's artifacts first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the child commands without executing them.",
    )
    args = parser.parse_args()
    if args.ft_epochs <= 0:
        parser.error("--ft-epochs must be positive")
    if args.promoted_ft_epochs is not None and args.promoted_ft_epochs <= 0:
        parser.error("--promoted-ft-epochs must be positive")
    if args.promoted_rungs < 0:
        parser.error("--promoted-rungs must be non-negative")
    if args.promoted_ft_epochs_step <= 0:
        parser.error("--promoted-ft-epochs-step must be positive")
    if args.continue_rungs <= 0:
        parser.error("--continue-rungs must be positive")
    if args.continue_ft_epochs is not None and args.continue_ft_epochs <= 0:
        parser.error("--continue-ft-epochs must be positive")
    if args.continue_ft_epochs_step is not None and args.continue_ft_epochs_step <= 0:
        parser.error("--continue-ft-epochs-step must be positive")
    if (
        args.continue_manifest_jsonl is None
        and args.validate_manifest_jsonl is None
        and args.hf_state_dict is None
        and args.key_preset == "auto"
    ):
        parser.error("--key-preset auto requires --hf-state-dict")
    if (
        args.continue_manifest_jsonl is not None
        and args.validate_manifest_jsonl is not None
    ):
        parser.error(
            "--continue-manifest-jsonl and --validate-manifest-jsonl are mutually exclusive"
        )
    if args.validate_produced_manifest and args.validate_manifest_jsonl is not None:
        parser.error(
            "--validate-produced-manifest is only valid for normal profile smoke runs"
        )
    if args.validate_produced_manifest and args.dry_run:
        parser.error("--validate-produced-manifest cannot be used with --dry-run")
    if (
        args.manifest_validation_jsonl is not None
        and args.validate_manifest_jsonl is None
        and not args.validate_produced_manifest
    ):
        parser.error(
            "--manifest-validation-jsonl requires --validate-manifest-jsonl "
            "or --validate-produced-manifest"
        )
    manifest_trace_validation_requested = any(
        [
            args.require_manifest_transformers_trace,
            args.require_manifest_transformers_trace_compare_pass,
            args.require_manifest_transformers_trace_runtime_metadata_match,
            args.require_manifest_transformers_trace_coimport,
            args.require_manifest_transformers_trace_runtime_imports,
            bool(args.require_manifest_transformers_trace_runtime_import),
            bool(args.require_manifest_transformers_trace_runtime_import_preset),
            args.max_manifest_transformers_trace_top_token_changed_rows is not None,
            args.max_manifest_transformers_trace_top_probability_regression
            is not None,
        ]
    )
    if (
        manifest_trace_validation_requested
        and args.validate_manifest_jsonl is None
        and not args.validate_produced_manifest
    ):
        parser.error(
            "manifest trace validation gates require --validate-manifest-jsonl "
            "or --validate-produced-manifest"
        )
    if (
        args.max_manifest_transformers_trace_top_token_changed_rows is not None
        and args.max_manifest_transformers_trace_top_token_changed_rows < 0
    ):
        parser.error(
            "--max-manifest-transformers-trace-top-token-changed-rows "
            "must be non-negative"
        )
    if (
        args.max_manifest_transformers_trace_top_probability_regression is not None
        and args.max_manifest_transformers_trace_top_probability_regression < 0.0
    ):
        parser.error(
            "--max-manifest-transformers-trace-top-probability-regression "
            "must be non-negative"
        )
    if args.continue_plan_jsonl is not None and args.continue_manifest_jsonl is None:
        parser.error("--continue-plan-jsonl requires --continue-manifest-jsonl")
    if args.checkpoint_source_gain is not None and args.checkpoint_source_gain <= 0.0:
        parser.error("--checkpoint-source-gain must be positive")
    if args.require_transformers_audit and not args.transformers_audit:
        parser.error("--require-transformers-audit requires --transformers-audit")
    trace_requested_by_arg = any(
        [
            args.transformers_trace_jsonl is not None,
            args.compare_transformers_trace_jsonl is not None,
            args.transformers_trace_compare_jsonl is not None,
            bool(args.transformers_trace_prompts),
            args.transformers_trace_prompt_file is not None,
            args.transformers_trace_zspace_project,
            bool(args.transformers_trace_runtime_imports),
            bool(args.transformers_trace_runtime_import_presets),
            args.require_transformers_trace_runtime_imports,
            args.require_transformers_trace_match,
            args.require_transformers_trace_runtime_metadata_match,
            args.require_transformers_trace_top_token_match,
            args.transformers_trace_max_top_logit_regression is not None,
            args.transformers_trace_max_top_probability_regression is not None,
            args.transformers_trace_max_logit_l2_change is not None,
            args.transformers_trace_max_hidden_state_l2_change is not None,
            args.transformers_trace_require_zspace_status is not None,
        ]
    )
    if trace_requested_by_arg and not args.transformers_trace:
        parser.error("Transformers trace options require --transformers-trace")
    if args.transformers_trace and args.transformers_trace_top_k <= 0:
        parser.error("--transformers-trace-top-k must be positive")
    trace_compare_gates = [
        args.require_transformers_trace_match,
        args.require_transformers_trace_runtime_metadata_match,
        args.require_transformers_trace_top_token_match,
        args.transformers_trace_max_top_logit_regression is not None,
        args.transformers_trace_max_top_probability_regression is not None,
        args.transformers_trace_max_logit_l2_change is not None,
        args.transformers_trace_max_hidden_state_l2_change is not None,
    ]
    if (
        args.require_transformers_trace_runtime_imports
        and not args.transformers_trace_runtime_imports
        and not args.transformers_trace_runtime_import_presets
    ):
        parser.error(
            "--require-transformers-trace-runtime-imports requires "
            "--transformers-trace-runtime-import or "
            "--transformers-trace-runtime-import-preset"
        )
    if any(trace_compare_gates) and args.compare_transformers_trace_jsonl is None:
        parser.error(
            "Transformers trace comparison gates require "
            "--compare-transformers-trace-jsonl"
        )
    if (
        args.transformers_trace_compare_jsonl is not None
        and args.compare_transformers_trace_jsonl is None
        and args.transformers_trace_require_zspace_status is None
    ):
        parser.error(
            "--transformers-trace-compare-jsonl requires "
            "--compare-transformers-trace-jsonl or "
            "--transformers-trace-require-zspace-status"
        )
    for name in [
        "transformers_trace_max_top_logit_regression",
        "transformers_trace_max_top_probability_regression",
        "transformers_trace_max_logit_l2_change",
        "transformers_trace_max_hidden_state_l2_change",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if (
        args.transformers_trace
        and args.transformers_model_path is None
        and args.hf_state_dict is None
    ):
        parser.error(
            "--transformers-trace requires --transformers-model-path when "
            "the smoke writes a generated checkpoint fixture"
        )
    if (
        args.require_checkpoint_preflight_match
        and args.compare_checkpoint_preflight_jsonl is None
    ):
        parser.error(
            "--require-checkpoint-preflight-match requires "
            "--compare-checkpoint-preflight-jsonl"
        )
    if args.skip_checkpoint_preflight and (
        args.compare_checkpoint_preflight_jsonl is not None
        or args.require_checkpoint_preflight_match
    ):
        parser.error("checkpoint preflight compare gates require checkpoint preflight")
    if not args.source_label:
        parser.error("--source-label must be non-empty")
    return args


def run_command(cmd, *, dry_run=False):
    print("profile_smoke_cmd " + shlex.join([str(part) for part in cmd]))
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True)


def clean_previous_outputs(paths, run_dirs):
    for path in paths:
        if path.exists():
            path.unlink()
    for run_dir in run_dirs:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def load_single_jsonl_row_type(path, row_type):
    rows = load_jsonl(path)
    matches = [row for row in rows if row.get("row_type") == row_type]
    if len(matches) != 1:
        raise ValueError(
            f"{path} must contain exactly one {row_type} row; found {len(matches)}"
        )
    return matches[0]


def load_single_profile_smoke_manifest(path):
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one profile_smoke_manifest row")
    row = rows[0]
    validate_profile_smoke_manifest_artifacts(row)
    return row


def optional_path(path):
    return str(path) if path is not None else None


def default_manifest_validation_jsonl(manifest_jsonl):
    path = Path(manifest_jsonl)
    if path.suffix:
        return path.with_name(f"{path.stem}-validation{path.suffix}")
    return path.with_name(f"{path.name}-validation.jsonl")


def produced_manifest_validation_jsonl(args, manifest_jsonl):
    return args.manifest_validation_jsonl or default_manifest_validation_jsonl(
        manifest_jsonl
    )


def required_manifest_artifact_fields(row):
    fields = [
        "out_dir",
        "checkpoint",
        "sweep_jsonl",
        "sweep_aggregate_jsonl",
        "source_compare_jsonl",
        "profile_jsonl",
        "profile_run_dir",
        "run_events_jsonl",
        "run_summary_jsonl",
        "promotion_jsonl",
        "promotion_compare_jsonl",
    ]
    optional_fields = [
        "checkpoint_shape_audit_jsonl",
        "checkpoint_preflight_jsonl",
        "compare_checkpoint_preflight_jsonl",
        "transformers_trace_jsonl",
        "compare_transformers_trace_jsonl",
        "transformers_trace_compare_jsonl",
        "promoted_rungs_jsonl",
        "promoted_final_run_summary_jsonl",
        "promoted_final_promotion_jsonl",
    ]
    for field in optional_fields:
        if row.get(field) is not None:
            fields.append(field)
    return fields


PROMOTED_RUNG_ARTIFACT_FIELDS = [
    "output_dir",
    "commands_jsonl",
    "promotion_selection_jsonl",
    "run_events_jsonl",
    "run_summary_jsonl",
    "promotion_jsonl",
]


def validate_profile_smoke_manifest_artifacts(row):
    if row.get("row_type") != "profile_smoke_manifest":
        raise ValueError(
            f"unsupported profile smoke manifest row_type: {row.get('row_type')!r}"
        )
    missing = []
    for field in required_manifest_artifact_fields(row):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            missing.append(f"{field}=missing")
            continue
        if not Path(value).exists():
            missing.append(f"{field}={value}")
    if missing:
        raise FileNotFoundError(
            "profile smoke manifest references missing artifact(s): "
            + ", ".join(missing)
        )
    return True


def manifest_int(row, key, default):
    value = row.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"profile smoke manifest {key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"profile smoke manifest {key} must be an integer") from exc


def manifest_promoted_ft_epochs(row):
    values = row.get("promoted_ft_epochs") or []
    if not isinstance(values, list):
        raise ValueError("profile smoke manifest promoted_ft_epochs must be a list")
    epochs = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("profile smoke manifest promoted_ft_epochs must be integers")
        epochs.append(int(value))
    return epochs


def inferred_continuation_ft_epochs_step(row):
    promoted_epochs = manifest_promoted_ft_epochs(row)
    if len(promoted_epochs) >= 2:
        return max(1, promoted_epochs[-1] - promoted_epochs[-2])
    if len(promoted_epochs) == 1:
        return max(1, promoted_epochs[0] - manifest_int(row, "ft_epochs", 1))
    return max(1, manifest_int(row, "promoted_ft_epochs_step", 1))


def continuation_ft_epochs(row, offset, *, first_epochs=None, step=None):
    step = step or inferred_continuation_ft_epochs_step(row)
    if first_epochs is not None:
        return first_epochs + (offset * step)
    promoted_epochs = manifest_promoted_ft_epochs(row)
    base = promoted_epochs[-1] if promoted_epochs else manifest_int(row, "ft_epochs", 1)
    return base + ((offset + 1) * step)


def manifest_path(row, field, context):
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"promoted rung manifest {context} missing {field}")
    return Path(value)


def promoted_rung_int(row, field, context):
    value = row.get(field)
    if isinstance(value, bool):
        raise ValueError(
            f"promoted rung manifest {context} {field} must be an integer"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"promoted rung manifest {context} {field} must be an integer"
        ) from exc


def validate_promoted_rung_manifest_consistency(manifest_row, rung_rows):
    if manifest_row.get("row_type") != "profile_smoke_manifest":
        raise ValueError(
            "promoted rung manifest requires a profile_smoke_manifest top-level row"
        )
    if not isinstance(rung_rows, list):
        raise ValueError("promoted rung manifest rows must be a list")

    expected_rungs = manifest_int(manifest_row, "promoted_rungs", 0)
    expected_epochs = manifest_promoted_ft_epochs(manifest_row)
    if len(expected_epochs) != expected_rungs:
        raise ValueError(
            "promoted rung manifest mismatch: "
            f"promoted_rungs={expected_rungs} but promoted_ft_epochs has "
            f"{len(expected_epochs)} entries"
        )
    if expected_rungs == 0:
        if rung_rows:
            raise ValueError(
                "promoted rung manifest mismatch: promoted_rungs=0 but "
                f"{len(rung_rows)} rung row(s) were provided"
            )
        return True
    if len(rung_rows) != expected_rungs:
        raise ValueError(
            "promoted rung manifest mismatch: "
            f"promoted_rungs={expected_rungs} but promoted-rungs JSONL has "
            f"{len(rung_rows)} row(s)"
        )

    previous_promotion = manifest_path(manifest_row, "promotion_jsonl", "top-level")
    missing = []
    final_summary = None
    final_promotion = None
    for index, row in enumerate(rung_rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"promoted rung manifest row {index} must be an object")
        if row.get("row_type") != "profile_smoke_promoted_rung":
            raise ValueError(
                "promoted rung manifest row "
                f"{index} has unsupported row_type: {row.get('row_type')!r}"
            )
        rung = promoted_rung_int(row, "rung", f"row {index}")
        if rung != index:
            raise ValueError(
                "promoted rung manifest mismatch: "
                f"row {index} declares rung={rung}, expected {index}"
            )
        ft_epochs = promoted_rung_int(row, "ft_epochs", f"rung {index}")
        if ft_epochs != expected_epochs[index - 1]:
            raise ValueError(
                "promoted rung manifest mismatch: "
                f"rung {index} ft_epochs={ft_epochs}, expected "
                f"{expected_epochs[index - 1]}"
            )

        input_promotion = manifest_path(row, "input_promotion_jsonl", f"rung {index}")
        if input_promotion != previous_promotion:
            raise ValueError(
                "promoted rung manifest chain mismatch: "
                f"rung {index} input_promotion_jsonl={input_promotion}, "
                f"expected {previous_promotion}"
            )
        for field in ["input_promotion_jsonl", *PROMOTED_RUNG_ARTIFACT_FIELDS]:
            path = manifest_path(row, field, f"rung {index}")
            if not path.exists():
                missing.append(f"rung {index} {field}={path}")
        final_summary = manifest_path(row, "run_summary_jsonl", f"rung {index}")
        final_promotion = manifest_path(row, "promotion_jsonl", f"rung {index}")
        previous_promotion = final_promotion

    if missing:
        raise FileNotFoundError(
            "promoted rung manifest references missing artifact(s): "
            + ", ".join(missing)
        )

    top_final_summary = manifest_path(
        manifest_row,
        "promoted_final_run_summary_jsonl",
        "top-level",
    )
    top_final_promotion = manifest_path(
        manifest_row,
        "promoted_final_promotion_jsonl",
        "top-level",
    )
    if final_summary != top_final_summary:
        raise ValueError(
            "promoted rung manifest final summary mismatch: "
            f"top-level={top_final_summary}, final rung={final_summary}"
        )
    if final_promotion != top_final_promotion:
        raise ValueError(
            "promoted rung manifest final promotion mismatch: "
            f"top-level={top_final_promotion}, final rung={final_promotion}"
        )
    return True


def promoted_rungs_jsonl_path_for_manifest(row):
    value = row.get("promoted_rungs_jsonl")
    if isinstance(value, str) and value:
        return Path(value)
    return manifest_path(row, "out_dir", "top-level") / "promoted-rungs.jsonl"


def load_profile_smoke_manifest_with_rungs(path):
    row = load_single_profile_smoke_manifest(path)
    promoted_rungs_jsonl = promoted_rungs_jsonl_path_for_manifest(row)
    rung_rows = load_jsonl(promoted_rungs_jsonl) if promoted_rungs_jsonl.exists() else []
    validate_promoted_rung_manifest_consistency(row, rung_rows)
    return row, promoted_rungs_jsonl, rung_rows


def transformers_trace_validation_fields(row):
    fields = {
        "transformers_trace": bool(row.get("transformers_trace", False)),
        "transformers_trace_jsonl": row.get("transformers_trace_jsonl"),
        "compare_transformers_trace_jsonl": row.get(
            "compare_transformers_trace_jsonl"
        ),
        "transformers_trace_compare_jsonl": row.get(
            "transformers_trace_compare_jsonl"
        ),
        "transformers_trace_manifest_available": False,
        "transformers_trace_manifest_error": None,
        "transformers_trace_spiraltorch_imported": None,
        "transformers_trace_spiraltorch_version": None,
        "transformers_trace_spiraltorch_module_name": None,
        "transformers_trace_transformers_imported": None,
        "transformers_trace_transformers_version": None,
        "transformers_trace_transformers_module_name": None,
        "transformers_trace_coimport_status": None,
        "transformers_trace_runtime_import_presets": None,
        "transformers_trace_runtime_imports_requested": None,
        "transformers_trace_runtime_import_probe_count": None,
        "transformers_trace_runtime_imports_imported": None,
        "transformers_trace_runtime_imports_failed": None,
        "transformers_trace_runtime_imports_all_ok": None,
        "transformers_trace_runtime_import_versions": None,
        "transformers_trace_runtime_import_module_names": None,
        "transformers_trace_runtime_imports_json": None,
        "transformers_trace_compare_summary_available": False,
        "transformers_trace_compare_passed": None,
        "transformers_trace_compare_failures": None,
        "transformers_trace_compared_prompt_rows": None,
        "transformers_trace_runtime_metadata_available": None,
        "transformers_trace_runtime_metadata_changed_count": None,
        "transformers_trace_runtime_metadata_changed_fields": None,
        "transformers_trace_runtime_metadata_failures": None,
        "transformers_trace_missing_prompt_rows": None,
        "transformers_trace_extra_prompt_rows": None,
        "transformers_trace_prompt_changed_rows": None,
        "transformers_trace_top_token_changed_rows": None,
        "transformers_trace_zspace_status_changed_rows": None,
        "transformers_trace_observed_max_top_logit_regression": None,
        "transformers_trace_observed_max_top_probability_regression": None,
        "transformers_trace_observed_max_logit_l2_change": None,
        "transformers_trace_observed_max_hidden_state_l2_change": None,
    }
    trace_jsonl = row.get("transformers_trace_jsonl")
    if trace_jsonl:
        try:
            manifest = load_single_jsonl_row_type(
                trace_jsonl,
                "transformers_trace_manifest",
            )
        except (OSError, ValueError) as exc:
            fields["transformers_trace_manifest_error"] = (
                f"{exc.__class__.__name__}: {exc}"
            )
        else:
            fields.update(
                {
                    "transformers_trace_manifest_available": True,
                    "transformers_trace_spiraltorch_imported": manifest.get(
                        "spiraltorch_imported"
                    ),
                    "transformers_trace_spiraltorch_version": manifest.get(
                        "spiraltorch_version"
                    ),
                    "transformers_trace_spiraltorch_module_name": manifest.get(
                        "spiraltorch_module_name"
                    ),
                    "transformers_trace_transformers_imported": manifest.get(
                        "transformers_imported"
                    ),
                    "transformers_trace_transformers_version": manifest.get(
                        "transformers_version"
                    ),
                    "transformers_trace_transformers_module_name": manifest.get(
                        "transformers_module_name"
                    ),
                    "transformers_trace_coimport_status": manifest.get(
                        "transformers_spiraltorch_coimport_status"
                    ),
                    "transformers_trace_runtime_import_presets": manifest.get(
                        "runtime_import_presets"
                    ),
                    "transformers_trace_runtime_imports_requested": manifest.get(
                        "runtime_imports_requested"
                    ),
                    "transformers_trace_runtime_import_probe_count": manifest.get(
                        "runtime_import_probe_count"
                    ),
                    "transformers_trace_runtime_imports_imported": manifest.get(
                        "runtime_imports_imported"
                    ),
                    "transformers_trace_runtime_imports_failed": manifest.get(
                        "runtime_imports_failed"
                    ),
                    "transformers_trace_runtime_imports_all_ok": manifest.get(
                        "runtime_imports_all_ok"
                    ),
                    "transformers_trace_runtime_import_versions": manifest.get(
                        "runtime_import_versions"
                    ),
                    "transformers_trace_runtime_import_module_names": manifest.get(
                        "runtime_import_module_names"
                    ),
                    "transformers_trace_runtime_imports_json": manifest.get(
                        "runtime_imports_json"
                    ),
                }
            )
    compare_jsonl = row.get("transformers_trace_compare_jsonl")
    if not compare_jsonl:
        return fields

    summary = load_single_jsonl_row_type(
        compare_jsonl,
        "transformers_trace_compare_summary",
    )
    fields.update(
        {
            "transformers_trace_compare_summary_available": True,
            "transformers_trace_compare_passed": summary.get("passed"),
            "transformers_trace_compare_failures": summary.get("failures"),
            "transformers_trace_compared_prompt_rows": summary.get(
                "compared_prompt_rows"
            ),
            "transformers_trace_runtime_metadata_available": summary.get(
                "runtime_metadata_available"
            ),
            "transformers_trace_runtime_metadata_changed_count": summary.get(
                "runtime_metadata_changed_count"
            ),
            "transformers_trace_runtime_metadata_changed_fields": summary.get(
                "runtime_metadata_changed_fields"
            ),
            "transformers_trace_runtime_metadata_failures": summary.get(
                "runtime_metadata_failures"
            ),
            "transformers_trace_missing_prompt_rows": summary.get(
                "missing_prompt_rows"
            ),
            "transformers_trace_extra_prompt_rows": summary.get("extra_prompt_rows"),
            "transformers_trace_prompt_changed_rows": summary.get(
                "prompt_changed_rows"
            ),
            "transformers_trace_top_token_changed_rows": summary.get(
                "top_token_changed_rows"
            ),
            "transformers_trace_zspace_status_changed_rows": summary.get(
                "zspace_status_changed_rows"
            ),
            "transformers_trace_observed_max_top_logit_regression": summary.get(
                "observed_max_top_logit_regression"
            ),
            "transformers_trace_observed_max_top_probability_regression": summary.get(
                "observed_max_top_probability_regression"
            ),
            "transformers_trace_observed_max_logit_l2_change": summary.get(
                "observed_max_logit_l2_change"
            ),
            "transformers_trace_observed_max_hidden_state_l2_change": summary.get(
                "observed_max_hidden_state_l2_change"
            ),
        }
    )
    return fields


def profile_smoke_manifest_validation_row(path, row, promoted_rungs_jsonl, rung_rows):
    promoted_rungs = manifest_int(row, "promoted_rungs", 0)
    promoted_epochs = manifest_promoted_ft_epochs(row)
    validation = {
        "row_type": "profile_smoke_manifest_validation",
        "valid": True,
        "manifest_jsonl": str(path),
        "out_dir": row["out_dir"],
        "checkpoint": row.get("checkpoint"),
        "checkpoint_source": row.get("checkpoint_source"),
        "source_label": row.get("source_label"),
        "cases": list(row.get("cases") or []),
        "configs": list(row.get("configs") or []),
        "profiles": list(row.get("profiles") or []),
        "promotion_metric": row.get("promotion_metric"),
        "promoted_rungs": promoted_rungs,
        "promoted_ft_epochs": promoted_epochs,
        "promoted_rung_rows": len(rung_rows),
        "promoted_rungs_jsonl": str(promoted_rungs_jsonl),
        "promoted_final_run_summary_jsonl": row.get("promoted_final_run_summary_jsonl"),
        "promoted_final_promotion_jsonl": row.get("promoted_final_promotion_jsonl"),
        "next_promoted_rung": promoted_rungs + 1,
        "next_ft_epochs": continuation_ft_epochs(row, 0),
        "required_manifest_artifacts_checked": len(required_manifest_artifact_fields(row)),
        "promoted_rung_artifacts_checked": len(rung_rows)
        * (1 + len(PROMOTED_RUNG_ARTIFACT_FIELDS)),
    }
    validation.update(transformers_trace_validation_fields(row))
    return validation


def manifest_validation_float(row, key):
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def manifest_validation_int(row, key):
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def manifest_validation_csv_set(row, key):
    value = row.get(key)
    if value is None or value == "none":
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item)}
    return {part for part in str(value).split(",") if part}


def manifest_validation_csv_label(values):
    items = [str(item) for item in values if str(item)]
    return ",".join(items) if items else "none"


def manifest_required_runtime_imports(args):
    if args is None:
        return []
    return list(
        dict.fromkeys(
            getattr(
                args,
                "require_manifest_transformers_trace_runtime_import",
                [],
            )
            or []
        )
    )


def manifest_required_runtime_import_presets(args):
    if args is None:
        return []
    return list(
        dict.fromkeys(
            getattr(
                args,
                "require_manifest_transformers_trace_runtime_import_preset",
                [],
            )
            or []
        )
    )


def runtime_import_gate_fields(validation_row, args):
    required = manifest_required_runtime_imports(args)
    required_presets = manifest_required_runtime_import_presets(args)
    imported = sorted(
        manifest_validation_csv_set(
            validation_row,
            "transformers_trace_runtime_imports_imported",
        )
    )
    observed_presets = sorted(
        manifest_validation_csv_set(
            validation_row,
            "transformers_trace_runtime_import_presets",
        )
    )
    missing = [module_name for module_name in required if module_name not in imported]
    missing_presets = [
        preset for preset in required_presets if preset not in observed_presets
    ]
    gate_requested = bool(required)
    preset_gate_requested = bool(required_presets)
    return {
        "transformers_trace_required_runtime_imports": manifest_validation_csv_label(
            required
        ),
        "transformers_trace_required_runtime_imports_imported": (
            manifest_validation_csv_label(imported) if gate_requested else "none"
        ),
        "transformers_trace_required_runtime_imports_missing": (
            manifest_validation_csv_label(missing) if gate_requested else "none"
        ),
        "transformers_trace_required_runtime_imports_passed": (
            None if not gate_requested else not missing
        ),
        "transformers_trace_required_runtime_import_presets": (
            manifest_validation_csv_label(required_presets)
        ),
        "transformers_trace_required_runtime_import_presets_observed": (
            manifest_validation_csv_label(observed_presets)
            if preset_gate_requested
            else "none"
        ),
        "transformers_trace_required_runtime_import_presets_missing": (
            manifest_validation_csv_label(missing_presets)
            if preset_gate_requested
            else "none"
        ),
        "transformers_trace_required_runtime_import_presets_passed": (
            None if not preset_gate_requested else not missing_presets
        ),
    }


def manifest_trace_validation_gate_failures(validation_row, args):
    failures = []
    if args is None:
        return failures
    if (
        args.require_manifest_transformers_trace
        and not validation_row["transformers_trace"]
    ):
        failures.append("transformers_trace_missing")
    if args.require_manifest_transformers_trace_compare_pass:
        if not validation_row["transformers_trace_compare_summary_available"]:
            failures.append("transformers_trace_compare_summary_missing")
        elif validation_row["transformers_trace_compare_passed"] is not True:
            failures.append("transformers_trace_compare_failed")
    if args.require_manifest_transformers_trace_coimport:
        if not validation_row["transformers_trace_manifest_available"]:
            failures.append("transformers_trace_manifest_missing")
        elif (
            validation_row["transformers_trace_spiraltorch_imported"] is not True
            or validation_row["transformers_trace_transformers_imported"] is not True
            or validation_row["transformers_trace_coimport_status"] != "ok"
        ):
            failures.append("transformers_trace_coimport_failed")
    if args.require_manifest_transformers_trace_runtime_imports:
        if not validation_row["transformers_trace_manifest_available"]:
            failures.append("transformers_trace_manifest_missing")
        else:
            probe_count = manifest_validation_int(
                validation_row,
                "transformers_trace_runtime_import_probe_count",
            )
            if probe_count is None or probe_count <= 0:
                failures.append("transformers_trace_runtime_imports_missing")
            elif (
                validation_row["transformers_trace_runtime_imports_all_ok"]
                is not True
            ):
                failures.append("transformers_trace_runtime_imports_failed")
    required_runtime_imports = manifest_required_runtime_imports(args)
    if required_runtime_imports:
        if not validation_row["transformers_trace_manifest_available"]:
            failures.append("transformers_trace_manifest_missing")
        else:
            imported = manifest_validation_csv_set(
                validation_row,
                "transformers_trace_runtime_imports_imported",
            )
            for module_name in required_runtime_imports:
                if module_name not in imported:
                    failures.append(
                        f"transformers_trace_runtime_import_missing:{module_name}"
                    )
    required_runtime_import_presets = manifest_required_runtime_import_presets(args)
    if required_runtime_import_presets:
        if not validation_row["transformers_trace_manifest_available"]:
            failures.append("transformers_trace_manifest_missing")
        else:
            observed_presets = manifest_validation_csv_set(
                validation_row,
                "transformers_trace_runtime_import_presets",
            )
            for preset in required_runtime_import_presets:
                if preset not in observed_presets:
                    failures.append(
                        "transformers_trace_runtime_import_preset_missing:"
                        f"{preset}"
                    )
    if args.require_manifest_transformers_trace_runtime_metadata_match:
        if not validation_row["transformers_trace_compare_summary_available"]:
            failures.append("transformers_trace_compare_summary_missing")
        elif (
            validation_row["transformers_trace_runtime_metadata_available"]
            is not True
        ):
            failures.append("transformers_trace_runtime_metadata_unavailable")
        else:
            changed_count = manifest_validation_int(
                validation_row,
                "transformers_trace_runtime_metadata_changed_count",
            )
            if changed_count is None:
                failures.append("transformers_trace_runtime_metadata_unavailable")
            elif changed_count != 0:
                failures.append("transformers_trace_runtime_metadata_changed")

    top_token_limit = args.max_manifest_transformers_trace_top_token_changed_rows
    if top_token_limit is not None:
        top_token_changed = manifest_validation_int(
            validation_row,
            "transformers_trace_top_token_changed_rows",
        )
        if top_token_changed is None:
            failures.append("transformers_trace_top_token_changed_rows_unavailable")
        elif top_token_changed > top_token_limit:
            failures.append("transformers_trace_top_token_changed_rows")

    probability_limit = (
        args.max_manifest_transformers_trace_top_probability_regression
    )
    if probability_limit is not None:
        probability_regression = manifest_validation_float(
            validation_row,
            "transformers_trace_observed_max_top_probability_regression",
        )
        if probability_regression is None:
            failures.append("transformers_trace_top_probability_regression_unavailable")
        elif probability_regression > probability_limit:
            failures.append("transformers_trace_top_probability_regression")
    return failures


def check_manifest_trace_validation_gates(validation_row, args):
    failures = manifest_trace_validation_gate_failures(validation_row, args)
    if args is None:
        return True
    gate_requested = any(
        [
            args.require_manifest_transformers_trace,
            args.require_manifest_transformers_trace_compare_pass,
            args.require_manifest_transformers_trace_runtime_metadata_match,
            args.require_manifest_transformers_trace_coimport,
            args.require_manifest_transformers_trace_runtime_imports,
            bool(
                getattr(
                    args,
                    "require_manifest_transformers_trace_runtime_import",
                    [],
                )
            ),
            bool(
                getattr(
                    args,
                    "require_manifest_transformers_trace_runtime_import_preset",
                    [],
                )
            ),
            args.max_manifest_transformers_trace_top_token_changed_rows
            is not None,
            args.max_manifest_transformers_trace_top_probability_regression
            is not None,
        ]
    )
    if not gate_requested:
        return True
    print(
        "profile_smoke_manifest_gate "
        f"gate=transformers_trace "
        f"failures={','.join(failures) if failures else 'none'} "
        f"passed={not failures}"
    )
    if failures:
        raise RuntimeError(
            "profile smoke manifest trace validation gate failed: "
            + ", ".join(failures)
        )
    return True


def validate_profile_smoke_manifest_file(path, validation_jsonl=None, args=None):
    row, promoted_rungs_jsonl, rung_rows = load_profile_smoke_manifest_with_rungs(path)
    validation_row = profile_smoke_manifest_validation_row(
        path,
        row,
        promoted_rungs_jsonl,
        rung_rows,
    )
    validation_row.update(runtime_import_gate_fields(validation_row, args))
    check_manifest_trace_validation_gates(validation_row, args)
    if validation_jsonl is not None:
        write_jsonl(validation_jsonl, [validation_row])
    output_parts = [
        "profile_smoke_manifest_validate",
        f"manifest={path}",
        f"out_dir={validation_row['out_dir']}",
        f"promoted_rungs={validation_row['promoted_rungs']}",
        "promoted_ft_epochs="
        f"{','.join(str(epoch) for epoch in validation_row['promoted_ft_epochs']) or 'none'}",
        f"promoted_rung_rows={validation_row['promoted_rung_rows']}",
        f"promoted_rungs_jsonl={validation_row['promoted_rungs_jsonl']}",
        f"next_promoted_rung={validation_row['next_promoted_rung']}",
        f"next_ft_epochs={validation_row['next_ft_epochs']}",
        "promoted_final_promotion_jsonl="
        f"{validation_row['promoted_final_promotion_jsonl']}",
    ]
    if validation_row["transformers_trace"]:
        output_parts.extend(
            [
                f"transformers_trace_jsonl={validation_row['transformers_trace_jsonl']}",
                "transformers_trace_coimport_status="
                f"{validation_row['transformers_trace_coimport_status']}",
                "transformers_trace_compare_passed="
                f"{validation_row['transformers_trace_compare_passed']}",
                "transformers_trace_runtime_metadata_changed_count="
                f"{validation_row['transformers_trace_runtime_metadata_changed_count']}",
                "transformers_trace_runtime_imports_all_ok="
                f"{validation_row['transformers_trace_runtime_imports_all_ok']}",
                "transformers_trace_runtime_import_presets="
                f"{validation_row['transformers_trace_runtime_import_presets']}",
                "transformers_trace_runtime_imports_failed="
                f"{validation_row['transformers_trace_runtime_imports_failed']}",
                "transformers_trace_runtime_imports_imported="
                f"{validation_row['transformers_trace_runtime_imports_imported']}",
                "transformers_trace_required_runtime_imports="
                f"{validation_row['transformers_trace_required_runtime_imports']}",
                "transformers_trace_required_runtime_imports_missing="
                f"{validation_row['transformers_trace_required_runtime_imports_missing']}",
                "transformers_trace_required_runtime_imports_passed="
                f"{validation_row['transformers_trace_required_runtime_imports_passed']}",
                "transformers_trace_required_runtime_import_presets="
                f"{validation_row['transformers_trace_required_runtime_import_presets']}",
                "transformers_trace_required_runtime_import_presets_missing="
                f"{validation_row['transformers_trace_required_runtime_import_presets_missing']}",
                "transformers_trace_required_runtime_import_presets_passed="
                f"{validation_row['transformers_trace_required_runtime_import_presets_passed']}",
                "transformers_trace_top_token_changed_rows="
                f"{validation_row['transformers_trace_top_token_changed_rows']}",
                "transformers_trace_observed_max_top_probability_regression="
                f"{validation_row['transformers_trace_observed_max_top_probability_regression']}",
            ]
        )
    print(" ".join(output_parts))
    return row, promoted_rungs_jsonl, rung_rows, validation_row


def extend_profile_filters(cmd, profiles):
    for profile in profiles:
        cmd.extend(["--profile", profile])


def checkpoint_shape_args():
    return [
        "--vocab",
        str(BYTE_LM_VOCAB),
        "--hidden",
        str(BYTE_LM_HIDDEN),
        "--target-classes",
        str(BYTE_LM_VOCAB),
    ]


def checkpoint_policy_args(args):
    flags = ["--key-preset", args.key_preset]
    for key in args.include_extra_keys:
        flags.extend(["--include-extra-key", key])
    if args.no_synthesize_missing_biases:
        flags.append("--no-synthesize-missing-biases")
    if args.allow_overlap_resize:
        flags.append("--allow-overlap-resize")
    if args.checkpoint_projection != "none":
        flags.extend(["--checkpoint-projection", args.checkpoint_projection])
    if args.checkpoint_projection_preset is not None:
        flags.extend(["--checkpoint-projection-preset", args.checkpoint_projection_preset])
    for name, flag in [
        ("checkpoint_projection_strength", "--checkpoint-projection-strength"),
        ("checkpoint_projection_curvature", "--checkpoint-projection-curvature"),
        ("checkpoint_projection_frequency", "--checkpoint-projection-frequency"),
    ]:
        value = getattr(args, name)
        if value is not None:
            flags.extend([flag, f"{float(value):g}"])
    if args.checkpoint_source_gain is not None:
        flags.extend(
            ["--checkpoint-source-gain", f"{float(args.checkpoint_source_gain):g}"]
        )
    return flags


def checkpoint_transformers_args(args):
    if not args.transformers_audit:
        return []
    flags = ["--transformers-audit"]
    if args.transformers_model_path is not None:
        flags.extend(["--transformers-model-path", args.transformers_model_path])
    if args.transformers_revision is not None:
        flags.extend(["--transformers-revision", args.transformers_revision])
    if args.allow_transformers_remote:
        flags.append("--allow-transformers-remote")
    if args.transformers_trust_remote_code:
        flags.append("--transformers-trust-remote-code")
    if args.skip_transformers_tokenizer:
        flags.append("--skip-transformers-tokenizer")
    if args.transformers_load_model:
        flags.append("--transformers-load-model")
    if args.require_transformers_audit:
        flags.append("--require-transformers-audit")
    return flags


def inferred_transformers_model_path(args, checkpoint_path):
    if args.transformers_model_path is not None:
        return args.transformers_model_path
    if args.hf_state_dict is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix:
        return checkpoint_path.parent
    return checkpoint_path


def transformers_trace_compare_requested(args):
    return (
        args.compare_transformers_trace_jsonl is not None
        or args.transformers_trace_require_zspace_status is not None
    )


def transformers_trace_args(args, model_path, trace_jsonl, trace_compare_jsonl):
    if not args.transformers_trace:
        return []
    flags = [
        "--model-path",
        model_path,
        "--jsonl",
        trace_jsonl,
        "--top-k",
        str(args.transformers_trace_top_k),
    ]
    for prompt in args.transformers_trace_prompts:
        flags.extend(["--prompt", prompt])
    if args.transformers_trace_prompt_file is not None:
        flags.extend(["--prompt-file", args.transformers_trace_prompt_file])
    if args.transformers_revision is not None:
        flags.extend(["--revision", args.transformers_revision])
    if args.allow_transformers_remote:
        flags.append("--allow-remote")
    if args.transformers_trust_remote_code:
        flags.append("--trust-remote-code")
    if args.transformers_trace_zspace_project:
        flags.extend(
            [
                "--zspace-project",
                "--zspace-source",
                args.transformers_trace_zspace_source,
            ]
        )
    for preset in getattr(args, "transformers_trace_runtime_import_presets", []) or []:
        flags.extend(["--runtime-import-preset", preset])
    for module_name in getattr(args, "transformers_trace_runtime_imports", []) or []:
        flags.extend(["--runtime-import", module_name])
    if getattr(args, "require_transformers_trace_runtime_imports", False):
        flags.append("--require-runtime-imports")
    if args.compare_transformers_trace_jsonl is not None:
        flags.extend(["--compare-jsonl", args.compare_transformers_trace_jsonl])
    if trace_compare_jsonl is not None:
        flags.extend(["--compare-output-jsonl", trace_compare_jsonl])
    if args.require_transformers_trace_match:
        flags.append("--require-trace-match")
    if args.require_transformers_trace_runtime_metadata_match:
        flags.append("--require-runtime-metadata-match")
    if args.require_transformers_trace_top_token_match:
        flags.append("--require-top-token-match")
    for name, flag in [
        ("transformers_trace_max_top_logit_regression", "--max-top-logit-regression"),
        (
            "transformers_trace_max_top_probability_regression",
            "--max-top-probability-regression",
        ),
        ("transformers_trace_max_logit_l2_change", "--max-logit-l2-change"),
        (
            "transformers_trace_max_hidden_state_l2_change",
            "--max-hidden-state-l2-change",
        ),
    ]:
        value = getattr(args, name)
        if value is not None:
            flags.extend([flag, f"{float(value):g}"])
    if args.transformers_trace_require_zspace_status is not None:
        flags.extend(
            [
                "--require-zspace-status",
                args.transformers_trace_require_zspace_status,
            ]
        )
    return flags


def run_guard_args(ft_epochs):
    return [
        "--require-run-guard-counts-available",
        "--min-run-guard-acceptance-rate-mean",
        "0.0",
        "--max-run-guard-retention-rejected-epochs-mean",
        "0.0",
        "--max-run-guard-retention-rejected-rate-mean",
        "0.0",
        "--max-run-guard-target-stale-epochs-mean",
        str(ft_epochs),
        "--max-run-guard-target-stale-rate-mean",
        "1.0",
    ]


def promotion_ready_args(ft_epochs, profiles, promotion_metric, promotion_jsonl):
    return [
        "--promotion-jsonl",
        str(promotion_jsonl),
        "--promotion-metric",
        promotion_metric,
        "--promotion-ready-top-k",
        str(len(profiles)),
        "--promotion-ready-require-guard-counts-available",
        "--promotion-ready-min-guard-acceptance-rate-mean",
        "0.0",
        "--promotion-ready-max-guard-retention-rejected-epochs-mean",
        "0.0",
        "--promotion-ready-max-guard-retention-rejected-rate-mean",
        "0.0",
        "--promotion-ready-max-guard-target-stale-epochs-mean",
        str(ft_epochs),
        "--promotion-ready-max-guard-target-stale-rate-mean",
        "1.0",
        "--min-promotion-ready-count",
        "1",
        "--min-promotion-ready-guard-policy-count",
        "1",
        "--require-promotion-ready-guard-policy",
    ]


def run_summary_compare_args():
    return [
        "--max-run-target-loss-regression",
        "0.0",
        "--max-run-retention-loss-regression",
        "0.0",
        "--max-run-target-retention-gap-regression",
        "0.0",
        "--max-run-accepted-rate-regression",
        "0.0",
        "--max-run-movement-ok-rate-regression",
        "0.0",
        "--max-run-guard-acceptance-rate-regression",
        "0.0",
        "--max-run-guard-retention-rejected-rate-regression",
        "0.0",
        "--max-run-guard-target-stale-rate-regression",
        "0.0",
    ]


def run_summary_identity_args():
    return [
        "--require-run-source-match",
        "--require-run-config-match",
        "--require-run-case-scope-match",
        "--require-run-training-policy-match",
        "--require-run-input-promotion-match",
    ]


def promoted_ft_epochs_for_rung(args, rung):
    first_epochs = (
        args.promoted_ft_epochs
        if args.promoted_ft_epochs is not None
        else args.ft_epochs + args.promoted_ft_epochs_step
    )
    return first_epochs + ((rung - 1) * args.promoted_ft_epochs_step)


def promoted_rung_artifacts(out_dir, output_prefix, rung):
    if rung <= 0:
        raise ValueError("rung must be positive")
    if rung == 1:
        return {
            "rung": rung,
            "run_dir": out_dir / "promoted-profile-runs",
            "output_prefix": output_prefix,
            "commands_jsonl": out_dir / "promoted-commands.jsonl",
            "selection_jsonl": out_dir / "promotion-selection.jsonl",
            "run_events_jsonl": out_dir / "promoted-profile-run-events.jsonl",
            "run_summary_jsonl": out_dir / "promoted-profile-run-summary.jsonl",
            "promotion_jsonl": out_dir / "promoted-promotion.jsonl",
        }
    label = f"promoted-rung{rung}"
    return {
        "rung": rung,
        "run_dir": out_dir / f"{label}-profile-runs",
        "output_prefix": f"{output_prefix}-rung{rung}",
        "commands_jsonl": out_dir / f"{label}-commands.jsonl",
        "selection_jsonl": out_dir / f"{label}-promotion-selection.jsonl",
        "run_events_jsonl": out_dir / f"{label}-profile-run-events.jsonl",
        "run_summary_jsonl": out_dir / f"{label}-profile-run-summary.jsonl",
        "promotion_jsonl": out_dir / f"{label}-promotion.jsonl",
    }


def promoted_rung_manifest_row(artifacts, *, ft_epochs, input_promotion_jsonl):
    return {
        "row_type": "profile_smoke_promoted_rung",
        "rung": artifacts["rung"],
        "ft_epochs": ft_epochs,
        "input_promotion_jsonl": str(input_promotion_jsonl),
        "output_dir": str(artifacts["run_dir"]),
        "output_prefix": artifacts["output_prefix"],
        "commands_jsonl": str(artifacts["commands_jsonl"]),
        "promotion_selection_jsonl": str(artifacts["selection_jsonl"]),
        "run_events_jsonl": str(artifacts["run_events_jsonl"]),
        "run_summary_jsonl": str(artifacts["run_summary_jsonl"]),
        "promotion_jsonl": str(artifacts["promotion_jsonl"]),
    }


def continuation_plan_row(
    *,
    manifest_path,
    output_manifest_jsonl,
    source_row,
    artifacts,
    ft_epochs,
    input_promotion_jsonl,
    profiles,
    promotion_metric,
    strict_aggregate_gates,
):
    row = promoted_rung_manifest_row(
        artifacts,
        ft_epochs=ft_epochs,
        input_promotion_jsonl=input_promotion_jsonl,
    )
    row.update(
        {
            "row_type": "profile_smoke_continue_plan",
            "manifest_jsonl": str(manifest_path),
            "output_manifest_jsonl": str(output_manifest_jsonl),
            "source_label": source_row.get("source_label"),
            "checkpoint": source_row.get("checkpoint"),
            "checkpoint_source": source_row.get("checkpoint_source"),
            "profile_jsonl": source_row.get("profile_jsonl"),
            "profiles": list(profiles),
            "promotion_metric": promotion_metric,
            "strict_aggregate_gates": strict_aggregate_gates,
        }
    )
    row.update(trace_policy_fields(source_row))
    return row


def trace_policy_fields(source):
    return {
        "transformers_trace": bool(source.get("transformers_trace", False)),
        "transformers_trace_jsonl": source.get("transformers_trace_jsonl"),
        "compare_transformers_trace_jsonl": source.get(
            "compare_transformers_trace_jsonl"
        ),
        "transformers_trace_compare_jsonl": source.get(
            "transformers_trace_compare_jsonl"
        ),
        "transformers_trace_prompts": list(
            source.get("transformers_trace_prompts") or []
        ),
        "transformers_trace_prompt_file": source.get(
            "transformers_trace_prompt_file"
        ),
        "transformers_trace_top_k": source.get("transformers_trace_top_k"),
        "transformers_trace_zspace_project": bool(
            source.get("transformers_trace_zspace_project", False)
        ),
        "transformers_trace_zspace_source": source.get(
            "transformers_trace_zspace_source"
        ),
        "transformers_trace_runtime_import_presets": list(
            source.get("transformers_trace_runtime_import_presets") or []
        ),
        "transformers_trace_runtime_imports": list(
            source.get("transformers_trace_runtime_imports") or []
        ),
        "require_transformers_trace_runtime_imports": bool(
            source.get("require_transformers_trace_runtime_imports", False)
        ),
        "require_transformers_trace_match": bool(
            source.get("require_transformers_trace_match", False)
        ),
        "require_transformers_trace_runtime_metadata_match": bool(
            source.get("require_transformers_trace_runtime_metadata_match", False)
        ),
        "require_transformers_trace_top_token_match": bool(
            source.get("require_transformers_trace_top_token_match", False)
        ),
        "transformers_trace_max_top_logit_regression": source.get(
            "transformers_trace_max_top_logit_regression"
        ),
        "transformers_trace_max_top_probability_regression": source.get(
            "transformers_trace_max_top_probability_regression"
        ),
        "transformers_trace_max_logit_l2_change": source.get(
            "transformers_trace_max_logit_l2_change"
        ),
        "transformers_trace_max_hidden_state_l2_change": source.get(
            "transformers_trace_max_hidden_state_l2_change"
        ),
        "transformers_trace_require_zspace_status": source.get(
            "transformers_trace_require_zspace_status"
        ),
    }


def profile_smoke_manifest_row(
    *,
    args,
    out_dir,
    checkpoint_path,
    checkpoint_source_kind,
    cases,
    configs,
    profiles,
    checkpoint_shape_audit_jsonl,
    checkpoint_preflight_jsonl,
    sweep_jsonl,
    sweep_aggregate_jsonl,
    source_compare_jsonl,
    profile_jsonl,
    run_dir,
    run_events_jsonl,
    run_summary_jsonl,
    promotion_jsonl,
    promotion_compare_jsonl,
    promoted_rungs,
    promoted_rungs_jsonl,
    promoted_artifacts,
    transformers_trace_jsonl=None,
    transformers_trace_compare_jsonl=None,
):
    promoted_epochs = [
        promoted_ft_epochs_for_rung(args, rung)
        for rung in range(1, promoted_rungs + 1)
    ]
    final_artifacts = promoted_artifacts[-1] if promoted_artifacts else None
    row = {
        "row_type": "profile_smoke_manifest",
        "out_dir": str(out_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_source": checkpoint_source_kind,
        "source_label": args.source_label,
        "key_preset": args.key_preset,
        "cases": list(cases),
        "configs": list(configs),
        "profiles": list(profiles),
        "ft_epochs": args.ft_epochs,
        "promotion_metric": args.promotion_metric,
        "promoted_output_prefix": args.promoted_output_prefix,
        "promoted_ft_epochs_step": args.promoted_ft_epochs_step,
        "strict_aggregate_gates": args.strict_aggregate_gates,
        "checkpoint_shape_audit_jsonl": optional_path(
            None if args.skip_checkpoint_shape_audit else checkpoint_shape_audit_jsonl
        ),
        "checkpoint_preflight_jsonl": optional_path(
            None if args.skip_checkpoint_preflight else checkpoint_preflight_jsonl
        ),
        "compare_checkpoint_preflight_jsonl": optional_path(
            args.compare_checkpoint_preflight_jsonl
        ),
        "require_checkpoint_preflight_match": args.require_checkpoint_preflight_match,
        "transformers_audit": bool(getattr(args, "transformers_audit", False)),
        "transformers_model_path": optional_path(
            getattr(args, "transformers_model_path", None)
        ),
        "transformers_revision": getattr(args, "transformers_revision", None),
        "allow_transformers_remote": bool(
            getattr(args, "allow_transformers_remote", False)
        ),
        "transformers_trust_remote_code": bool(
            getattr(args, "transformers_trust_remote_code", False)
        ),
        "skip_transformers_tokenizer": bool(
            getattr(args, "skip_transformers_tokenizer", False)
        ),
        "transformers_load_model": bool(
            getattr(args, "transformers_load_model", False)
        ),
        "require_transformers_audit": bool(
            getattr(args, "require_transformers_audit", False)
        ),
        "sweep_jsonl": str(sweep_jsonl),
        "sweep_aggregate_jsonl": str(sweep_aggregate_jsonl),
        "source_compare_jsonl": str(source_compare_jsonl),
        "profile_jsonl": str(profile_jsonl),
        "profile_run_dir": str(run_dir),
        "run_events_jsonl": str(run_events_jsonl),
        "run_summary_jsonl": str(run_summary_jsonl),
        "promotion_jsonl": str(promotion_jsonl),
        "promotion_compare_jsonl": str(promotion_compare_jsonl),
        "promoted_rungs": promoted_rungs,
        "promoted_ft_epochs": promoted_epochs,
        "promoted_rungs_jsonl": optional_path(
            promoted_rungs_jsonl if promoted_rungs > 0 else None
        ),
        "promoted_final_run_summary_jsonl": optional_path(
            final_artifacts["run_summary_jsonl"] if final_artifacts else None
        ),
        "promoted_final_promotion_jsonl": optional_path(
            final_artifacts["promotion_jsonl"] if final_artifacts else None
        ),
    }
    row.update(
        trace_policy_fields(
            {
                "transformers_trace": bool(
                    getattr(args, "transformers_trace", False)
                ),
                "transformers_trace_jsonl": optional_path(
                    transformers_trace_jsonl
                    if getattr(args, "transformers_trace", False)
                    else None
                ),
                "compare_transformers_trace_jsonl": optional_path(
                    getattr(args, "compare_transformers_trace_jsonl", None)
                ),
                "transformers_trace_compare_jsonl": optional_path(
                    transformers_trace_compare_jsonl
                ),
                "transformers_trace_prompts": list(
                    getattr(args, "transformers_trace_prompts", []) or []
                ),
                "transformers_trace_prompt_file": optional_path(
                    getattr(args, "transformers_trace_prompt_file", None)
                ),
                "transformers_trace_top_k": getattr(
                    args,
                    "transformers_trace_top_k",
                    None,
                ),
                "transformers_trace_zspace_project": bool(
                    getattr(args, "transformers_trace_zspace_project", False)
                ),
                "transformers_trace_zspace_source": getattr(
                    args,
                    "transformers_trace_zspace_source",
                    None,
                ),
                "transformers_trace_runtime_import_presets": list(
                    getattr(args, "transformers_trace_runtime_import_presets", [])
                    or []
                ),
                "transformers_trace_runtime_imports": list(
                    getattr(args, "transformers_trace_runtime_imports", []) or []
                ),
                "require_transformers_trace_runtime_imports": bool(
                    getattr(args, "require_transformers_trace_runtime_imports", False)
                ),
                "require_transformers_trace_match": bool(
                    getattr(args, "require_transformers_trace_match", False)
                ),
                "require_transformers_trace_runtime_metadata_match": bool(
                    getattr(
                        args,
                        "require_transformers_trace_runtime_metadata_match",
                        False,
                    )
                ),
                "require_transformers_trace_top_token_match": bool(
                    getattr(args, "require_transformers_trace_top_token_match", False)
                ),
                "transformers_trace_max_top_logit_regression": getattr(
                    args,
                    "transformers_trace_max_top_logit_regression",
                    None,
                ),
                "transformers_trace_max_top_probability_regression": getattr(
                    args,
                    "transformers_trace_max_top_probability_regression",
                    None,
                ),
                "transformers_trace_max_logit_l2_change": getattr(
                    args,
                    "transformers_trace_max_logit_l2_change",
                    None,
                ),
                "transformers_trace_max_hidden_state_l2_change": getattr(
                    args,
                    "transformers_trace_max_hidden_state_l2_change",
                    None,
                ),
                "transformers_trace_require_zspace_status": getattr(
                    args,
                    "transformers_trace_require_zspace_status",
                    None,
                ),
            }
        )
    )
    return row


def continue_profile_smoke_from_manifest(args):
    manifest_path = args.continue_manifest_jsonl
    row, promoted_rungs_jsonl, rung_manifest_rows = load_profile_smoke_manifest_with_rungs(
        manifest_path
    )
    out_dir = Path(row["out_dir"])
    output_manifest_jsonl = args.continue_manifest_output_jsonl or manifest_path
    output_prefix = (
        args.continue_promoted_output_prefix
        or row.get("promoted_output_prefix")
        or args.promoted_output_prefix
    )
    existing_rungs = manifest_int(row, "promoted_rungs", 0)
    existing_epochs = manifest_promoted_ft_epochs(row)
    step = args.continue_ft_epochs_step or inferred_continuation_ft_epochs_step(row)
    previous_promotion_jsonl = Path(
        row.get("promoted_final_promotion_jsonl") or row["promotion_jsonl"]
    )
    continued_epochs = []
    final_artifacts = None
    profiles = row.get("profiles") or []
    promotion_metric = row.get("promotion_metric") or args.promotion_metric
    strict_aggregate_gates = bool(row.get("strict_aggregate_gates", False))

    planned_rungs = []
    plan_previous_promotion_jsonl = previous_promotion_jsonl
    for offset in range(args.continue_rungs):
        rung = existing_rungs + offset + 1
        promoted_ft_epochs = continuation_ft_epochs(
            row,
            offset,
            first_epochs=args.continue_ft_epochs,
            step=step,
        )
        artifacts = promoted_rung_artifacts(out_dir, output_prefix, rung)
        input_promotion_jsonl = plan_previous_promotion_jsonl
        planned_rungs.append(
            {
                "artifacts": artifacts,
                "ft_epochs": promoted_ft_epochs,
                "input_promotion_jsonl": input_promotion_jsonl,
                "plan_row": continuation_plan_row(
                    manifest_path=manifest_path,
                    output_manifest_jsonl=output_manifest_jsonl,
                    source_row=row,
                    artifacts=artifacts,
                    ft_epochs=promoted_ft_epochs,
                    input_promotion_jsonl=input_promotion_jsonl,
                    profiles=profiles,
                    promotion_metric=promotion_metric,
                    strict_aggregate_gates=strict_aggregate_gates,
                ),
            }
        )
        plan_previous_promotion_jsonl = artifacts["promotion_jsonl"]

    if args.continue_plan_jsonl is not None:
        write_jsonl(
            args.continue_plan_jsonl,
            [planned["plan_row"] for planned in planned_rungs],
        )

    for planned in planned_rungs:
        artifacts = planned["artifacts"]
        promoted_ft_epochs = planned["ft_epochs"]
        input_promotion_jsonl = planned["input_promotion_jsonl"]
        promoted_runner_cmd = [
            args.python,
            SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
            "--profile-jsonl",
            row["profile_jsonl"],
            "--promotion-input-jsonl",
            input_promotion_jsonl,
            "--source-path",
            f"{row['source_label']}={row['checkpoint']}",
            "--output-dir",
            artifacts["run_dir"],
            "--output-prefix",
            artifacts["output_prefix"],
            "--commands-jsonl",
            artifacts["commands_jsonl"],
            "--promotion-selection-jsonl",
            artifacts["selection_jsonl"],
            "--run",
            "--run-events-jsonl",
            artifacts["run_events_jsonl"],
            "--run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--override-ft-epochs",
            str(promoted_ft_epochs),
            "--max-run-input-promotion-metric-regression",
            "0.0",
            "--min-promotion-ready-count",
            "1",
            "--min-promotion-ready-guard-policy-count",
            "1",
            "--require-promotion-ready-guard-policy",
        ]
        promoted_runner_cmd.extend(run_guard_args(promoted_ft_epochs))
        if not strict_aggregate_gates:
            promoted_runner_cmd.append("--no-aggregate-gates")
        extend_profile_filters(promoted_runner_cmd, profiles)
        run_command(promoted_runner_cmd, dry_run=args.dry_run)

        promoted_compare_cmd = [
            args.python,
            SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
            "--current-run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--compare-run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--max-run-input-promotion-metric-regression",
            "0.0",
        ]
        promoted_compare_cmd.extend(run_summary_compare_args())
        promoted_compare_cmd.extend(run_guard_args(promoted_ft_epochs))
        promoted_compare_cmd.extend(run_summary_identity_args())
        promoted_compare_cmd.extend(
            promotion_ready_args(
                promoted_ft_epochs,
                profiles,
                promotion_metric,
                artifacts["promotion_jsonl"],
            )
        )
        run_command(promoted_compare_cmd, dry_run=args.dry_run)

        rung_manifest_rows.append(
            promoted_rung_manifest_row(
                artifacts,
                ft_epochs=promoted_ft_epochs,
                input_promotion_jsonl=input_promotion_jsonl,
            )
        )
        if not args.dry_run:
            write_jsonl(promoted_rungs_jsonl, rung_manifest_rows)
        continued_epochs.append(promoted_ft_epochs)
        previous_promotion_jsonl = artifacts["promotion_jsonl"]
        final_artifacts = artifacts

    updated = dict(row)
    updated["promoted_rungs"] = existing_rungs + args.continue_rungs
    updated["promoted_ft_epochs"] = existing_epochs + continued_epochs
    updated["promoted_output_prefix"] = output_prefix
    updated["promoted_ft_epochs_step"] = step
    updated["promoted_rungs_jsonl"] = str(promoted_rungs_jsonl)
    if final_artifacts is not None:
        updated["promoted_final_run_summary_jsonl"] = str(
            final_artifacts["run_summary_jsonl"]
        )
        updated["promoted_final_promotion_jsonl"] = str(
            final_artifacts["promotion_jsonl"]
        )

    validation_jsonl = (
        produced_manifest_validation_jsonl(args, output_manifest_jsonl)
        if args.validate_produced_manifest
        else None
    )
    if not args.dry_run:
        write_jsonl(output_manifest_jsonl, [updated])
        validate_profile_smoke_manifest_artifacts(updated)
        validate_promoted_rung_manifest_consistency(updated, rung_manifest_rows)
        if args.validate_produced_manifest:
            validate_profile_smoke_manifest_file(
                output_manifest_jsonl,
                validation_jsonl=validation_jsonl,
                args=args,
            )

    output_parts = [
        "profile_smoke_manifest_continue",
        f"manifest={manifest_path}",
        f"output_manifest={output_manifest_jsonl}",
        f"continue_plan_jsonl={args.continue_plan_jsonl}",
        f"continued_rungs={args.continue_rungs}",
        f"promoted_rungs={updated['promoted_rungs']}",
        f"continued_ft_epochs={','.join(str(epoch) for epoch in continued_epochs)}",
        "promoted_final_promotion_jsonl="
        f"{updated.get('promoted_final_promotion_jsonl')}",
    ]
    if args.validate_produced_manifest:
        output_parts.append("validated_produced_manifest=True")
        output_parts.append(f"manifest_validation_jsonl={validation_jsonl}")
    print(" ".join(output_parts))


def main():
    args = parse_args()
    if args.validate_manifest_jsonl is not None:
        validate_profile_smoke_manifest_file(
            args.validate_manifest_jsonl,
            validation_jsonl=args.manifest_validation_jsonl,
            args=args,
        )
        return
    if args.continue_manifest_jsonl is not None:
        continue_profile_smoke_from_manifest(args)
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "profile-runs"
    cases = args.cases or ["adapter_ja"]
    configs = args.configs or ["r12_a64_lr4"]
    profiles = args.profiles or ["strong_effect"]
    promoted_rungs = 0 if args.skip_promoted_follow_up else args.promoted_rungs
    promoted_artifacts = [
        promoted_rung_artifacts(out_dir, args.promoted_output_prefix, rung)
        for rung in range(1, promoted_rungs + 1)
    ]

    generated_checkpoint_path = out_dir / "pytorch_model.bin"
    checkpoint_path = args.hf_state_dict or generated_checkpoint_path
    checkpoint_source_kind = "external" if args.hf_state_dict is not None else "generated"
    checkpoint_shape_audit_jsonl = args.checkpoint_shape_audit_jsonl or (
        out_dir / "checkpoint-shape-audit.jsonl"
    )
    checkpoint_preflight_jsonl = args.checkpoint_preflight_jsonl or (
        out_dir / "checkpoint-preflight.jsonl"
    )
    transformers_trace_jsonl = None
    transformers_trace_compare_jsonl = None
    if args.transformers_trace:
        transformers_trace_jsonl = args.transformers_trace_jsonl or (
            out_dir / "transformers-trace.jsonl"
        )
        if transformers_trace_compare_requested(args):
            transformers_trace_compare_jsonl = (
                args.transformers_trace_compare_jsonl
                or (out_dir / "transformers-trace-compare.jsonl")
            )
    sweep_jsonl = out_dir / "sweep.jsonl"
    sweep_aggregate_jsonl = out_dir / "sweep-aggregate.jsonl"
    source_compare_jsonl = out_dir / "source-compare.jsonl"
    profile_jsonl = out_dir / "profiles.jsonl"
    run_events_jsonl = out_dir / "profile-run-events.jsonl"
    run_summary_jsonl = out_dir / "profile-run-summary.jsonl"
    promotion_jsonl = out_dir / "promotion.jsonl"
    promotion_compare_jsonl = out_dir / "promotion-compare.jsonl"
    promoted_rungs_jsonl = args.promoted_rungs_jsonl or (out_dir / "promoted-rungs.jsonl")
    manifest_jsonl = args.manifest_jsonl or (out_dir / "profile-smoke-manifest.jsonl")

    if not args.keep_existing and not args.dry_run:
        promoted_output_paths = []
        for artifacts in promoted_artifacts:
            promoted_output_paths.extend(
                [
                    artifacts["commands_jsonl"],
                    artifacts["selection_jsonl"],
                    artifacts["run_events_jsonl"],
                    artifacts["run_summary_jsonl"],
                    artifacts["promotion_jsonl"],
                ]
            )
        output_paths = [
            sweep_jsonl,
            sweep_aggregate_jsonl,
            source_compare_jsonl,
            profile_jsonl,
            run_events_jsonl,
            run_summary_jsonl,
            promotion_jsonl,
            promotion_compare_jsonl,
            promoted_rungs_jsonl,
            manifest_jsonl,
            *promoted_output_paths,
        ]
        if args.hf_state_dict is None:
            output_paths.append(generated_checkpoint_path)
        if not args.skip_checkpoint_shape_audit:
            output_paths.append(checkpoint_shape_audit_jsonl)
        if not args.skip_checkpoint_preflight:
            output_paths.append(checkpoint_preflight_jsonl)
        if args.transformers_trace:
            output_paths.append(transformers_trace_jsonl)
            if transformers_trace_compare_jsonl is not None:
                output_paths.append(transformers_trace_compare_jsonl)
        clean_previous_outputs(
            output_paths,
            [run_dir, *[artifacts["run_dir"] for artifacts in promoted_artifacts]],
        )

    if args.hf_state_dict is not None and not args.dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(f"--hf-state-dict does not exist: {checkpoint_path}")

    if args.hf_state_dict is None:
        write_checkpoint_cmd = [
            args.python,
            SCRIPT_DIR / "write_byte_lm_hf_state_dict.py",
            "--out",
            checkpoint_path,
            "--key-preset",
            args.key_preset,
            "--include-biases",
        ]
        run_command(write_checkpoint_cmd, dry_run=args.dry_run)

    if args.transformers_trace:
        transformers_model_path = inferred_transformers_model_path(args, checkpoint_path)
        if transformers_model_path is None:
            raise ValueError("--transformers-trace requires a Transformers model path")
        transformers_trace_cmd = [
            args.python,
            SCRIPT_DIR / "byte_lm_transformers_trace.py",
        ]
        transformers_trace_cmd.extend(
            transformers_trace_args(
                args,
                transformers_model_path,
                transformers_trace_jsonl,
                transformers_trace_compare_jsonl,
            )
        )
        run_command(transformers_trace_cmd, dry_run=args.dry_run)

    if not args.skip_checkpoint_shape_audit:
        shape_audit_cmd = [
            args.python,
            SCRIPT_DIR / "checkpoint_preflight.py",
            "--hf-state-dict",
            checkpoint_path,
            "--shape-only",
            "--jsonl",
            checkpoint_shape_audit_jsonl,
            "--require-shape-materializable",
        ]
        shape_audit_cmd.extend(checkpoint_shape_args())
        shape_audit_cmd.extend(checkpoint_policy_args(args))
        shape_audit_cmd.extend(checkpoint_transformers_args(args))
        if args.shape_audit_require_exact_shape_match:
            shape_audit_cmd.append("--require-exact-shape-match")
        if args.shape_audit_require_detected_key_preset is not None:
            shape_audit_cmd.extend(
                [
                    "--require-detected-key-preset",
                    args.shape_audit_require_detected_key_preset,
                ]
            )
        run_command(shape_audit_cmd, dry_run=args.dry_run)

    if not args.skip_checkpoint_preflight:
        checkpoint_preflight_cmd = [
            args.python,
            SCRIPT_DIR / "checkpoint_preflight.py",
            "--hf-state-dict",
            checkpoint_path,
            "--jsonl",
            checkpoint_preflight_jsonl,
        ]
        checkpoint_preflight_cmd.extend(checkpoint_shape_args())
        checkpoint_preflight_cmd.extend(checkpoint_policy_args(args))
        checkpoint_preflight_cmd.extend(checkpoint_transformers_args(args))
        if args.compare_checkpoint_preflight_jsonl is not None:
            checkpoint_preflight_cmd.extend(
                ["--compare-jsonl", args.compare_checkpoint_preflight_jsonl]
            )
        if args.require_checkpoint_preflight_match:
            checkpoint_preflight_cmd.append("--require-preflight-match")
        run_command(checkpoint_preflight_cmd, dry_run=args.dry_run)

    sweep_cmd = [
        args.python,
        SCRIPT_DIR / "byte_lm_mlp_lora_sweep.py",
        "--hf-state-dict",
        checkpoint_path,
        "--checkpoint-source-label",
        args.source_label,
        "--ft-epochs-list",
        str(args.ft_epochs),
        "--jsonl",
        sweep_jsonl,
        "--aggregate-jsonl",
        sweep_aggregate_jsonl,
        "--min-aggregate-cases",
        str(len(cases)),
    ]
    sweep_cmd.extend(checkpoint_policy_args(args))
    for case in cases:
        sweep_cmd.extend(["--case", case, "--require-aggregate-case", case])
    for config in configs:
        sweep_cmd.extend(["--config", config])
    run_command(sweep_cmd, dry_run=args.dry_run)

    source_compare_cmd = [
        args.python,
        SCRIPT_DIR / "byte_lm_mlp_lora_source_compare.py",
        "--aggregate-jsonl",
        sweep_aggregate_jsonl,
        "--min-sources",
        "1",
        "--require-source",
        args.source_label,
        "--min-cases",
        str(len(cases)),
        "--min-movement-ok-rate",
        "1.0",
        "--jsonl",
        source_compare_jsonl,
        "--profile-jsonl",
        profile_jsonl,
    ]
    for case in cases:
        source_compare_cmd.extend(["--require-case", case])
    for profile in profiles:
        source_compare_cmd.extend(["--profile", profile])
    run_command(source_compare_cmd, dry_run=args.dry_run)

    profile_runner_cmd = [
        args.python,
        SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
        "--profile-jsonl",
        profile_jsonl,
        "--source-path",
        f"{args.source_label}={checkpoint_path}",
        "--output-dir",
        run_dir,
        "--output-prefix",
        "profile-smoke",
        "--run",
        "--run-events-jsonl",
        run_events_jsonl,
        "--run-summary-jsonl",
        run_summary_jsonl,
    ]
    profile_runner_cmd.extend(run_guard_args(args.ft_epochs))
    profile_runner_cmd.extend(
        promotion_ready_args(args.ft_epochs, profiles, args.promotion_metric, promotion_jsonl)
    )
    if not args.strict_aggregate_gates:
        profile_runner_cmd.append("--no-aggregate-gates")
    extend_profile_filters(profile_runner_cmd, profiles)
    run_command(profile_runner_cmd, dry_run=args.dry_run)

    compare_cmd = [
        args.python,
        SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
        "--current-run-summary-jsonl",
        run_summary_jsonl,
        "--compare-run-summary-jsonl",
        run_summary_jsonl,
    ]
    compare_cmd.extend(run_summary_compare_args())
    compare_cmd.extend(run_guard_args(args.ft_epochs))
    compare_cmd.extend(run_summary_identity_args())
    compare_cmd.extend(
        promotion_ready_args(
            args.ft_epochs,
            profiles,
            args.promotion_metric,
            promotion_compare_jsonl,
        )
    )
    run_command(compare_cmd, dry_run=args.dry_run)

    previous_promotion_jsonl = promotion_jsonl
    promoted_manifest_rows = []
    for artifacts in promoted_artifacts:
        rung = artifacts["rung"]
        promoted_ft_epochs = promoted_ft_epochs_for_rung(args, rung)
        input_promotion_jsonl = previous_promotion_jsonl
        promoted_runner_cmd = [
            args.python,
            SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
            "--profile-jsonl",
            profile_jsonl,
            "--promotion-input-jsonl",
            input_promotion_jsonl,
            "--source-path",
            f"{args.source_label}={checkpoint_path}",
            "--output-dir",
            artifacts["run_dir"],
            "--output-prefix",
            artifacts["output_prefix"],
            "--commands-jsonl",
            artifacts["commands_jsonl"],
            "--promotion-selection-jsonl",
            artifacts["selection_jsonl"],
            "--run",
            "--run-events-jsonl",
            artifacts["run_events_jsonl"],
            "--run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--override-ft-epochs",
            str(promoted_ft_epochs),
            "--max-run-input-promotion-metric-regression",
            "0.0",
            "--min-promotion-ready-count",
            "1",
            "--min-promotion-ready-guard-policy-count",
            "1",
            "--require-promotion-ready-guard-policy",
        ]
        promoted_runner_cmd.extend(run_guard_args(promoted_ft_epochs))
        if not args.strict_aggregate_gates:
            promoted_runner_cmd.append("--no-aggregate-gates")
        extend_profile_filters(promoted_runner_cmd, profiles)
        run_command(promoted_runner_cmd, dry_run=args.dry_run)

        promoted_compare_cmd = [
            args.python,
            SCRIPT_DIR / "byte_lm_mlp_lora_profile_runner.py",
            "--current-run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--compare-run-summary-jsonl",
            artifacts["run_summary_jsonl"],
            "--max-run-input-promotion-metric-regression",
            "0.0",
        ]
        promoted_compare_cmd.extend(run_summary_compare_args())
        promoted_compare_cmd.extend(run_guard_args(promoted_ft_epochs))
        promoted_compare_cmd.extend(run_summary_identity_args())
        promoted_compare_cmd.extend(
            promotion_ready_args(
                promoted_ft_epochs,
                profiles,
                args.promotion_metric,
                artifacts["promotion_jsonl"],
            )
        )
        run_command(promoted_compare_cmd, dry_run=args.dry_run)
        promoted_manifest_rows.append(
            promoted_rung_manifest_row(
                artifacts,
                ft_epochs=promoted_ft_epochs,
                input_promotion_jsonl=input_promotion_jsonl,
            )
        )
        if not args.dry_run:
            write_jsonl(promoted_rungs_jsonl, promoted_manifest_rows)
        previous_promotion_jsonl = artifacts["promotion_jsonl"]

    manifest_row = profile_smoke_manifest_row(
        args=args,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_source_kind=checkpoint_source_kind,
        cases=cases,
        configs=configs,
        profiles=profiles,
        checkpoint_shape_audit_jsonl=checkpoint_shape_audit_jsonl,
        checkpoint_preflight_jsonl=checkpoint_preflight_jsonl,
        transformers_trace_jsonl=transformers_trace_jsonl,
        transformers_trace_compare_jsonl=transformers_trace_compare_jsonl,
        sweep_jsonl=sweep_jsonl,
        sweep_aggregate_jsonl=sweep_aggregate_jsonl,
        source_compare_jsonl=source_compare_jsonl,
        profile_jsonl=profile_jsonl,
        run_dir=run_dir,
        run_events_jsonl=run_events_jsonl,
        run_summary_jsonl=run_summary_jsonl,
        promotion_jsonl=promotion_jsonl,
        promotion_compare_jsonl=promotion_compare_jsonl,
        promoted_rungs=promoted_rungs,
        promoted_rungs_jsonl=promoted_rungs_jsonl,
        promoted_artifacts=promoted_artifacts,
    )
    validation_jsonl = (
        produced_manifest_validation_jsonl(args, manifest_jsonl)
        if args.validate_produced_manifest
        else None
    )
    if not args.dry_run:
        write_jsonl(manifest_jsonl, [manifest_row])
        validate_profile_smoke_manifest_artifacts(manifest_row)
        validate_promoted_rung_manifest_consistency(
            manifest_row,
            promoted_manifest_rows,
        )
        if args.validate_produced_manifest:
            validate_profile_smoke_manifest_file(
                manifest_jsonl,
                validation_jsonl=validation_jsonl,
                args=args,
            )

    output_parts = [
        "profile_smoke_outputs",
        f"out_dir={out_dir}",
        f"checkpoint={checkpoint_path}",
        f"checkpoint_source={checkpoint_source_kind}",
        f"profile_smoke_manifest_jsonl={manifest_jsonl}",
        f"sweep_jsonl={sweep_jsonl}",
        f"sweep_aggregate_jsonl={sweep_aggregate_jsonl}",
        f"profile_jsonl={profile_jsonl}",
        f"run_summary_jsonl={run_summary_jsonl}",
        f"promotion_jsonl={promotion_jsonl}",
        f"promotion_compare_jsonl={promotion_compare_jsonl}",
    ]
    if not args.skip_checkpoint_shape_audit:
        output_parts.append(
            f"checkpoint_shape_audit_jsonl={checkpoint_shape_audit_jsonl}"
        )
    if not args.skip_checkpoint_preflight:
        output_parts.append(f"checkpoint_preflight_jsonl={checkpoint_preflight_jsonl}")
    if args.compare_checkpoint_preflight_jsonl is not None:
        output_parts.append(
            f"compare_checkpoint_preflight_jsonl={args.compare_checkpoint_preflight_jsonl}"
        )
    if args.validate_produced_manifest:
        output_parts.append("validated_produced_manifest=True")
        output_parts.append(f"manifest_validation_jsonl={validation_jsonl}")
    if args.transformers_trace:
        output_parts.append(f"transformers_trace_jsonl={transformers_trace_jsonl}")
        if args.compare_transformers_trace_jsonl is not None:
            output_parts.append(
                f"compare_transformers_trace_jsonl={args.compare_transformers_trace_jsonl}"
            )
        if transformers_trace_compare_jsonl is not None:
            output_parts.append(
                f"transformers_trace_compare_jsonl={transformers_trace_compare_jsonl}"
            )
    if promoted_rungs == 0:
        output_parts.append("promoted_follow_up=skipped")
    else:
        first_artifacts = promoted_artifacts[0]
        final_artifacts = promoted_artifacts[-1]
        promoted_epochs = [
            str(promoted_ft_epochs_for_rung(args, rung))
            for rung in range(1, promoted_rungs + 1)
        ]
        output_parts.extend(
            [
                f"promoted_rungs={promoted_rungs}",
                f"promoted_ft_epochs={','.join(promoted_epochs)}",
                f"promoted_rungs_jsonl={promoted_rungs_jsonl}",
                f"promotion_selection_jsonl={first_artifacts['selection_jsonl']}",
                f"promoted_commands_jsonl={first_artifacts['commands_jsonl']}",
                f"promoted_run_summary_jsonl={first_artifacts['run_summary_jsonl']}",
                f"promoted_promotion_jsonl={first_artifacts['promotion_jsonl']}",
                f"promoted_final_run_summary_jsonl={final_artifacts['run_summary_jsonl']}",
                f"promoted_final_promotion_jsonl={final_artifacts['promotion_jsonl']}",
            ]
        )
    print(" ".join(output_parts))


if __name__ == "__main__":
    main()
