"""Installed CLI entrypoints for Hugging Face/Z-Space workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

from .hf_ft import (
    HF_FINETUNE_DEFAULT_MODEL_PROFILE,
    hf_finetune_model_profile_catalog,
    hf_finetune_model_profile_catalog_lines,
    hf_finetune_model_profile_cli_args,
    hf_finetune_model_profile_launch_bundle_lines,
    hf_finetune_model_profile_launch_bundle_report,
    hf_finetune_model_profile_launch_bundle_report_lines,
    hf_finetune_model_profile_launch_plan,
    hf_finetune_model_profile_launch_plan_lines,
    hf_finetune_model_profile_lines,
    hf_finetune_model_profile_preflight_lines,
    hf_finetune_model_profile_preflight_report,
    hf_finetune_model_profile_runtime_contract,
    hf_finetune_model_profile_runtime_contract_from_artifact,
    hf_finetune_model_profile_runtime_contract_lines,
    resolve_hf_finetune_model_profile,
    write_hf_finetune_model_profile_launch_bundle,
    write_hf_finetune_model_profile_launch_plan,
    write_hf_finetune_model_profile_launch_script,
)
from .hf_generation import (
    compare_zspace_generation_control_sweeps,
    summarize_zspace_generation_control_sweep_comparison_lines,
    zspace_checkpoint_generation_control_report,
    zspace_generation_control_profile_config,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = _PACKAGE_ROOT / "examples"


def _example_path(name: str) -> Path:
    path = _EXAMPLES_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(
            "SpiralTorch HF CLI example payload is missing: "
            f"{path}. Reinstall the wheel or run the repository example directly."
        )
    return path


def _module_name(path: Path) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in path.stem)
    return f"_spiraltorch_installed_{token}"


def _load_example(name: str) -> ModuleType:
    path = _example_path(name)
    module_name = _module_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load SpiralTorch HF CLI example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_EXAMPLES_ROOT))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop(module_name, None)
    return module


def _run_example(name: str, argv: Sequence[str] | None = None) -> int:
    module = _load_example(name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError(f"SpiralTorch HF CLI example has no main(): {name}")
    return int(main(None if argv is None else list(argv)))


def _generation_control_profile_config_lines(report: dict) -> list[str]:
    profile = report.get("model_profile")
    profile_id = profile.get("profile_id") if isinstance(profile, dict) else None
    model_name = profile.get("model_name") if isinstance(profile, dict) else None
    fields = [
        f"status={report.get('status')}",
    ]
    if profile_id is not None:
        fields.append(f"profile={profile_id}")
    if model_name is not None:
        fields.append(f"model={model_name}")
    lines = ["zspace_generation_control_profile_config " + " ".join(fields)]
    contract_lines = report.get("model_profile_runtime_contract_lines")
    if isinstance(contract_lines, list):
        lines.extend(str(line) for line in contract_lines)
    bridge_cli_args = report.get("bridge_cli_args")
    if isinstance(bridge_cli_args, list) and bridge_cli_args:
        lines.append(
            "zspace_generation_control_bridge_cli_args "
            + " ".join(str(item) for item in bridge_cli_args)
        )
    sweep_cli_args = report.get("sweep_cli_args")
    if isinstance(sweep_cli_args, list) and sweep_cli_args:
        lines.append(
            "zspace_generation_control_sweep_cli_args "
            + " ".join(str(item) for item in sweep_cli_args)
        )
    return lines


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a SpiralTorch Hugging Face fine-tune model profile.",
    )
    parser.add_argument("--model-configs", type=Path, default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available profiles in the selected config.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve the profile and run import/device preflight for a mode.",
    )
    parser.add_argument(
        "--launch-plan",
        action="store_true",
        help="Build a launchable generic HF fine-tune command plan.",
    )
    parser.add_argument(
        "--generation-control-config",
        "--zspace-generation-control-config",
        action="store_true",
        help=(
            "Resolve the profile's generation section into reusable Z-Space "
            "generation-control kwargs and CLI args."
        ),
    )
    parser.add_argument(
        "--runtime-contract",
        action="store_true",
        help=(
            "Resolve the profile into a runtime contract for FT, local "
            "inference, and Z-Space generation consumers."
        ),
    )
    parser.add_argument(
        "--runtime-contract-artifact",
        "--runtime-contract-from-artifact",
        dest="runtime_contract_artifact",
        type=Path,
        default=None,
        help=(
            "Recover a runtime contract from a saved run card/report/contract "
            "JSON artifact."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(
            "auto",
            "runtime",
            "inference",
            "finetune",
            "full-finetune",
            "gpt2-ft",
            "peft",
            "trl-sft",
        ),
        default="auto",
        help=(
            "Runtime preset mode used with --preflight/--launch-plan. auto uses "
            "finetune for metadata/preflight and full-finetune for train plans."
        ),
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="With --preflight, require the selected mode's runtime import preset.",
    )
    parser.add_argument("--runtime-device-backend", action="append", default=[])
    parser.add_argument(
        "--require-runtime-device-backend",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--require-runtime-device-ready-backend",
        action="append",
        default=[],
    )
    parser.add_argument("--require-wgpu-ready", action="store_true")
    parser.add_argument(
        "--command",
        default="spiral-hf-finetune",
        help="Base command used with --launch-plan.",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--zspace-probe", action="store_true")
    parser.add_argument("--corpus-scan", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--script-out", type=Path, default=None)
    parser.add_argument("--script-cd", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--bundle-plan-filename", default="profile-launch-plan.json")
    parser.add_argument("--bundle-lines-filename", default="profile-launch-plan.lines")
    parser.add_argument("--bundle-script-filename", default="profile-launch-plan.sh")
    parser.add_argument("--inspect-bundle", type=Path, default=None)
    parser.add_argument("--refresh-preflight", action="store_true")
    parser.add_argument("--require-refresh-preflight", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--cli-args",
        action="store_true",
        help="Print bridge CLI args instead of compact profile lines.",
    )
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--no-training", action="store_true")
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-runtime", action="store_true")
    args = parser.parse_args(argv)
    exclusive_modes = [
        args.list,
        args.preflight,
        args.launch_plan,
        args.generation_control_config,
        args.runtime_contract,
        args.runtime_contract_artifact is not None,
        args.cli_args,
        args.inspect_bundle is not None,
    ]
    if sum(1 for enabled in exclusive_modes if enabled) > 1:
        parser.error(
            "--list, --preflight, --launch-plan, --generation-control-config, "
            "--runtime-contract, --runtime-contract-artifact, --cli-args, and "
            "--inspect-bundle are mutually exclusive"
        )
    if args.train and args.metadata_only:
        parser.error("--train and --metadata-only are mutually exclusive")
    if args.inspect_bundle is not None:
        report = hf_finetune_model_profile_launch_bundle_report(
            args.inspect_bundle,
            plan_filename=args.bundle_plan_filename,
            lines_filename=args.bundle_lines_filename,
            script_filename=args.bundle_script_filename,
            refresh_preflight=(
                args.refresh_preflight or args.require_refresh_preflight
            ),
            require_refreshed_preflight=args.require_refresh_preflight,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["status"] == "ready" else 1
        for line in hf_finetune_model_profile_launch_bundle_report_lines(report):
            print(line)
        return 0 if report["status"] == "ready" else 1
    if args.list:
        catalog = hf_finetune_model_profile_catalog(args.model_configs)
        if args.json:
            print(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        for line in hf_finetune_model_profile_catalog_lines(catalog):
            print(line)
        return 0
    if args.preflight:
        required_ready_backends = list(args.require_runtime_device_ready_backend)
        if args.require_wgpu_ready and "wgpu" not in required_ready_backends:
            required_ready_backends.append("wgpu")
        report = hf_finetune_model_profile_preflight_report(
            args.model_configs,
            profile=args.model_profile,
            mode=args.mode,
            require=args.require,
            runtime_device_backends=args.runtime_device_backend,
            required_runtime_device_backends=args.require_runtime_device_backend,
            required_runtime_device_ready_backends=required_ready_backends,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["runtime_import_preflight_passed"] else 1
        for line in hf_finetune_model_profile_preflight_lines(report):
            print(line)
        return 0 if report["runtime_import_preflight_passed"] else 1
    if args.launch_plan:
        required_ready_backends = list(args.require_runtime_device_ready_backend)
        if args.require_wgpu_ready and "wgpu" not in required_ready_backends:
            required_ready_backends.append("wgpu")
        plan = hf_finetune_model_profile_launch_plan(
            args.model_configs,
            profile=args.model_profile,
            mode=args.mode,
            require=args.require,
            command=args.command,
            train=args.train,
            metadata_only=True if args.metadata_only else None,
            output_dir=args.output_dir,
            run_card=args.run_card,
            trainer_trace_jsonl=args.trainer_trace_jsonl,
            zspace_probe=args.zspace_probe,
            corpus_scan=args.corpus_scan,
            extra_args=args.extra_arg,
            runtime_device_backends=args.runtime_device_backend,
            required_runtime_device_backends=args.require_runtime_device_backend,
            required_runtime_device_ready_backends=required_ready_backends,
        )
        lines = hf_finetune_model_profile_launch_plan_lines(plan)
        bundle = None
        if args.bundle_dir is not None:
            bundle = write_hf_finetune_model_profile_launch_bundle(
                plan,
                args.bundle_dir,
                plan_filename=args.bundle_plan_filename,
                lines_filename=args.bundle_lines_filename,
                script_filename=args.bundle_script_filename,
                script_cd=args.script_cd,
            )
        if args.json:
            payload = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
            print(payload)
            if args.out is not None and bundle is None:
                write_hf_finetune_model_profile_launch_plan(
                    plan,
                    args.out,
                    lines_path=args.lines_out,
                )
            elif args.lines_out is not None and bundle is None:
                args.lines_out.parent.mkdir(parents=True, exist_ok=True)
                args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if args.script_out is not None and bundle is None:
                write_hf_finetune_model_profile_launch_script(
                    plan,
                    args.script_out,
                    cd=args.script_cd,
                )
            return 0 if plan["runtime_import_preflight_passed"] else 1
        if bundle is not None:
            for line in hf_finetune_model_profile_launch_bundle_lines(bundle):
                print(line)
            return 0 if plan["runtime_import_preflight_passed"] else 1
        if args.out is not None:
            written = write_hf_finetune_model_profile_launch_plan(
                plan,
                args.out,
                lines_path=args.lines_out,
            )
            print(f"hf_ft_model_profile_launch_plan_out {written['path']}")
            if written.get("lines_path"):
                print(
                    "hf_ft_model_profile_launch_plan_lines_out "
                    f"{written['lines_path']}"
                )
            if args.script_out is not None:
                script_written = write_hf_finetune_model_profile_launch_script(
                    plan,
                    args.script_out,
                    cd=args.script_cd,
                )
                print(
                    "hf_ft_model_profile_launch_script_out "
                    f"{script_written['path']}"
                )
            return 0 if plan["runtime_import_preflight_passed"] else 1
        if args.lines_out is not None:
            args.lines_out.parent.mkdir(parents=True, exist_ok=True)
            args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"hf_ft_model_profile_launch_plan_lines_out {args.lines_out}")
            if args.script_out is not None:
                script_written = write_hf_finetune_model_profile_launch_script(
                    plan,
                    args.script_out,
                    cd=args.script_cd,
                )
                print(
                    "hf_ft_model_profile_launch_script_out "
                    f"{script_written['path']}"
                )
            return 0 if plan["runtime_import_preflight_passed"] else 1
        if args.script_out is not None:
            script_written = write_hf_finetune_model_profile_launch_script(
                plan,
                args.script_out,
                cd=args.script_cd,
            )
            print(f"hf_ft_model_profile_launch_script_out {script_written['path']}")
            return 0 if plan["runtime_import_preflight_passed"] else 1
        for line in lines:
            print(line)
        return 0 if plan["runtime_import_preflight_passed"] else 1
    if args.generation_control_config:
        report = zspace_generation_control_profile_config(
            args.model_configs,
            model_profile=args.model_profile,
        )
        lines = _generation_control_profile_config_lines(report)
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.json:
            print(payload, end="")
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(payload, encoding="utf-8")
            if args.lines_out is not None:
                args.lines_out.parent.mkdir(parents=True, exist_ok=True)
                args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 0
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"zspace_generation_control_profile_config_out {args.out}")
        else:
            for line in lines:
                print(line)
        if args.lines_out is not None:
            args.lines_out.parent.mkdir(parents=True, exist_ok=True)
            args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(
                "zspace_generation_control_profile_config_lines_out "
                f"{args.lines_out}"
            )
        return 0
    if args.runtime_contract:
        report = hf_finetune_model_profile_runtime_contract(
            args.model_configs,
            profile=args.model_profile,
            mode=args.mode,
        )
        lines = hf_finetune_model_profile_runtime_contract_lines(report)
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.json:
            print(payload, end="")
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(payload, encoding="utf-8")
            if args.lines_out is not None:
                args.lines_out.parent.mkdir(parents=True, exist_ok=True)
                args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 0
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"hf_ft_model_profile_runtime_contract_out {args.out}")
        else:
            for line in lines:
                print(line)
        if args.lines_out is not None:
            args.lines_out.parent.mkdir(parents=True, exist_ok=True)
            args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(
                "hf_ft_model_profile_runtime_contract_lines_out "
                f"{args.lines_out}"
            )
        return 0
    if args.runtime_contract_artifact is not None:
        report = hf_finetune_model_profile_runtime_contract_from_artifact(
            args.runtime_contract_artifact,
            mode=args.mode,
        )
        lines = hf_finetune_model_profile_runtime_contract_lines(report)
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.json:
            print(payload, end="")
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(payload, encoding="utf-8")
            if args.lines_out is not None:
                args.lines_out.parent.mkdir(parents=True, exist_ok=True)
                args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 0
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"hf_ft_model_profile_runtime_contract_out {args.out}")
        else:
            for line in lines:
                print(line)
        if args.lines_out is not None:
            args.lines_out.parent.mkdir(parents=True, exist_ok=True)
            args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(
                "hf_ft_model_profile_runtime_contract_lines_out "
                f"{args.lines_out}"
            )
        return 0
    profile = resolve_hf_finetune_model_profile(
        args.model_configs,
        profile=args.model_profile,
    )
    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cli_args:
        print(
            " ".join(
                hf_finetune_model_profile_cli_args(
                    profile,
                    include_model=not args.no_model,
                    include_training=not args.no_training,
                    include_dataset=not args.no_dataset,
                    include_generation=not args.no_generation,
                    include_runtime=not args.no_runtime,
                )
            )
        )
        return 0
    for line in hf_finetune_model_profile_lines(profile):
        print(line)
    return 0


def finetune_bridge_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_bridge.py", argv)


def finetune_sweep_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_sweep.py", argv)


def finetune_scale_up_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_scale_up.py", argv)


def finetune_trace_summary_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_trace_summary.py", argv)


def finetune_run_status_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_run_status.py", argv)


def finetune_monitor_snapshot_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_monitor_snapshot.py", argv)


def finetune_status_history_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_status_history_summary.py", argv)


def finetune_wait_launch_summary_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_wait_launch_summary.py", argv)


def finetune_wait_launch_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_wait_launch.py", argv)


def finetune_milestone_capture_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_milestone_capture.py", argv)


def finetune_milestone_runtime_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_milestone_runtime.py", argv)


def _path_values(values: Sequence[Path] | None) -> list[Path]:
    return [] if values is None else list(values)


def _generic_checkpoint_generation_control_report(report: dict) -> dict:
    payload = dict(report)
    if payload.get("row_type") == "hf_gpt2_ft_checkpoint_generation_control":
        payload["row_type"] = "hf_checkpoint_generation_control"
    return payload


def checkpoint_generation_control_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or plan Z-Space generation-control sweeps for Hugging Face "
            "fine-tune checkpoints."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help=(
            "Prompt spec as LABEL::TEXT. Defaults to SpiralTorch's checkpoint "
            "generation prompt set."
        ),
    )
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sweep-script", type=Path, default=None)
    parser.add_argument("--compare-script", type=Path, default=None)
    parser.add_argument("--curve-script", type=Path, default=None)
    parser.add_argument("--model-configs", type=Path, default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sample-temperature", type=float, default=None)
    parser.add_argument("--sample-top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--compare-out", type=Path, default=None)
    parser.add_argument("--compare-lines-out", type=Path, default=None)
    parser.add_argument("--curve-out", type=Path, default=None)
    parser.add_argument("--curve-lines-out", type=Path, default=None)
    parser.add_argument("--curve-run-card", type=Path, default=None)
    parser.add_argument("--curve-trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--curve-model-name", default=None)
    parser.add_argument("--curve-dataset-name", default=None)
    parser.add_argument("--curve-dataset-config", default=None)
    parser.add_argument("--compare-with-sweep", action="append", type=Path, default=None)
    parser.add_argument("--compare-with-label", action="append", default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-process-pid-file", type=Path, default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--ready-file", action="append", default=None)
    parser.add_argument("--no-ready-file-check", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--process-poll-seconds", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.sample_temperature is not None and args.sample_temperature <= 0.0:
        parser.error("--sample-temperature must be positive")
    if args.sample_top_k is not None and args.sample_top_k < 0:
        parser.error("--sample-top-k must be non-negative")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.process_poll_seconds is not None and args.process_poll_seconds <= 0.0:
        parser.error("--process-poll-seconds must be positive")
    if args.timeout_seconds < 0.0:
        parser.error("--timeout-seconds must be non-negative")
    if args.process_timeout_seconds < 0.0:
        parser.error("--process-timeout-seconds must be non-negative")
    compare_with_sweep = _path_values(args.compare_with_sweep)
    compare_with_label = list(args.compare_with_label or [])
    if compare_with_label and len(compare_with_label) != len(compare_with_sweep):
        parser.error("--compare-with-label must match --compare-with-sweep count")
    if (
        (args.curve_run_card is not None or args.curve_trainer_trace_jsonl is not None)
        and args.curve_out is None
        and args.curve_lines_out is None
    ):
        parser.error("curve source options require --curve-out or --curve-lines-out")
    model_profile = args.model_profile
    if (
        args.model_configs is None
        and model_profile is None
        and args.tokenizer_name is None
    ):
        model_profile = HF_FINETUNE_DEFAULT_MODEL_PROFILE
    report = zspace_checkpoint_generation_control_report(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        label_prefix=args.label_prefix,
        python=args.python,
        sweep_script=args.sweep_script,
        compare_script=args.compare_script,
        curve_script=args.curve_script,
        tokenizer_name=args.tokenizer_name,
        model_configs=args.model_configs,
        model_profile=model_profile,
        allow_remote=args.allow_remote,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        sample_temperature=args.sample_temperature,
        sample_top_k=args.sample_top_k,
        overwrite=args.overwrite,
        no_compare=args.no_compare,
        top_n=args.top_n,
        compare_out=args.compare_out,
        compare_lines_out=args.compare_lines_out,
        curve_out=args.curve_out,
        curve_lines_out=args.curve_lines_out,
        curve_run_card=args.curve_run_card,
        curve_trainer_trace_jsonl=args.curve_trainer_trace_jsonl,
        curve_model_name=args.curve_model_name,
        curve_dataset_name=args.curve_dataset_name,
        curve_dataset_config=args.curve_dataset_config,
        compare_with_sweep=compare_with_sweep,
        compare_with_label=compare_with_label,
        run_card=args.run_card,
        dry_run=args.dry_run,
        wait_for_process_pid_file=args.wait_for_process_pid_file,
        wait=args.wait,
        ready_file=args.ready_file,
        no_ready_file_check=args.no_ready_file_check,
        poll_seconds=args.poll_seconds,
        process_poll_seconds=args.process_poll_seconds,
        timeout_seconds=args.timeout_seconds,
        process_timeout_seconds=args.process_timeout_seconds,
    )
    report = _generic_checkpoint_generation_control_report(report)
    if args.run_card is not None:
        args.run_card.parent.mkdir(parents=True, exist_ok=True)
        args.run_card.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.dry_run and args.run_card is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def zspace_generation_control_sweep_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_zspace_generation_control_sweep.py", argv)


def zspace_generation_control_compare_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face Z-Space generation-control sweep artifacts.",
    )
    parser.add_argument("sweeps", nargs="+", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.label and len(args.label) != len(args.sweeps):
        parser.error("--label must be repeated exactly once per sweep path")
    missing = [path for path in args.sweeps if not path.is_file()]
    if missing:
        parser.error("sweep artifact does not exist: " + ", ".join(map(str, missing)))
    sources: dict[str, Path] | list[Path]
    if args.label:
        sources = {
            str(label): path
            for label, path in zip(args.label, args.sweeps)
        }
    else:
        sources = list(args.sweeps)
    comparison = compare_zspace_generation_control_sweeps(
        sources,
        top_n=int(args.top_n),
    )
    lines = summarize_zspace_generation_control_sweep_comparison_lines(
        comparison,
        top_n=int(args.top_n),
    )
    payload = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"generation_control_compare {args.out}")
    else:
        print(payload, end="")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"generation_control_compare_lines {args.lines_out}")
    else:
        for line in lines:
            print(line, file=sys.stderr)
    return 0


def zspace_inference_distortion_probe_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("zspace_inference_distortion_probe.py", argv)


def zspace_inference_distortion_sweep_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("zspace_inference_distortion_sweep.py", argv)


def finetune_generation_curve_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_generation_curve.py", argv)


def finetune_run_artifacts_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_run_artifacts.py", argv)


def finetune_run_ops_main(argv: Sequence[str] | None = None) -> int:
    return _run_example("hf_finetune_run_ops.py", argv)
