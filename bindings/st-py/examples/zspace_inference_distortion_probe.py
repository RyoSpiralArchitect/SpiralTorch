from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    provided_flags = _provided_flags(raw_argv)
    parser = argparse.ArgumentParser(
        description=(
            "Probe one shared Z-space inference distortion against a local HF "
            "model and an API-model-shaped callable."
        )
    )
    parser.add_argument("--prompt", default="Describe SpiralTorch as a Z-space runtime.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--from-sweep-report",
        type=Path,
        default=None,
        help="Load the recommended prompt/runtime/distortion config from a sweep-report.json.",
    )
    parser.add_argument("--local-model", type=Path, default=None)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--activation-module-name", action="append", default=[])
    parser.add_argument("--activation-name-contains", action="append", default=[])
    parser.add_argument(
        "--api-provider",
        choices=["fake", "openai-responses", "openai-chat", "anthropic"],
        default="fake",
    )
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-max-tokens", type=int, default=160)
    parser.add_argument("--desire-pressure", type=float, default=0.8)
    parser.add_argument("--desire-stability", type=float, default=0.45)
    parser.add_argument("--psi-total", type=float, default=0.7)
    parser.add_argument("--coherence", type=float, default=0.45)
    parser.add_argument("--distortion-strength", type=float, default=1.0)
    parser.add_argument("--base-temperature", type=float, default=0.7)
    parser.add_argument("--base-top-p", type=float, default=0.95)
    parser.add_argument("--include-penalties", action="store_true")
    args = parser.parse_args(argv)
    if args.from_sweep_report is not None:
        try:
            _apply_sweep_handoff(args, provided_flags)
        except Exception as exc:
            parser.error(str(exc))
    return args


def _provided_flags(argv: list[str]) -> set[str]:
    flags = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        flags.add(item.split("=", 1)[0])
    return flags


def _flag_was_provided(flags: set[str], flag: str) -> bool:
    return flag in flags


def _set_if_not_provided(
    args: argparse.Namespace,
    flags: set[str],
    flag: str,
    attr: str,
    value: Any,
) -> None:
    if value is None or _flag_was_provided(flags, flag):
        return
    setattr(args, attr, value)


def _extend_if_not_provided(
    args: argparse.Namespace,
    flags: set[str],
    flag: str,
    attr: str,
    values: Any,
) -> None:
    if values is None or _flag_was_provided(flags, flag):
        return
    if isinstance(values, list):
        setattr(args, attr, list(values))


def _apply_sweep_handoff(args: argparse.Namespace, flags: set[str]) -> None:
    summary = st.summarize_zspace_inference_distortion_sweep(args.from_sweep_report)
    config = summary.get("recommended_config")
    runtime = st.load_zspace_inference_distortion_sweep(args.from_sweep_report).get(
        "runtime"
    )
    if not isinstance(config, dict):
        raise ValueError(f"{args.from_sweep_report} did not contain a recommended config")
    runtime = runtime if isinstance(runtime, dict) else {}
    args.sweep_handoff = {
        "source": str(args.from_sweep_report),
        "recommended_probe": summary.get("recommended_probe"),
        "recommendation_reason": summary.get("recommendation_reason"),
        "recommended_probe_path": summary.get("recommended_probe_path"),
        "applied_config": dict(config),
    }
    _set_if_not_provided(args, flags, "--prompt", "prompt", summary.get("prompt"))
    _set_if_not_provided(
        args,
        flags,
        "--local-model",
        "local_model",
        Path(str(runtime["local_model"])) if runtime.get("local_model") else None,
    )
    if runtime.get("allow_remote") and not _flag_was_provided(flags, "--allow-remote"):
        args.allow_remote = True
    if runtime.get("trust_remote_code") and not _flag_was_provided(
        flags,
        "--trust-remote-code",
    ):
        args.trust_remote_code = True
    _set_if_not_provided(
        args,
        flags,
        "--max-new-tokens",
        "max_new_tokens",
        runtime.get("max_new_tokens"),
    )
    _extend_if_not_provided(
        args,
        flags,
        "--activation-module-name",
        "activation_module_name",
        runtime.get("activation_module_name"),
    )
    _extend_if_not_provided(
        args,
        flags,
        "--activation-name-contains",
        "activation_name_contains",
        runtime.get("activation_name_contains"),
    )
    _set_if_not_provided(
        args,
        flags,
        "--api-provider",
        "api_provider",
        runtime.get("api_provider"),
    )
    _set_if_not_provided(args, flags, "--api-model", "api_model", runtime.get("api_model"))
    _set_if_not_provided(
        args,
        flags,
        "--api-max-tokens",
        "api_max_tokens",
        runtime.get("api_max_tokens"),
    )
    _set_if_not_provided(
        args,
        flags,
        "--desire-pressure",
        "desire_pressure",
        config.get("desire_pressure"),
    )
    _set_if_not_provided(
        args,
        flags,
        "--desire-stability",
        "desire_stability",
        config.get("desire_stability"),
    )
    _set_if_not_provided(args, flags, "--psi-total", "psi_total", config.get("psi_total"))
    _set_if_not_provided(args, flags, "--coherence", "coherence", config.get("coherence"))
    _set_if_not_provided(
        args,
        flags,
        "--distortion-strength",
        "distortion_strength",
        config.get("distortion_strength"),
    )
    _set_if_not_provided(
        args,
        flags,
        "--base-temperature",
        "base_temperature",
        config.get("base_temperature"),
    )
    _set_if_not_provided(args, flags, "--base-top-p", "base_top_p", config.get("base_top_p"))
    if config.get("include_penalties") and not _flag_was_provided(
        flags,
        "--include-penalties",
    ):
        args.include_penalties = True


def _text_from_tokens(tokenizer: Any, token_ids: Any) -> str:
    decode = getattr(tokenizer, "decode")
    return str(decode(token_ids, skip_special_tokens=True))


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if device is None or not isinstance(batch, dict):
        return batch
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        to_device = getattr(value, "to", None)
        moved[key] = to_device(device) if callable(to_device) else value
    return moved


def _next_token_from_logits(
    torch: Any,
    logits: Any,
    *,
    input_ids: Any,
    logits_processor: Any | None,
) -> Any:
    last_logits = logits[:, -1, :]
    if logits_processor is not None:
        last_logits = logits_processor(input_ids, last_logits)
    return torch.argmax(last_logits, dim=-1, keepdim=True)


def _manual_forward_generate(
    torch: Any,
    tokenizer: Any,
    model: Any,
    batch: dict[str, Any],
    *,
    max_new_tokens: int,
    logits_processor: Any | None = None,
) -> Any:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("tokenizer output did not include input_ids")
    generated = input_ids
    attention_mask = batch.get("attention_mask")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(max(0, int(max_new_tokens))):
        call_kwargs = {"input_ids": generated}
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        outputs = model(**call_kwargs)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        if logits is None:
            raise ValueError("model forward output did not include logits")
        next_token = _next_token_from_logits(
            torch,
            logits,
            input_ids=generated,
            logits_processor=logits_processor,
        )
        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)],
                dim=-1,
            )
        if eos_token_id is not None:
            try:
                if bool((next_token == eos_token_id).all().item()):
                    break
            except Exception:
                pass
    return generated


def _generate_ids(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    model: Any,
    batch: dict[str, Any],
    *,
    max_new_tokens: int,
    pad_token_id: int | None,
    logits_processor: Any | None = None,
) -> tuple[Any, str, str | None]:
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": pad_token_id,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = transformers.LogitsProcessorList(
            [logits_processor]
        )
    try:
        with _prepare_special_tokens_batch_size_compat(model):
            output_ids = model.generate(**batch, **generate_kwargs)
        method = "model.generate"
        if logits_processor is not None:
            method += "+zspace_repression_softmax"
        return output_ids, method, None
    except Exception as exc:
        reset_report = getattr(logits_processor, "reset_report", None)
        if callable(reset_report):
            reset_report()
        output_ids = _manual_forward_generate(
            torch,
            tokenizer,
            model,
            batch,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        method = "manual_forward_fallback"
        if logits_processor is not None:
            method += "+zspace_repression_softmax"
        return output_ids, method, f"{exc.__class__.__name__}: {exc}"


@contextlib.contextmanager
def _prepare_special_tokens_batch_size_compat(model: Any):
    prepare = getattr(model, "_prepare_special_tokens", None)
    if not callable(prepare):
        yield False
        return
    try:
        signature = inspect.signature(prepare)
    except (TypeError, ValueError):
        yield False
        return
    parameters = signature.parameters
    accepts_batch_size = "batch_size" in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if accepts_batch_size:
        yield False
        return

    sentinel = object()
    previous = getattr(model, "_prepare_special_tokens", sentinel)

    def _compat_prepare_special_tokens(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("batch_size", None)
        return prepare(*args, **kwargs)

    try:
        setattr(model, "_prepare_special_tokens", _compat_prepare_special_tokens)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            if previous is sentinel:
                delattr(model, "_prepare_special_tokens")
            else:
                setattr(model, "_prepare_special_tokens", previous)
        except Exception:
            pass


def _run_local_hf(args: argparse.Namespace, adapter: dict[str, Any]) -> dict[str, Any]:
    if args.local_model is None:
        return {"status": "skipped", "reason": "no --local-model supplied"}
    try:
        import torch
        import transformers
    except Exception as exc:
        return {
            "status": "error",
            "stage": "import",
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    load_kwargs = {
        "local_files_only": not bool(args.allow_remote),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    model_path = str(args.local_model)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        model.eval()
        encoded = tokenizer(args.prompt, return_tensors="pt")
        batch = _move_batch_to_device(dict(encoded), _model_device(model))
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        with torch.no_grad():
            baseline_ids, baseline_method, baseline_error = _generate_ids(
                torch,
                transformers,
                tokenizer,
                model,
                batch,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=pad_token_id,
            )
        processor_kwargs = st.zspace_inference_distortion_processor_kwargs(adapter)
        processor = st.build_zspace_repression_logits_processor(**processor_kwargs)
        hook_config = dict(adapter.get("activation_hook") or {})
        if args.activation_module_name:
            hook_config["module_names"] = list(args.activation_module_name)
        if args.activation_name_contains:
            hook_config["name_contains"] = list(args.activation_name_contains)
        hook = st.build_zspace_activation_probe_hook(**hook_config).attach(model)
        try:
            with torch.no_grad():
                distorted_ids, distorted_method, distorted_error = _generate_ids(
                    torch,
                    transformers,
                    tokenizer,
                    model,
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_token_id,
                    logits_processor=processor,
                )
            activation_report = hook.report(limit=16)
        finally:
            hook.close()
        baseline_text = _text_from_tokens(tokenizer, baseline_ids[0])
        distorted_text = _text_from_tokens(tokenizer, distorted_ids[0])
        return {
            "status": "ok",
            "model": model_path,
            "baseline_text": baseline_text,
            "distorted_text": distorted_text,
            "changed": bool(baseline_text != distorted_text),
            "baseline_method": baseline_method,
            "baseline_fallback_error": baseline_error,
            "distorted_method": distorted_method,
            "distorted_fallback_error": distorted_error,
            "processor_kwargs": processor_kwargs,
            "generation_control": processor.report(limit=16),
            "activation_report": activation_report,
        }
    except Exception as exc:
        return {
            "status": "error",
            "stage": "local_hf",
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _api_invoke(args: argparse.Namespace):
    if args.api_provider == "openai-responses":
        return st.make_openai_responses_invoke(
            model=args.api_model,
            max_output_tokens=args.api_max_tokens,
        )
    if args.api_provider == "openai-chat":
        return st.make_openai_chat_invoke(
            model=args.api_model,
            max_tokens=args.api_max_tokens,
        )
    if args.api_provider == "anthropic":
        return st.make_anthropic_messages_invoke(
            model=args.api_model,
            max_tokens=args.api_max_tokens,
        )

    def _fake(prompt: str, **request: Any) -> dict[str, Any]:
        temperature = request.get("temperature")
        top_p = request.get("top_p")
        return {
            "model": args.api_model or "fake-distorted-api",
            "output_text": (
                "Fake API distortion route: "
                f"temperature={temperature} top_p={top_p}. "
                "SpiralTorch telemetry is visible before the live call."
            ),
            "usage": {"prompt_tokens": 32, "completion_tokens": 24, "total_tokens": 56},
        }

    return _fake


def _run_api(args: argparse.Namespace, adapter: dict[str, Any]) -> dict[str, Any]:
    runtime = st.ApiLLMZSpaceRuntime(
        [0.12, -0.04, 0.33, -0.11],
        provider=args.api_provider,
        model=args.api_model,
        create_session=False,
    )
    invoke = _api_invoke(args)
    trace = runtime.call(
        invoke,
        args.prompt,
        provider=args.api_provider,
        model=args.api_model,
        runtime_adapter=adapter,
        context_prompt=True,
        context_prompt_options={"max_telemetry": 32},
    )
    payload = trace.as_dict()
    request_filter = getattr(invoke, "last_request_filter", None)
    if isinstance(request_filter, Mapping):
        payload["request_filter"] = dict(request_filter)
    return payload


def _probe_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "desire_pressure": float(args.desire_pressure),
        "desire_stability": float(args.desire_stability),
        "psi_total": float(args.psi_total),
        "coherence": float(args.coherence),
        "distortion_strength": float(args.distortion_strength),
        "base_temperature": float(args.base_temperature),
        "base_top_p": float(args.base_top_p),
        "include_penalties": bool(args.include_penalties),
    }


def _probe_runtime(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "local_model": str(args.local_model) if args.local_model is not None else None,
        "allow_remote": bool(args.allow_remote),
        "trust_remote_code": bool(args.trust_remote_code),
        "max_new_tokens": int(args.max_new_tokens),
        "activation_module_name": list(args.activation_module_name),
        "activation_name_contains": list(args.activation_name_contains),
        "api_provider": args.api_provider,
        "api_model": args.api_model,
        "api_max_tokens": int(args.api_max_tokens),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = st.api_llm_zspace_inference_distortion_adapter(
        desire_pressure=args.desire_pressure,
        desire_stability=args.desire_stability,
        psi_total=args.psi_total,
        coherence=args.coherence,
        distortion_strength=args.distortion_strength,
        base_temperature=args.base_temperature,
        base_top_p=args.base_top_p,
        include_penalties=args.include_penalties,
        activation_module_names=args.activation_module_name,
        activation_name_contains=args.activation_name_contains,
    )
    report = {
        "row_type": "zspace_inference_distortion_probe",
        "prompt": args.prompt,
        "config": _probe_config(args),
        "runtime": _probe_runtime(args),
        "handoff": getattr(args, "sweep_handoff", None),
        "adapter": adapter,
        "local_hf": _run_local_hf(args, adapter),
        "api": _run_api(args, adapter),
    }
    if args.out is not None:
        report["probe_path"] = str(args.out)
    report["summary"] = st.summarize_zspace_inference_distortion_probe(report)
    report["summary_lines"] = st.summarize_zspace_inference_distortion_probe_lines(
        report
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
