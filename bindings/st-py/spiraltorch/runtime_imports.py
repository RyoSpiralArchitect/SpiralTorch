from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterable, Mapping

__all__ = [
    "RUNTIME_IMPORT_INSTALL_HINTS",
    "TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS",
    "csv_label",
    "csv_values",
    "unique_csv_values",
    "runtime_import_preset_modules",
    "runtime_import_preset_module_map",
    "runtime_import_preset_module_rows",
    "runtime_import_preset_modules_label",
    "runtime_import_preset_missing_modules_label",
    "runtime_import_preset_status_rows",
    "required_runtime_import_presets_from_source",
    "required_runtime_import_presets_from_args",
    "required_runtime_imports_from_source",
    "required_runtime_imports_from_args",
    "module_file",
    "module_name",
    "module_version",
    "runtime_import_coimport_status",
    "runtime_import_install_hint",
    "runtime_import_install_hints_label",
    "runtime_import_kv_label",
    "runtime_import_names_from_source",
    "runtime_import_names_from_args",
    "runtime_import_probe",
    "runtime_import_probe_fields",
    "runtime_import_probe_rows",
    "runtime_import_preflight_report",
    "runtime_import_preflight_summary_lines",
    "runtime_import_presets_from_source",
    "runtime_import_presets_from_args",
    "runtime_imports_from_args",
    "runtime_imports_from_source",
    "runtime_import_required_gate_fields",
    "runtime_import_requirement_failures",
]

TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS: dict[str, list[str]] = {
    "transformers": ["transformers"],
    "torch": ["torch"],
    "tokenizers": ["tokenizers"],
    "torch-transformers": ["transformers", "torch"],
    "hf-runtime": ["transformers", "torch", "tokenizers"],
    "hf-datasets": ["transformers", "torch", "tokenizers", "datasets"],
    "hf-finetune": [
        "transformers",
        "torch",
        "tokenizers",
        "datasets",
        "accelerate",
        "safetensors",
    ],
    "hf-peft": [
        "transformers",
        "torch",
        "tokenizers",
        "accelerate",
        "peft",
        "safetensors",
    ],
}

RUNTIME_IMPORT_INSTALL_HINTS: dict[str, str] = {
    "accelerate": "pip install accelerate",
    "datasets": "pip install datasets",
    "peft": "pip install peft",
    "safetensors": "pip install safetensors",
    "tokenizers": "pip install tokenizers",
    "torch": "pip install torch",
    "transformers": "pip install transformers",
}


def csv_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(",") if part and part != "none"]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item) and str(item) != "none"]
    return [str(value)] if str(value) else []


def unique_csv_values(value: object) -> list[str]:
    return list(dict.fromkeys(csv_values(value)))


def unique_stripped_values(value: object) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in csv_values(value)
            if item.strip()
        )
    )


def _runtime_import_source_values(
    source: object,
    key: str | None,
) -> list[str]:
    if key is None:
        return []
    if isinstance(source, Mapping):
        return csv_values(source.get(key))
    return csv_values(getattr(source, key, None))


def csv_label(values: object) -> str:
    items = csv_values(values)
    return ",".join(items) if items else "none"


def module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)


def module_name(module: object) -> str | None:
    name = getattr(module, "__name__", None)
    return None if name is None else str(name)


def module_file(module: object) -> str | None:
    file = getattr(module, "__file__", None)
    return None if file is None else str(file)


def runtime_import_probe(name: object) -> dict[str, object]:
    module_name_value = str(name).strip()
    try:
        module = importlib.import_module(module_name_value)
    except Exception as exc:  # pragma: no cover - import failures vary by environment.
        return {
            "module": module_name_value,
            "imported": False,
            "version": None,
            "module_name": None,
            "module_file": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    return {
        "module": module_name_value,
        "imported": True,
        "version": module_version(module),
        "module_name": module_name(module),
        "module_file": module_file(module),
        "error": None,
    }


def runtime_import_probe_rows(names: object) -> list[dict[str, object]]:
    return [runtime_import_probe(name) for name in unique_stripped_values(names)]


def runtime_import_coimport_status(
    probes: Iterable[Mapping[str, object]],
) -> str:
    rows = [probe for probe in probes if isinstance(probe, Mapping)]
    if not rows:
        return "not_requested"
    if any(probe.get("imported") is not True for probe in rows):
        return "missing"
    return "ok"


def runtime_import_kv_label(
    probes: Iterable[Mapping[str, object]],
    key: str,
) -> str:
    return csv_label(
        [
            f"{probe['module']}={probe[key] if probe.get(key) is not None else 'none'}"
            for probe in probes
            if isinstance(probe, Mapping) and probe.get("module")
        ]
    )


def runtime_import_install_hint(
    name: object,
    *,
    install_hints: Mapping[str, str] | None = None,
) -> str | None:
    module_name_value = str(name).strip()
    if not module_name_value:
        return None
    hints = install_hints or RUNTIME_IMPORT_INSTALL_HINTS
    hint = hints.get(module_name_value)
    return str(hint) if hint else None


def runtime_import_install_hints_label(
    names: object,
    *,
    install_hints: Mapping[str, str] | None = None,
) -> str:
    rows = []
    for name in unique_stripped_values(names):
        hint = runtime_import_install_hint(name, install_hints=install_hints)
        if hint:
            rows.append(f"{name}={hint}")
    return csv_label(rows)


def runtime_import_preset_modules(
    presets: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    rows = []
    for preset in unique_csv_values(presets):
        modules = [str(module) for module in module_map.get(preset, [])]
        rows.append(f"{preset}={'|'.join(modules) or 'none'}")
    return rows


def runtime_import_preset_module_map(value: object) -> dict[str, str]:
    modules = {}
    for item in csv_values(value):
        preset, sep, _module_list = item.partition("=")
        if preset and sep:
            modules[preset] = item
    return modules


def runtime_import_preset_module_rows(value: object, presets: object) -> list[str]:
    module_map = runtime_import_preset_module_map(value)
    return [
        module_map[preset]
        for preset in csv_values(presets)
        if preset in module_map
    ]


def runtime_import_preset_status_rows(
    presets: object,
    probes: Iterable[Mapping[str, object]],
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, object]]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    probes_by_module = {
        str(probe["module"]): probe
        for probe in probes
        if isinstance(probe, Mapping) and probe.get("module")
    }
    rows = []
    for preset in unique_csv_values(presets):
        modules = [str(module) for module in module_map.get(preset, [])]
        imported = [
            module
            for module in modules
            if probes_by_module.get(module, {}).get("imported") is True
        ]
        missing = [module for module in modules if module not in imported]
        rows.append(
            {
                "preset": preset,
                "modules": modules,
                "imported": imported,
                "missing": missing,
                "passed": not missing,
            }
        )
    return rows


def runtime_import_preset_modules_label(rows: Iterable[Mapping[str, object]]) -> str:
    return csv_label(
        [
            f"{row['preset']}={'|'.join(row['modules']) or 'none'}"
            for row in rows
        ]
    )


def runtime_import_preset_missing_modules_label(
    rows: Iterable[Mapping[str, object]],
) -> str:
    return csv_label(
        [
            f"{row['preset']}={'|'.join(row['missing']) or 'none'}"
            for row in rows
            if row["missing"]
        ]
    )


def runtime_import_presets_from_source(
    source: object,
    *,
    runtime_import_presets_key: str = "runtime_import_presets",
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    return unique_stripped_values(
        [
            *_runtime_import_source_values(source, runtime_import_presets_key),
            *_runtime_import_source_values(source, required_runtime_import_presets_key),
        ]
    )


def runtime_import_presets_from_args(args: object) -> list[str]:
    return runtime_import_presets_from_source(args)


def runtime_imports_from_source(
    source: object,
    *,
    runtime_imports_key: str = "runtime_imports",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, runtime_imports_key)
    )


def runtime_imports_from_args(args: object) -> list[str]:
    return runtime_imports_from_source(args)


def required_runtime_imports_from_source(
    source: object,
    *,
    required_runtime_imports_key: str = "required_runtime_imports",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, required_runtime_imports_key)
    )


def required_runtime_imports_from_args(args: object) -> list[str]:
    return required_runtime_imports_from_source(args)


def required_runtime_import_presets_from_source(
    source: object,
    *,
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    return unique_stripped_values(
        _runtime_import_source_values(source, required_runtime_import_presets_key)
    )


def required_runtime_import_presets_from_args(args: object) -> list[str]:
    return required_runtime_import_presets_from_source(args)


def runtime_import_names_from_source(
    source: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
    runtime_imports_key: str = "runtime_imports",
    runtime_import_presets_key: str = "runtime_import_presets",
    required_runtime_imports_key: str = "required_runtime_imports",
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> list[str]:
    module_map = preset_modules or TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS
    names = []
    for preset in runtime_import_presets_from_source(
        source,
        runtime_import_presets_key=runtime_import_presets_key,
        required_runtime_import_presets_key=required_runtime_import_presets_key,
    ):
        names.extend(str(module) for module in module_map.get(preset, []))
    names.extend(_runtime_import_source_values(source, runtime_imports_key))
    names.extend(_runtime_import_source_values(source, required_runtime_imports_key))
    return unique_stripped_values(names)


def runtime_import_names_from_args(
    args: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    return runtime_import_names_from_source(args, preset_modules=preset_modules)


def runtime_import_probe_fields(
    source: object,
    *,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
    field_prefix: str = "",
    runtime_imports_key: str = "runtime_imports",
    runtime_import_presets_key: str = "runtime_import_presets",
    required_runtime_imports_key: str = "required_runtime_imports",
    required_runtime_import_presets_key: str = "required_runtime_import_presets",
) -> dict[str, object]:
    presets = runtime_import_presets_from_source(
        source,
        runtime_import_presets_key=runtime_import_presets_key,
        required_runtime_import_presets_key=required_runtime_import_presets_key,
    )
    probes = runtime_import_probe_rows(
        runtime_import_names_from_source(
            source,
            preset_modules=preset_modules,
            runtime_imports_key=runtime_imports_key,
            runtime_import_presets_key=runtime_import_presets_key,
            required_runtime_imports_key=required_runtime_imports_key,
            required_runtime_import_presets_key=required_runtime_import_presets_key,
        )
    )
    preset_status = runtime_import_preset_status_rows(
        presets,
        probes,
        preset_modules=preset_modules,
    )
    imported = [probe["module"] for probe in probes if probe["imported"]]
    failed = [probe["module"] for probe in probes if not probe["imported"]]
    satisfied_presets = [row["preset"] for row in preset_status if row["passed"]]
    failed_presets = [row["preset"] for row in preset_status if not row["passed"]]
    coimport_status = runtime_import_coimport_status(probes)
    fields: dict[str, object] = {
        f"{field_prefix}runtime_import_presets": csv_label(presets),
        f"{field_prefix}runtime_import_preset_modules": (
            runtime_import_preset_modules_label(preset_status)
        ),
        f"{field_prefix}runtime_import_presets_satisfied": (
            csv_label(satisfied_presets)
        ),
        f"{field_prefix}runtime_import_presets_failed": csv_label(failed_presets),
        f"{field_prefix}runtime_import_preset_missing_modules": (
            runtime_import_preset_missing_modules_label(preset_status)
        ),
        f"{field_prefix}runtime_imports_requested": (
            csv_label([probe["module"] for probe in probes])
        ),
        f"{field_prefix}runtime_import_probe_count": len(probes),
        f"{field_prefix}runtime_imports_imported": csv_label(imported),
        f"{field_prefix}runtime_imports_failed": csv_label(failed),
        f"{field_prefix}runtime_imports_all_ok": not failed,
        f"{field_prefix}runtime_import_coimport_status": coimport_status,
        f"{field_prefix}runtime_imports_coimported": coimport_status == "ok",
        f"{field_prefix}runtime_import_coimport_modules": csv_label(imported),
        f"{field_prefix}runtime_import_coimport_missing_modules": csv_label(failed),
        f"{field_prefix}runtime_import_versions": (
            runtime_import_kv_label(probes, "version")
        ),
        f"{field_prefix}runtime_import_install_hints": (
            runtime_import_install_hints_label([probe["module"] for probe in probes])
        ),
        f"{field_prefix}runtime_import_failed_install_hints": (
            runtime_import_install_hints_label(failed)
        ),
        f"{field_prefix}runtime_import_module_names": (
            runtime_import_kv_label(probes, "module_name")
        ),
        f"{field_prefix}runtime_imports_json": json.dumps(
            probes,
            ensure_ascii=False,
            sort_keys=True,
        ),
        f"{field_prefix}runtime_import_preset_status_json": json.dumps(
            preset_status,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    fields.update(
        runtime_import_required_gate_fields(
            required_runtime_imports_from_source(
                source,
                required_runtime_imports_key=required_runtime_imports_key,
            ),
            required_runtime_import_presets_from_source(
                source,
                required_runtime_import_presets_key=(
                    required_runtime_import_presets_key
                ),
            ),
            probes=probes,
            preset_status=preset_status,
            field_prefix=field_prefix,
        )
    )
    return fields


def runtime_import_required_gate_fields(
    required_imports: object,
    required_presets: object,
    *,
    imported_modules: object | None = None,
    observed_presets: object | None = None,
    satisfied_presets: object | None = None,
    failed_presets: object | None = None,
    probes: Iterable[Mapping[str, object]] | None = None,
    preset_status: Iterable[Mapping[str, object]] | None = None,
    field_prefix: str = "",
    include_failed_presets: bool = False,
) -> dict[str, object]:
    required = unique_csv_values(required_imports)
    required_preset_rows = unique_csv_values(required_presets)
    if probes is not None:
        imported = [
            str(probe["module"])
            for probe in probes
            if isinstance(probe, Mapping)
            and probe.get("module")
            and probe.get("imported") is True
        ]
    else:
        imported = unique_csv_values(imported_modules)

    if preset_status is not None:
        status_rows = [row for row in preset_status if isinstance(row, Mapping)]
        observed = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset")
        ]
        satisfied = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset") and row.get("passed") is True
        ]
        failed = [
            str(row["preset"])
            for row in status_rows
            if row.get("preset") and row.get("passed") is not True
        ]
    else:
        observed = unique_csv_values(observed_presets)
        satisfied = unique_csv_values(satisfied_presets)
        failed = unique_csv_values(failed_presets)

    missing = [module for module in required if module not in imported]
    missing_presets = [
        preset for preset in required_preset_rows if preset not in observed
    ]
    unsatisfied_presets = [
        preset
        for preset in required_preset_rows
        if preset in observed and preset not in satisfied
    ]
    gate_requested = bool(required)
    preset_gate_requested = bool(required_preset_rows)
    fields: dict[str, object] = {
        f"{field_prefix}required_runtime_imports": csv_label(required),
        f"{field_prefix}required_runtime_imports_imported": (
            csv_label(imported) if gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_imports_missing": (
            csv_label(missing) if gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_imports_passed": (
            None if not gate_requested else not missing
        ),
        f"{field_prefix}required_runtime_import_presets": (
            csv_label(required_preset_rows)
        ),
        f"{field_prefix}required_runtime_import_presets_observed": (
            csv_label(observed) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_satisfied": (
            csv_label(satisfied) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_missing": (
            csv_label(missing_presets) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_unsatisfied": (
            csv_label(unsatisfied_presets) if preset_gate_requested else "none"
        ),
        f"{field_prefix}required_runtime_import_presets_passed": (
            None
            if not preset_gate_requested
            else not missing_presets and not unsatisfied_presets
        ),
    }
    if include_failed_presets:
        fields[f"{field_prefix}required_runtime_import_presets_failed"] = (
            csv_label(failed) if preset_gate_requested else "none"
        )
    return fields


def runtime_import_requirement_failures(
    row: Mapping[str, object],
    *,
    field_prefix: str = "",
    failure_prefix: str = "runtime_import",
) -> list[str]:
    failures = []
    if row.get(f"{field_prefix}required_runtime_imports_passed") is False:
        for module_name in sorted(
            csv_values(row.get(f"{field_prefix}required_runtime_imports_missing"))
        ):
            failures.append(f"{failure_prefix}_missing:{module_name}")
    if row.get(f"{field_prefix}required_runtime_import_presets_passed") is False:
        for preset in sorted(
            csv_values(
                row.get(f"{field_prefix}required_runtime_import_presets_missing")
            )
        ):
            failures.append(f"{failure_prefix}_preset_missing:{preset}")
        for preset in sorted(
            csv_values(
                row.get(
                    f"{field_prefix}required_runtime_import_presets_unsatisfied"
                )
            )
        ):
            failures.append(f"{failure_prefix}_preset_unsatisfied:{preset}")
    return failures


def runtime_import_preflight_report(
    *,
    runtime_imports: object = None,
    runtime_import_presets: object = None,
    required_runtime_imports: object = None,
    required_runtime_import_presets: object = None,
    require_all: bool = False,
    preset_modules: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, object]:
    requested_imports = unique_stripped_values(runtime_imports)
    requested_presets = unique_stripped_values(runtime_import_presets)
    required_imports = unique_stripped_values(required_runtime_imports)
    required_presets = unique_stripped_values(required_runtime_import_presets)
    if require_all:
        required_imports = unique_stripped_values(
            [*requested_imports, *required_imports]
        )
        required_presets = unique_stripped_values(
            [*requested_presets, *required_presets]
        )

    report = runtime_import_probe_fields(
        {
            "runtime_imports": requested_imports,
            "runtime_import_presets": requested_presets,
            "required_runtime_imports": required_imports,
            "required_runtime_import_presets": required_presets,
        },
        preset_modules=preset_modules,
    )
    failures = runtime_import_requirement_failures(report)
    report.update(
        {
            "runtime_import_preflight_required": bool(
                required_imports or required_presets
            ),
            "runtime_import_preflight_require_all": bool(require_all),
            "runtime_import_preflight_failures": csv_label(failures),
            "runtime_import_preflight_passed": not failures,
        }
    )
    return report


def runtime_import_preflight_summary_lines(
    report: Mapping[str, object],
) -> list[str]:
    lines = [
        (
            "runtime_import_preflight "
            f"passed={report.get('runtime_import_preflight_passed')} "
            f"required={report.get('runtime_import_preflight_required')} "
            f"require_all={report.get('runtime_import_preflight_require_all')}"
        ),
        (
            "runtime_imports "
            f"requested={report.get('runtime_imports_requested', 'none')} "
            f"imported={report.get('runtime_imports_imported', 'none')} "
            f"failed={report.get('runtime_imports_failed', 'none')} "
            f"all_ok={report.get('runtime_imports_all_ok')}"
        ),
        (
            "runtime_import_presets "
            f"requested={report.get('runtime_import_presets', 'none')} "
            f"satisfied={report.get('runtime_import_presets_satisfied', 'none')} "
            f"failed={report.get('runtime_import_presets_failed', 'none')} "
            "missing_modules="
            f"{report.get('runtime_import_preset_missing_modules', 'none')}"
        ),
    ]
    install_hints = str(report.get("runtime_import_failed_install_hints", "none"))
    if install_hints and install_hints != "none":
        lines.append(f"runtime_import_failed_install_hints {install_hints}")
    failures = str(report.get("runtime_import_preflight_failures", "none"))
    if failures and failures != "none":
        lines.append(f"runtime_import_preflight_failures {failures}")
    return lines


def _runtime_import_arg_parser() -> argparse.ArgumentParser:
    preset_choices = sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS)
    parser = argparse.ArgumentParser(
        description=(
            "Probe optional Python runtime imports for SpiralTorch interop and "
            "fine-tuning handoffs."
        ),
    )
    parser.add_argument(
        "--runtime-import",
        "--import",
        dest="runtime_imports",
        action="append",
        default=[],
        help="Python module to import. May be repeated.",
    )
    parser.add_argument(
        "--runtime-import-preset",
        "--preset",
        dest="runtime_import_presets",
        action="append",
        choices=preset_choices,
        default=[],
        help=(
            "Named import bundle to probe. Use hf-runtime for Transformers, "
            "hf-finetune for datasets/accelerate/safetensors, or hf-peft for "
            "PEFT adapter workflows."
        ),
    )
    parser.add_argument(
        "--require",
        dest="require_all",
        action="store_true",
        help="Fail unless every requested import and preset is satisfied.",
    )
    parser.add_argument(
        "--require-runtime-import",
        "--require-import",
        dest="required_runtime_imports",
        action="append",
        default=[],
        help="Module that must import successfully. May be repeated.",
    )
    parser.add_argument(
        "--require-runtime-import-preset",
        "--require-preset",
        dest="required_runtime_import_presets",
        action="append",
        choices=preset_choices,
        default=[],
        help="Preset that must be satisfied. May be repeated.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available preset expansions and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of summary lines.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output; exit code still reflects gates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _runtime_import_arg_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        if args.json:
            print(
                json.dumps(
                    TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif not args.quiet:
            for row in runtime_import_preset_modules(
                sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS)
            ):
                print(row)
        return 0

    report = runtime_import_preflight_report(
        runtime_imports=args.runtime_imports,
        runtime_import_presets=args.runtime_import_presets,
        required_runtime_imports=args.required_runtime_imports,
        required_runtime_import_presets=args.required_runtime_import_presets,
        require_all=args.require_all,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    elif not args.quiet:
        for line in runtime_import_preflight_summary_lines(report):
            print(line)
    return 0 if report["runtime_import_preflight_passed"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke.
    raise SystemExit(main(sys.argv[1:]))
