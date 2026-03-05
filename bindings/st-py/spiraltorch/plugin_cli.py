"""Command-line helpers for SpiralTorch's plugin registry.

This CLI introspects the in-process plugin registry and can optionally load
plugins from entry points or a filesystem path before emitting summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import spiraltorch as st

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spiral-plugin", description=__doc__)
    parser.add_argument(
        "--load-entrypoints",
        action="append",
        default=[],
        metavar="GROUP",
        help="Load plugins from Python entrypoints before running the command.",
    )
    parser.add_argument(
        "--load-path",
        action="append",
        default=[],
        metavar="PATH",
        help="Load plugins from a filesystem path before running the command.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scanning when using --load-path.",
    )
    parser.add_argument(
        "--load-strict",
        action="store_true",
        help="Fail fast when a plugin load operation reports no matches or errors.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered plugin IDs.")
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of newline-separated IDs.",
    )

    graph_parser = subparsers.add_parser("graph", help="Print a dependency graph as JSON.")
    graph_parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Only include dependencies that are currently registered.",
    )

    dot_parser = subparsers.add_parser("dot", help="Print a dependency graph as Graphviz DOT.")
    dot_parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Only include dependencies that are currently registered.",
    )
    dot_parser.add_argument(
        "--no-missing",
        action="store_true",
        help="Omit nodes/edges to dependencies that are not registered.",
    )
    dot_parser.add_argument(
        "--rankdir",
        default="LR",
        help="Graphviz rankdir (LR, RL, TB, BT). Default: LR.",
    )

    explain_parser = subparsers.add_parser("explain", help="Explain a plugin's deps/dependents as JSON.")
    explain_parser.add_argument("plugin_id", help="Plugin ID to explain.")

    validate_parser = subparsers.add_parser("validate", help="Validate dependencies for missing deps/cycles.")
    validate_parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Ignore dependencies that are not currently registered.",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero and raise on validation errors.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary.",
    )

    return parser


def _load_plugins(args: argparse.Namespace) -> None:
    recursive = not bool(getattr(args, "no_recursive", False))
    strict = bool(getattr(args, "load_strict", False))

    for group in getattr(args, "load_entrypoints", []) or []:
        st.plugin.load_entrypoints(group=group, instantiate=True, replace=False)

    for path in getattr(args, "load_path", []) or []:
        st.plugin.load_path(
            path,
            recursive=recursive,
            instantiate=True,
            strict=strict,
            reload=False,
            replace=False,
            module_prefix="spiraltorch_cli_",
            add_sys_path=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        _load_plugins(args)
    except Exception as exc:
        print(f"[spiral-plugin] load failed: {exc}", file=sys.stderr)
        return 2

    try:
        cmd = args.command
        if cmd == "list":
            plugins = sorted(st.plugin.list_plugins())
            if args.json:
                print(json.dumps(plugins, indent=2, sort_keys=True))
            else:
                for pid in plugins:
                    print(pid)
            return 0

        if cmd == "graph":
            graph = st.plugin.dependency_graph(internal_only=bool(args.internal_only))
            print(json.dumps(graph, indent=2, sort_keys=True))
            return 0

        if cmd == "dot":
            dot = st.plugin.dependency_dot(
                internal_only=bool(args.internal_only),
                show_missing=not bool(args.no_missing),
                rankdir=str(args.rankdir),
            )
            print(dot, end="")
            return 0

        if cmd == "explain":
            info = st.plugin.explain(str(args.plugin_id))
            print(json.dumps(info, indent=2, sort_keys=True))
            return 0

        if cmd == "validate":
            strict = bool(args.strict)
            summary = st.plugin.validate_dependencies(
                internal_only=bool(args.internal_only),
                strict=strict,
            )
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                ok = bool(summary.get("ok"))
                print("ok" if ok else "invalid")
                missing = summary.get("missing", {}) or {}
                cycles = summary.get("cycles", []) or []
                if missing:
                    print(f"missing: {len(missing)} plugin(s)")
                if cycles:
                    print(f"cycles: {len(cycles)}")
            return 0 if bool(summary.get("ok")) else 1

        parser.error(f"Unknown command {cmd!r}")
        return 2
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        print(f"[spiral-plugin] {exc}", file=sys.stderr)
        return 2

