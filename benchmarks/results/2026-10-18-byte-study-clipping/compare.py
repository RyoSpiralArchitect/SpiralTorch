"""Run all four fixed clipping controls; retain failures without relaxing gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import byte_corpus_study as protocol

LABELS = ("sgd-off", "sgd-active", "adamw-off", "adamw-active")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "browser"))
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.mode == "native" and args.wheel is None:
        parser.error("native requires --wheel and its isolated installed interpreter")
    args.output.mkdir(mode=0o700, exist_ok=False)
    browser = protocol.decode_json((args.fixture / "training.json").read_bytes()) if args.mode == "browser" else None
    rows = []
    for name in LABELS:
        request = args.fixture / (name + ".json")
        reference = args.output / (name + "-torch.json")
        row = dict(name=name, request_sha256=protocol.digest(request.read_bytes()))
        try:
            if args.mode == "native":
                run = args.output / name
                with (args.output / (name + ".log")).open("xb") as log:
                    subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tools/qualify_byte_corpus_runs.py"),
                        "run", str(request), str(run), "--stop", "1", "--wheel", str(args.wheel)],
                        check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
                actual = (run / "continuous/report-2.json").read_bytes()
                row["recovery"] = protocol.decode_json((run / "result.json").read_bytes())
            else:
                saved, = [r for r in browser["runs"] if r["name"] == name]
                actual = saved["report_json"].encode()
            protocol.reference(SimpleNamespace(request=request, output=reference))
            comparison = protocol.compare(request.read_bytes(), protocol.decode_json(reference.read_bytes()),
                                          protocol.decode_json(actual))
            protocol.write_new(args.output / (name + "-comparison.json"), comparison)
            row.update(passed=True, report_sha256=protocol.digest(actual),
                       reference_sha256=protocol.digest(reference.read_bytes()),
                       comparison_sha256=protocol.digest(protocol.encoded(comparison)))
        except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
            row.update(passed=False, error=str(error))
        rows.append(row)
        print(json.dumps(dict(name=name, passed=row["passed"])), flush=True)
    protocol.write_new(args.output / "matrix.json", dict(schema="spiraltorch.byte_clipping.matrix.v1",
        mode=args.mode, rows=rows, passed=all(row["passed"] for row in rows),
        scope="Two-update synthetic mechanism check; no quality or performance claim"))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
