"""Run the frozen fresh-process artifact comparison; serialize GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "baseline", "candidate", "fixture", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    protocol_bytes = a.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    require(protocol["schema"] == "spiraltorch.geometry_guard_batching.performance_protocol.v1", "protocol schema")
    require(protocol["blocks"] == 3 and protocol["warmup_order"] == ["A","B"]
            and protocol["measurement_order"] == ["A","B","B","A"], "frozen schedule")
    binaries = {"A":a.baseline.resolve(), "B":a.candidate.resolve()}
    require(digest(a.fixture) == protocol["fixture_sha256"], "fixture identity")
    for arm, binary in binaries.items():
        require(digest(binary) == protocol["native_binary_sha256"][arm], "binary identity")
    a.output.mkdir(parents=True, exist_ok=False)
    ledger = a.output / "samples.jsonl"
    record = dict(schema="spiraltorch.geometry_guard_batching.timing.v1", engine="native",
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(), fixture_sha256=digest(a.fixture),
        timer="fresh_process_perf_counter_ns_including_output_write", completed=False, samples=[])

    def append(event, row):
        with ledger.open("a") as stream:
            stream.write(json.dumps(dict(event=event, sample=row), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    try:
        for block in range(1, protocol["blocks"] + 1):
            for stage, order in (("warmup",protocol["warmup_order"]), ("measure",protocol["measurement_order"])):
                for index, arm in enumerate(order):
                    name = f"block-{block}-{stage}-{index}-{arm}"
                    require(digest(a.protocol) == record["protocol_sha256"], "protocol changed")
                    require(digest(binaries[arm]) == protocol["native_binary_sha256"][arm], "binary changed")
                    require(digest(a.fixture) == protocol["fixture_sha256"], "fixture changed")
                    row = dict(block=block, stage=stage, index=index, arm=arm, seconds=None,
                        status="started_or_interrupted", exit_code=None,
                        report_file=name + ".json", log_file=name + ".log")
                    record["samples"].append(row)
                    append("started", row)
                    start = None
                    try:
                        with (a.output / row["report_file"]).open("xb") as out, (a.output / row["log_file"]).open("xb") as err:
                            start = time.perf_counter_ns()
                            try:
                                result = subprocess.run([str(binaries[arm]), str(a.fixture.resolve()), "--query-mixture"],
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False)
                            finally:
                                row["seconds"] = (time.perf_counter_ns() - start) / 1e9
                        row["exit_code"] = result.returncode
                        row["report_sha256"] = digest(a.output / row["report_file"])
                        require(result.returncode == 0, "nonzero child exit; no replacement trial")
                        require(row["report_sha256"] == protocol["report_sha256"]["native"], "report changed; preserve this run")
                        row["status"] = "completed"
                    except BaseException as error:
                        row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        if (a.output / row["report_file"]).is_file():
                            row["report_sha256"] = digest(a.output / row["report_file"])
                        raise
                    finally:
                        append("finished", row)
                    print(json.dumps({k:row[k] for k in ("block","stage","index","arm","seconds","exit_code")}), flush=True)
        record["completed"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (a.output / "timing.json").open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
