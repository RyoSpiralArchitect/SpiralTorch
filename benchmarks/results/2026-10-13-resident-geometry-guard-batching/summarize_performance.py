"""Validate the complete frozen timing screen and publish scalar-only results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(protocol_bytes, records, native_report):
    p = json.loads(protocol_bytes)
    require(p["schema"] == "spiraltorch.geometry_guard_batching.performance_protocol.v1", "protocol schema")
    require(p["blocks"] == 3 and p["warmup_order"] == ["A","B"]
            and p["measurement_order"] == ["A","B","B","A"]
            and p["maximum_block_ratio"] == 1.0 and p["maximum_geomean_ratio"] == .95, "frozen rule")
    expected = [(b,stage,i,arm) for b in range(1,4)
                for stage,order in (("warmup",p["warmup_order"]),("measure",p["measurement_order"]))
                for i,arm in enumerate(order)]
    result = dict(schema="spiraltorch.geometry_guard_batching.performance_screen.v1",
        protocol_sha256=sha(protocol_bytes), fixture_sha256=p["fixture_sha256"], runtimes={}, adopted=False,
        original_browser_model_qualification_passed=False,
        scope="Fixed artifact-pair qualification workload, not kernel latency or general training speed. Full-corpus candidate checks and review remain required for adoption.")
    for engine, timer in (("native","fresh_process_perf_counter_ns_including_output_write"),
                          ("browser","awaited_rust_export_including_gpu_runtime_and_rust_json")):
        documents = records[engine]
        require(len(documents) == (1 if engine == "native" else 3), "missing or duplicate timing blocks")
        rows = []
        for index, document in enumerate(documents):
            require(document["schema"] == "spiraltorch.geometry_guard_batching.timing.v1"
                    and document["engine"] == engine and document["timer"] == timer
                    and document["completed"] is True and "error" not in document and "persistence_error" not in document
                    and document["protocol_sha256"] == sha(protocol_bytes)
                    and document["fixture_sha256"] == p["fixture_sha256"], "incomplete or foreign timing run")
            if engine == "browser":
                require(type(document["block"]) is int and document["block"] == index+1
                        and len(document["samples"]) == 6, "browser page block coverage")
            rows.extend(document["samples"])
        require(len(rows) == len(expected), "incomplete sample count")
        scalars = []
        for row, (block, stage, index, arm) in zip(rows,expected):
            require(row["status"] == "completed" and "error" not in row, "failed or interrupted invocation")
            require(type(row["block"]) is int and type(row["index"]) is int
                    and (row["block"],row["stage"],row["index"],row["arm"]) == (block,stage,index,arm), "sample order/identity")
            seconds = row["seconds"]
            require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0, "invalid duration")
            if engine == "native":
                filename = f"block-{block}-{stage}-{index}-{arm}.json"
                require(type(row["exit_code"]) is int and row["exit_code"] == 0
                        and row["report_file"] == filename, "native child/report identity")
                report = native_report(filename)
            else:
                require(type(row["report_json"]) is str, "missing browser normalized report")
                report = row["report_json"].encode()
            require(sha(report) == row["report_sha256"] == p["report_sha256"][engine], "changed output")
            scalars.append({k:row[k] for k in ("block","stage","index","arm","seconds","report_sha256")})
        blocks = []
        for block in range(1,4):
            medians = {arm:statistics.median(r["seconds"] for r in rows
                if r["block"]==block and r["stage"]=="measure" and r["arm"]==arm) for arm in ("A","B")}
            blocks.append(dict(block=block,median_seconds=medians,ratio_b_over_a=medians["B"]/medians["A"]))
        gm = math.exp(statistics.mean(math.log(b["ratio_b_over_a"]) for b in blocks))
        result["runtimes"][engine] = dict(timer=timer, samples=scalars, blocks=blocks,
            geometric_mean_ratio=gm, all_output_bytes_exact=True,
            screen_passed=gm <= p["maximum_geomean_ratio"]
                and all(b["ratio_b_over_a"] <= p["maximum_block_ratio"] for b in blocks))
    result["combined_screen_passed"] = all(r["screen_passed"] for r in result["runtimes"].values())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","raw","output"):
        p.add_argument(name,type=Path)
    a = p.parse_args()
    records = {"native":[json.loads((a.raw/"native/timing.json").read_bytes())],
        "browser":[json.loads((a.raw/f"browser-block-{i}.json").read_bytes()) for i in range(1,4)]}
    result = summarize(a.protocol.read_bytes(), records, lambda name:(a.raw/"native"/name).read_bytes())
    with a.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print({e:r["screen_passed"] for e,r in result["runtimes"].items()})


if __name__ == "__main__":
    main()
