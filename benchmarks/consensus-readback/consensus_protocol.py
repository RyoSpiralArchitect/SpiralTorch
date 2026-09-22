"""Fail-closed, complete 2x2 comparisons; timings are descriptive, not kernel clocks."""
import hashlib
import math
import statistics
import struct

SCHEMA = "spiraltorch.consensus_readback_bench.v1"
TORCH = "spiraltorch.consensus_torch_bench.v1"
SUMMARY = "spiraltorch.consensus_readback_summary.v1"
CONTROL_TOKENS = "ed511b0ea291e3e34612334bceafa1340b1f35961a9cf24320f383e0ac2f6e39"
KEYS = {(r, c, n) for r, c in [(1, 31), (1, 256), (17, 257), (65, 1025),
                                (128, 1025), (17, 4096)] for n in (2, 4)}
ATOL, RTOL = 2e-6, 5e-6
VARIANTS = ("separate_separate", "paired_separate", "separate_batch", "paired_batch")
ROUTES = tuple(f + "_" + v for f in ("native", "browser") for v in VARIANTS) + ("cpu", "mps")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def f32(values):
    if any(not finite(v) for v in values):
        raise ValueError("nonfinite or nonnumeric array")
    return struct.pack("<" + "f" * len(values), *values)


def rounded(values):
    return list(struct.unpack("<" + "f" * len(values), f32(values)))


def key(case):
    result = tuple(case[k] for k in ("rows", "cols", "count"))
    if any(type(v) is not int for v in result) or result not in KEYS:
        raise ValueError("unknown shape/count")
    return result


def inputs(rows, cols):
    return [((i * 37 + 17) % 1009 - 504) / 128 for i in range(rows * cols)]


def parameters(cols):
    return rounded([1.618034, .618034, .381966, .75 * .0019295743094039225 * math.sqrt(24),
                    1., 1 / cols, 1e-7])


def oracle(values, rows, cols, count):
    soft, mask, spiral, metrics = [], [], [], []
    phi, conjugate, bias, leech, ratio, inv_cols, epsilon = parameters(cols)
    for row in range(rows):
        x = values[row * cols:(row + 1) * cols]
        top = max(x)
        exps = [math.exp(v - top) for v in x]
        total = math.fsum(exps)
        p = [v / total for v in exps]
        m = [float(v == top) for v in x]
        soft.extend(p)
        mask.extend(m)
        if count == 4:
            entropy = -math.fsum(v * math.log(max(v, epsilon)) for v in p)
            mass = math.fsum(m)
            geodesic = entropy * ratio + mass * phi
            enrichment = leech * geodesic if abs(geodesic) > epsilon else 0.
            clamp = lambda v: min(1., max(0., v))
            coherence = (clamp(entropy / (entropy + 1)) + clamp(mass * inv_cols)
                         + clamp(enrichment / (1 + abs(enrichment)))) / 3
            spiral.extend((1 + enrichment) * (conjugate * a + bias * b) for a, b in zip(p, m))
            metrics.extend([entropy, mass, enrichment, coherence])
    return rounded(soft + mask + spiral + metrics)


def close(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("output length")
    maximum, scaled = 0., 0.
    for a, b in zip(actual, reference):
        if not finite(a) or not finite(b):
            raise ValueError("nonfinite output")
        error = abs(a - b)
        relative = error / (ATOL + RTOL * abs(b))
        if relative > 1:
            raise ValueError("consensus oracle mismatch")
        maximum, scaled = max(maximum, error), max(scaled, relative)
    return maximum, scaled


def order(case, block, torch=False):
    r, c, n = key(case)
    if torch:
        return [0, 1] if (block + r + c + n) % 2 == 0 else [1, 0]
    start = (block + r + c + n) % 4
    result = list(range(4))[start:] + list(range(4))[:start]
    scheme = case.get("order_scheme", "legacy-alternating")
    if scheme not in ("legacy-alternating", "balanced-cycle-v1"):
        raise ValueError("unknown route ordering")
    reverse = block % 2 if scheme == "legacy-alternating" else (block // 4) % 2
    return result[::-1] if reverse else result


def intervals(case, routes, rounds=False):
    expected = {(r, b, n, d) for r in (range(3) if rounds else [0])
                for b in range(9) for n in (1, 4) for d in routes}
    index = {}
    for item in case["intervals"]:
        r, b, n, d = item.get("round", 0), item["block"], item["burst"], item["route"]
        if any(type(v) is not int for v in (r, b, n)):
            raise ValueError("noninteger interval identity")
        ident = r, b, n, d
        if ident not in expected or ident in index:
            raise ValueError("unknown/duplicate interval")
        if not finite(item["elapsed_ms"]) or item["elapsed_ms"] <= 0:
            raise ValueError("invalid duration")
        for field, limit in [("max_scaled_error", 1.), ("max_abs_error", ATOL + RTOL * case["cols"])]:
            if not finite(item[field]) or not 0 <= item[field] <= limit:
                raise ValueError("invalid interval error")
        if (item["order"] != order(case, b + 3, d in ("cpu", "mps"))
                or any(type(i) is not int for i in item["order"])):
            raise ValueError("rotated route order")
        index[ident] = item["elapsed_ms"]
    if set(index) != expected:
        raise ValueError("incomplete intervals")
    return index


def admit(report, family):
    is_torch = family == "torch"
    if family not in ("native", "browser", "torch"):
        raise ValueError("unknown runtime")
    if (report["schema"] != (TORCH if is_torch else SCHEMA) or report["status"] != "passed"
            or report["warmup"] != 3 or report["blocks"] != 9 or report["bursts"] != [1, 4]):
        raise ValueError("protocol identity")
    if is_torch:
        if (report["input_protocol"] != SCHEMA or report["devices"] != ["cpu", "mps"]
                or report["compiled"] is not False or report["preallocated_outputs"] is not True
                or report["intra_op_threads"] != 4 or report["inter_op_threads"] != 1):
            raise ValueError("Torch control identity")
    else:
        digest = hashlib.sha256(" ".join(report["control_shader"].split()).encode()).hexdigest()
        if (digest != CONTROL_TOKENS or report["domain_cases"] != 12
                or report["domain_bitwise_equal"] is not True or report["routes"] != list(VARIANTS)
                or not report["adapter"] or "device_type: Cpu" in report["adapter"]):
            raise ValueError("WGPU domain/control admission")
        expected = dict(ordered_snapshot=True, empty_prefixes=True, preflight_errors=5,
                        unread_drop=True, source_drop=True, pending_map_cancellation=family == "browser")
        if report["ownership"] != expected:
            raise ValueError("snapshot ownership admission")
    if family == "browser":
        if (report["page_errors"] or report["console_messages"]
                or report["browser_adapter_probe"]["is_fallback_adapter"] is not False
                or not report["asset_sha256"] or not report["browser_version"]):
            raise ValueError("browser runtime/probe admission")
        stream = report["consensus_state_artifacts"]
        if (stream["rows"] != 12 or type(stream["bytes"]) is not int or stream["bytes"] <= 0
                or not stream["path"].endswith(".cases.jsonl") or "/" in stream["path"]
                or len(stream["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in stream["sha256"])):
            raise ValueError("streamed case completion")
    routes = ("cpu", "mps") if is_torch else VARIANTS
    cases = {}
    for case in report["cases"]:
        ident = key(case)
        if ident in cases:
            raise ValueError("duplicate condition")
        r, c, n = ident
        data = inputs(r, c)
        if f32(case["input"]) != f32(data):
            raise ValueError("input bytes drift")
        reference = oracle(data, r, c, n)
        close(case["reference"], reference)
        if len(case["last_outputs"]) != len(routes):
            raise ValueError("route output count")
        for output in case["last_outputs"]:
            close(output, reference)
        if not is_torch and any(f32(o) != f32(case["last_outputs"][0]) for o in case["last_outputs"][1:]):
            raise ValueError("WGPU control/candidate output bits differ")
        intervals(case, routes)
        cases[ident] = case
    if set(cases) != KEYS:
        raise ValueError("incomplete grid")
    return cases


FACTORS = {"combined": (0, 3), "batch_with_separate_reduction": (0, 2),
           "batch_with_paired_reduction": (1, 3), "reduction_with_separate_readback": (0, 1),
           "reduction_with_batched_readback": (2, 3)}


def summarize(case):
    index = intervals(case, ROUTES, rounds=True)
    result = []
    for burst in (1, 4):
        medians = {d: statistics.median(v for (_, _, n, route), v in index.items()
                                       if n == burst and route == d) for d in ROUTES}
        ratios = {family: {factor: statistics.median(
            index[r, b, burst, family + "_" + VARIANTS[a]] / index[r, b, burst, family + "_" + VARIANTS[z]]
            for r in range(3) for b in range(9)) for factor, (a, z) in FACTORS.items()}
            for family in ("native", "browser")}
        result.append(dict(burst=burst, median_interval_ms=medians, paired_ratios=ratios))
    return result


def describe(cases):
    result = {}
    for family in ("native", "browser"):
        result[family] = {}
        for count in (2, 4):
            factors = {}
            for factor in FACTORS:
                values = [s["paired_ratios"][family][factor] for c in cases if c["count"] == count for s in c["summary"]]
                factors[factor] = dict(geomean=math.exp(statistics.mean(math.log(v) for v in values)),
                                       min=min(values), max=max(values), cells_over_1=sum(v > 1 for v in values), cells=len(values))
            result[family][str(count)] = factors
    return result


def analyze(native, browser, torch):
    reports = [native, browser, torch]
    if any(len(group) != 3 for group in reports):
        raise ValueError("exactly three complete rounds required")
    admitted = [[admit(r, f) for r in group] for f, group in zip(("native", "browser", "torch"), reports)]
    schemes = {case.get("order_scheme", "legacy-alternating")
               for rounds in admitted for cases in rounds for case in cases.values()}
    if len(schemes) != 1:
        raise ValueError("mixed route ordering protocols")
    scheme = schemes.pop()
    records = []
    for ident in sorted(KEYS):
        r, c, n = ident
        reference = oracle(inputs(r, c), r, c, n)
        record = dict(rows=r, cols=c, count=n, intervals=[],
                      input_sha256=hashlib.sha256(f32(inputs(r, c))).hexdigest(),
                      oracle_sha256=hashlib.sha256(f32(reference)).hexdigest(),
                      max_abs_errors={}, max_scaled_errors={})
        if scheme != "legacy-alternating":
            record["order_scheme"] = scheme
        for family, rounds in zip(("native", "browser", "torch"), admitted):
            routes = ("cpu", "mps") if family == "torch" else tuple(family + "_" + v for v in VARIANTS)
            for repeat, cases in enumerate(rounds):
                case = cases[ident]
                for route, output in zip(routes, case["last_outputs"]):
                    absolute, scaled = close(output, reference)
                    for field, value in [("max_abs_errors", absolute), ("max_scaled_errors", scaled)]:
                        record[field][route] = max(record[field].get(route, 0), value)
                for item in case["intervals"]:
                    route = item["route"] if family == "torch" else family + "_" + item["route"]
                    record["intervals"].append({**item, "route": route, "round": repeat})
        record["summary"] = summarize(record)
        records.append(record)
    return dict(schema=SUMMARY, status="passed", rounds=3, tolerance=dict(atol=ATOL, rtol=RTOL), cases=records,
                descriptive_summary=describe(records), control_tokens_sha256=CONTROL_TOKENS,
                native_metadata=[{k: r[k] for k in ("adapter", "ownership", "domain_cases", "domain_bitwise_equal")} for r in native],
                browser_metadata=[{k: r[k] for k in ("adapter", "ownership", "domain_cases", "domain_bitwise_equal",
                    "browser_adapter_probe", "asset_sha256", "browser_version", "consensus_state_artifacts")} for r in browser],
                torch_metadata=[{k: r[k] for k in ("torch_version", "devices", "intra_op_threads", "inter_op_threads",
                    "compiled", "preallocated_outputs")} for r in torch],
                boundary="Prepared portable f32 softmax/all-peak pair (count=2), or raw GPU consensus plus per-row metrics (count=4). Factorial separate/paired reductions and separate/batched readback. Count=2 never executes consensus: reduction factor is a placebo. Matched Rust ownership for both WGPU readback controls, not byte-identical old native read_buffer. Encoding, submissions and terminal owning CPU completion are timed; setup, checks, JSON and Python lists excluded. Torch eager reusable outputs and scratch, scalar f64 and Torch CPU f64 oracles. Three rotated serial rounds on shared M4; coarse browser clock. No Tensor CPU telemetry blend, isolated kernel clock, torch.compile, training or universal PyTorch superiority claim.")


def validate_summary(result):
    if (result["schema"] != SUMMARY or result["status"] != "passed" or result["rounds"] != 3
            or result["tolerance"] != dict(atol=ATOL, rtol=RTOL) or result["control_tokens_sha256"] != CONTROL_TOKENS):
        raise ValueError("summary identity")
    seen = set()
    for case in result["cases"]:
        ident = key(case)
        if ident in seen:
            raise ValueError("duplicate summary condition")
        seen.add(ident)
        for field in ("input_sha256", "oracle_sha256"):
            value = case[field]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("summary content hash")
        for field, bound in (("max_abs_errors", ATOL + RTOL * case["cols"]), ("max_scaled_errors", 1.)):
            if set(case[field]) != set(ROUTES) or any(not finite(v) or not 0 <= v <= bound for v in case[field].values()):
                raise ValueError("summary numeric gate")
        if case["summary"] != summarize(case):
            raise ValueError("summary does not match all intervals")
    if seen != KEYS or result["descriptive_summary"] != describe(result["cases"]):
        raise ValueError("summary grid/aggregation")
