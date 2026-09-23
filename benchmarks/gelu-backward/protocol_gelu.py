"""Complete liveness comparisons with frozen semantics and descriptive timing."""
import hashlib
import math
import statistics
import struct

SCHEMA = "spiraltorch.gelu_liveness_bench.v1"
TORCH = "spiraltorch.gelu_torch_bench.v1"
SUMMARY = "spiraltorch.gelu_liveness_summary.v1"
KEYS = {(r,c,n) for r,c in [(1,31),(1,256),(17,257),(65,1025),(128,1025),(17,4096)] for n in (1,3)}
DOMAIN_KEYS = {(r,c,n,p) for r,c in [(1,1),(1,31),(3,9),(17,65),(33,257),(2,1025)] for n in (1,3) for p in (0,3)}
ATOL, RTOL = 2e-6, 1e-5
LEGACY = {"legacy_tensor_source":"c87fa86bad37c3a8d3bbd6ce9f8fc20a9e2c631aa8bef5081e3a59bafb26e8d1",
          "legacy_backend_source":"1cabbf183cd1ebbb76ed838283126797fcd7e3e3c295d3572393e71485a1efbd"}


def finite(v):
    return type(v) in (int,float) and math.isfinite(v)


def packed(values):
    if any(not finite(v) for v in values):
        raise ValueError("nonfinite/nonnumeric array")
    return struct.pack("<" + "f"*len(values), *values)


def inputs(rows,cols):
    size = rows*cols
    return ([((i*37+17)%1009)/128-4 for i in range(size)],
            [((i*13+7)%127)/64-1 for i in range(size)],
            [(i%17)/32-.25 for i in range(size)])


def oracle(rows,cols,count):
    z,g,r = inputs(rows,cols)
    c = math.sqrt(2/math.pi)
    def derivative(x):
        if abs(x) >= 10:
            return float(x > 0)
        t = math.tanh(c*(x+.044715*x*x*x))
        return .5*(1+t)+.5*x*(1-t*t)*c*(1+3*.044715*x*x)
    out = [derivative(x)*seed for x,seed in zip(z,g)]
    if count == 3:
        out += [v+base for v,base in zip(out,r)] + [math.fsum(out[j::cols]) for j in range(cols)]
    return out


def close(output,reference,rows,cols):
    if len(output) != len(reference):
        raise ValueError("output length")
    absolute,scaled = 0.,0.
    for i,(a,b) in enumerate(zip(output,reference)):
        if not finite(a) or not finite(b):
            raise ValueError("nonfinite output")
        error = abs(a-b)
        ratio = error/(ATOL*(rows if i >= 2*rows*cols else 1)+RTOL*abs(b))
        if ratio > 1:
            raise ValueError("GELU oracle mismatch")
        absolute,scaled = max(absolute,error),max(scaled,ratio)
    return absolute,scaled


def key(case):
    k = tuple(case[f] for f in ("rows","cols","count"))
    if any(type(v) is not int for v in k) or k not in KEYS:
        raise ValueError("unknown condition")
    return k


def names(count):
    return ("legacy_full","fused_selected","plain") if count == 1 else ("legacy_full","fused_shared","fused_batch")


def routes(count):
    return tuple(f+"_"+n for f in ("native","browser") for n in names(count))+("cpu","mps")


def order(block,rows,cols,count,torch=False):
    if torch:
        return [0,1] if (block+rows+cols+count)%2 == 0 else [1,0]
    start = (block+rows+cols+count)%3
    result = [(start+i)%3 for i in range(3)]
    return result[::-1] if (block//3)%2 else result


def intervals(case,allowed,repeats=False):
    r,c,n = key(case)
    if case["order_scheme"] != "balanced-cycle-v1":
        raise ValueError("route ordering scheme")
    expected = {(round,b,burst,d) for round in (range(3) if repeats else [0])
                for b in range(9) for burst in (1,4) for d in allowed}
    seen = {}
    for item in case["intervals"]:
        rnd,b,burst,d = item.get("round",0),item["block"],item["burst"],item["route"]
        ident = rnd,b,burst,d
        if any(type(v) is not int for v in (rnd,b,burst)) or ident not in expected or ident in seen:
            raise ValueError("interval identity")
        if not finite(item["elapsed_ms"]) or item["elapsed_ms"] <= 0:
            raise ValueError("duration")
        if any(not finite(item[k]) or item[k] < 0 for k in ("max_abs_error","max_scaled_error")) or item["max_scaled_error"] > 1:
            raise ValueError("interval numerical gate")
        if item["order"] != order(b+3,r,c,n,d in ("cpu","mps")) or any(type(i) is not int for i in item["order"]):
            raise ValueError("position schedule")
        seen[ident] = item["elapsed_ms"]
    if seen.keys() != expected:
        raise ValueError("incomplete intervals")
    return seen


def admit(report,family):
    torch = family == "torch"
    if family not in ("native","browser","torch") or report["schema"] != (TORCH if torch else SCHEMA):
        raise ValueError("runtime schema")
    if report["status"] != "passed" or report["warmup"] != 3 or report["blocks"] != 9 or report["bursts"] != [1,4]:
        raise ValueError("protocol identity")
    if torch:
        if (report["devices"] != ["cpu","mps"] or report["intra_op_threads"] != 4
                or report["inter_op_threads"] != 1 or report["compiled"] is not False
                or report["observation"] != "one_packed_owning_cpu_copy"
                or report["operation"] != "aten.gelu_backward.grad_input:tanh"):
            raise ValueError("Torch control")
    else:
        if not report["adapter"] or "device_type: Cpu" in report["adapter"]:
            raise ValueError("GPU identity")
        for field,digest in LEGACY.items():
            if hashlib.sha256(report[field].encode()).hexdigest() != digest:
                raise ValueError("legacy source drift")
        seen_domains = set()
        for case in report["domains"]["cases"]:
            k = tuple(case[f] for f in ("rows","cols","count","padding"))
            if any(type(v) is not int for v in k) or k not in DOMAIN_KEYS or k in seen_domains:
                raise ValueError("domain coverage")
            seen_domains.add(k)
            count = 2 if k[2] == 1 and k[3] else 3
            if len(case["errors"]) != count or any(len(v)!=2 or any(not finite(x) or x<0 for x in v) or v[1]>1 for v in case["errors"]):
                raise ValueError("domain numerical gate")
        if seen_domains != DOMAIN_KEYS:
            raise ValueError("incomplete domains")
        diagnostic = report["domains"]["legacy_backend_syntax_repaired_only"]
        if len(diagnostic["input"]) != 15 or len(diagnostic["output_strings"]) != 15:
            raise ValueError("legacy diagnostic")
        if diagnostic["nonfinite_count"] != sum(not math.isfinite(float(v)) for v in diagnostic["output_strings"]):
            raise ValueError("legacy diagnostic count")
    if family == "browser":
        stream = report["gelu_state_artifacts"]
        if (report["page_errors"] or report["console_messages"] or not report["asset_sha256"]
                or not report["browser_version"] or report["browser_adapter_probe"]["is_fallback_adapter"] is not False
                or stream["rows"] != len(KEYS) or type(stream["bytes"]) is not int or stream["bytes"] <= 0
                or not stream["path"].endswith(".cases.jsonl") or "/" in stream["path"]
                or len(stream["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in stream["sha256"])):
            raise ValueError("browser completion")
    result = {}
    for case in report["cases"]:
        k = key(case)
        if k in result:
            raise ValueError("duplicate condition")
        r,c,n = k
        for field,want in zip(("input","seed","residual"),inputs(r,c)):
            if packed(case[field]) != packed(want):
                raise ValueError("input bytes")
        ref = oracle(r,c,n)
        close(case["reference"],ref,r,c)
        allowed = ("cpu","mps") if torch else names(n)
        if len(case["last_outputs"]) != len(allowed):
            raise ValueError("output route count")
        for output in case["last_outputs"]:
            close(output,ref,r,c)
        intervals(case,allowed)
        result[k] = case
    if result.keys() != KEYS:
        raise ValueError("incomplete grid")
    return result


def summarize(case):
    count = case["count"]
    idx = intervals(case,routes(count),True)
    result = []
    for burst in (1,4):
        medians = {d:statistics.median(v for (_,_,n,route),v in idx.items() if n==burst and route==d) for d in routes(count)}
        ratios = {f:{label:statistics.median(idx[r,b,burst,f+"_"+names(count)[a]]/idx[r,b,burst,f+"_"+names(count)[z]]
                    for r in range(3) for b in range(9))
                    for label,a,z in [("combined",0,2),("first_step",0,1),("second_step",1,2)]} for f in ("native","browser")}
        result.append(dict(burst=burst,median_interval_ms=medians,paired_ratios=ratios))
    return result


def describe(cases):
    result = {}
    for f in ("native","browser"):
        result[f] = {}
        for n in (1,3):
            result[f][str(n)] = {}
            for factor in ("combined","first_step","second_step"):
                vals = [s["paired_ratios"][f][factor] for c in cases if c["count"]==n for s in c["summary"]]
                result[f][str(n)][factor] = dict(geomean=math.exp(statistics.mean(map(math.log,vals))),
                    min=min(vals),max=max(vals),cells_over_1=sum(v>1 for v in vals),cells=len(vals))
    return result


def analyze(native,browser,torch):
    if any(len(reports)!=3 for reports in (native,browser,torch)):
        raise ValueError("three complete rounds required")
    admitted = [[admit(report,f) for report in reports] for f,reports in zip(("native","browser","torch"),(native,browser,torch))]
    cases = []
    for k in sorted(KEYS):
        r,c,n = k
        reference = oracle(r,c,n)
        record = dict(rows=r,cols=c,count=n,order_scheme="balanced-cycle-v1",intervals=[],max_abs_errors={},max_scaled_errors={},
            input_sha256=[hashlib.sha256(packed(v)).hexdigest() for v in inputs(r,c)],
            oracle_sha256=hashlib.sha256(struct.pack("<"+"d"*len(reference),*reference)).hexdigest())
        for f,rounds in zip(("native","browser","torch"),admitted):
            route_names = ("cpu","mps") if f == "torch" else tuple(f+"_"+v for v in names(n))
            for rnd,report in enumerate(rounds):
                case = report[k]
                for route,out in zip(route_names,case["last_outputs"]):
                    absolute,scaled = close(out,reference,r,c)
                    for field,v in [("max_abs_errors",absolute),("max_scaled_errors",scaled)]:
                        record[field][route] = max(record[field].get(route,0),v)
                record["intervals"].extend({**item,"round":rnd,"route":item["route"] if f=="torch" else f+"_"+item["route"]} for item in case["intervals"])
        record["summary"] = summarize(record)
        cases.append(record)
    return dict(schema=SUMMARY,status="passed",rounds=3,tolerance=dict(atol=ATOL,rtol=RTOL,bias_absolute_scale="rows"),
        cases=cases,descriptive_summary=describe(cases),
        native_metadata=[{k:r[k] for k in ("adapter","domains")} for r in native],
        browser_metadata=[{k:r[k] for k in ("adapter","domains","asset_sha256","browser_version","browser_adapter_probe","gelu_state_artifacts")} for r in browser],
        torch_metadata=[{k:r[k] for k in ("torch_version","devices","operation","observation","intra_op_threads","inter_op_threads")} for r in torch],
        legacy_source_sha256=LEGACY,
        boundary="Prepared f32 tanh-GELU backward; count=1 derivative only, count=3 gradient/residual+bias. Count=1 compares old fused/three reads, identical old fused/one read, and canonical plain/one read. Count=3 compares old/shared fused with separate reads and shared fused with batched read. Same Rust snapshot decoder, not byte-identical old native read_buffer. Encoding/binds, per-op submissions, per-op GPU residual reset and terminal owning CPU completion timed; setup/checks/JSON excluded. Torch ATen tanh backward with preallocated packed outputs and one owning CPU copy; no torch.compile, full training, high-level Tensor multiplier or universal PyTorch superiority claim.")


def validate_summary(result):
    if result["schema"] != SUMMARY or result["status"] != "passed" or result["rounds"] != 3 or result["legacy_source_sha256"] != LEGACY:
        raise ValueError("summary identity")
    if result["tolerance"] != dict(atol=ATOL,rtol=RTOL,bias_absolute_scale="rows"):
        raise ValueError("summary tolerance")
    seen = set()
    for case in result["cases"]:
        k = key(case)
        if k in seen:
            raise ValueError("duplicate condition")
        seen.add(k)
        if len(case["input_sha256"]) != 3:
            raise ValueError("input hash count")
        for h in [*case["input_sha256"],case["oracle_sha256"]]:
            if len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("content hash")
        for field in ("max_abs_errors","max_scaled_errors"):
            if set(case[field])!=set(routes(k[2])) or any(not finite(v) or v<0 for v in case[field].values()):
                raise ValueError("summary numerical gate")
        if max(case["max_scaled_errors"].values())>1 or case["summary"]!=summarize(case):
            raise ValueError("invalid summary")
    if seen!=KEYS or result["descriptive_summary"]!=describe(result["cases"]):
        raise ValueError("summary aggregation")
