"""Independent mixture contract and Torch reference; no production fitting."""
import copy
import itertools
import math
import struct
import importlib.util
from pathlib import Path

P, E, F = "poincare_squared.v1", "euclidean_chord_squared.v1", "categorical_fisher_rao_squared.v1"
COMBINATIONS = list(itertools.combinations((P,E,F),2)) + [(P,E,F)]


def study():
    spec = importlib.util.spec_from_file_location("byte_corpus_study", Path(__file__).with_name("byte_corpus_study.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parameter_bits(parameters):
    result = []
    for p in parameters:
        require(type(p["shape"]) is list and p["shape"] and all(type(d) is int and d > 0 for d in p["shape"])
                and type(p["values"]) is list and math.prod(p["shape"]) == len(p["values"])
                and all(type(v) in (int,float) and math.isfinite(v) for v in p["values"]), "invalid mixture parameter")
        result.append((p["name"],p["shape"],[struct.pack("<f",v) for v in p["values"]]))
    return result


def unit_scales(sources, metrics):
    def gains(c):
        return next(p["values"] for p in c["parameters"] if p["name"] == "geometry.raw_gain.0")
    softplus = lambda x: max(x,0.) + math.log1p(math.exp(-abs(x)))
    f32 = study().f32
    base = [softplus(f32(v)) for v in gains(sources[P])]
    require(all(v > 0 and math.isfinite(v) for v in base), "invalid positive gain source")
    return [f32(softplus(f32(gains(sources[m])[h])) / base[h]) for h in range(2) for m in metrics]


def validate_request(request):
    metadata = request.get("mixture_controls")
    require(type(metadata) is dict and set(metadata) == {"source_request_sha256"}
            and type(metadata["source_request_sha256"]) is str and len(metadata["source_request_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in metadata["source_request_sha256"]), "mixture source identity")
    cfg = request["config"]
    require(type(cfg["heads"]) is int and cfg["heads"] == 2 and type(cfg["blocks"]) is list
            and len(cfg["blocks"]) == 1 and type(cfg["blocks"][0]) is bool, "one-block/two-head mixture study")
    require("headwise_controls" not in request and all("head_pair_metrics" not in c for c in request["cases"]), "foreign head routing")
    cases = request["cases"]
    require(17 <= len(cases) <= 51 and len({c["name"] for c in cases}) == len(cases), "mixture case count/names")
    base = copy.deepcopy(request)
    base["schema"] = "spiraltorch.byte_corpus.request.v5"
    del base["mixture_controls"]
    base["cases"] = [c for c in base["cases"] if "mixture" not in c]
    study().request_version(base)
    seeds = {c["seed"] for c in base["cases"]}
    require(all(type(c["seed"]) is int and c["seed"] in seeds for c in cases), "unknown mixture seed")
    for seed in seeds:
        group = [c for c in cases if c["seed"] == seed]
        require(len(group) == 17, "incomplete mixture seed")
        sources = {m:next(c for c in base["cases"] if c["seed"] == seed and c.get("pair_metric") == m
                   and c["geometry_update"] == "train" and c["bias_initialization"] == ("original" if m == P else "matched_poincare_rms")) for m in (P,E,F)}
        for c in group:
            if "mixture" not in c:
                continue
            m = c["mixture"]
            require(type(m) is dict and set(m) == {"metrics","scales","router_update"}
                    and type(m["metrics"]) is list and tuple(m["metrics"]) in COMBINATIONS
                    and m["router_update"] in ("train","frozen"), "invalid mixture fields")
        for metrics in COMBINATIONS:
            pair = []
            expected_scales = unit_scales(sources,metrics)
            for update in ("train","frozen"):
                found = [c for c in group if c.get("mixture",{}).get("metrics") == list(metrics)
                         and c["mixture"]["router_update"] == update]
                require(len(found) == 1, "missing/repeated mixture router arm")
                c = found[0]
                require(c["geometry"] is True and c["geometry_update"] == "train" and "pair_metric" not in c
                        and c["bias_initialization"] == "matched_poincare_rms", "mixed policy identity")
                values = c["mixture"]["scales"]
                require(type(values) is list and len(values) == 1 and type(values[0]) is list and len(values[0]) == len(expected_scales)
                        and all(type(a) in (int,float) and math.isfinite(a) and a>0 and abs(a/e-1.) <= 1e-6
                                for a,e in zip(values[0],expected_scales)), "component units changed")
                parameter_bits(c["parameters"])
                expected = copy.deepcopy(sources[P]["parameters"])
                expected[6]["values"] = c["parameters"][6]["values"]
                expected.insert(7,dict(name="geometry.mixture_logits.0",shape=[2,len(metrics)],values=[0.]* (2*len(metrics))))
                require(parameter_bits(c["parameters"]) == parameter_bits(expected), "mixture changed initial values beyond gain/router slots")
                pair.append(c)
            require(parameter_bits(pair[0]["parameters"]) == parameter_bits(pair[1]["parameters"])
                    and [struct.pack("<f",v) for v in pair[0]["mixture"]["scales"][0]]
                    == [struct.pack("<f",v) for v in pair[1]["mixture"]["scales"][0]], "router twins must share every initial bit")


def pair_bias(coordinates, gains, logits, mixture, block, curvature, poincare, fisher, projection=None):
    import torch
    import torch.nn.functional as functional
    parts = []
    for metric in mixture["metrics"]:
        if metric == P:
            scores = poincare(coordinates,gains,curvature)
        elif metric == E:
            square = (coordinates[:,:,None,:] - coordinates[:,None,:,:]).square().sum(-1)
            scores = (-functional.softplus(gains)[None,:,None,None] * (4*square[:,None])).tril()
        elif metric == F:
            scores = fisher(coordinates,gains)
        else:
            raise ValueError("unknown component metric")
        parts.append(scores)
    scales = torch.tensor(mixture["scales"][block],device="cpu",dtype=torch.float32).reshape(logits.shape)
    if projection is not None:
        require("query_projection" in mixture and len(projection) == 2, "query projection coverage")
        h, k = logits.shape
        query = coordinates @ projection[0] + projection[1]
        query = (query + logits.reshape(h*k)).reshape(*coordinates.shape[:2], h, k)
        weights = query.permute(0, 2, 1, 3).softmax(-1)[:, :, :, None, :]
    else:
        require("query_projection" not in mixture, "missing query projection")
        weights = logits.softmax(-1)[None,:,None,None,:]
    return (torch.stack(parts,dim=-1) * scales[None,:,None,None,:] * weights).sum(-1)


def router_deltas(initial, actual, expected, shape):
    require(len(shape)==2 and math.prod(shape)==len(initial)==len(actual)==len(expected), "router delta coverage")
    rows = []
    for h in range(shape[0]):
        for k in range(shape[1]):
            i=h*shape[1]+k
            norm=abs(expected[i]-initial[i])
            error=abs(actual[i]-expected[i])
            require(norm>1e-8 and actual[i]!=initial[i] and math.isfinite(error) and error/norm<=.002,
                    "unqualified individual mixture router delta")
            rows.append(dict(head=h,component=k,reference_delta_abs=norm,relative_error=error/norm))
    return rows


def contrasts(summaries):
    rows=[]
    for seed in sorted({c["seed"] for c in summaries}):
        group=[c for c in summaries if c["seed"]==seed]
        for metrics in COMBINATIONS:
            pair=[next(c for c in group if c.get("mixture",{}).get("metrics")==list(metrics)
                       and c["mixture"]["router_update"]==u) for u in ("train","frozen")]
            def differences(key):
                result={"learned_router_minus_fixed_router_bpb":pair[0][key]-pair[1][key]}
                for metric,label in ((P,"poincare"),(E,"calibrated_flat"),(F,"fisher")):
                    baseline=next(c for c in group if c.get("pair_metric")==metric and c["geometry_update"]=="train"
                                  and c["bias_initialization"]==("original" if metric==P else "matched_poincare_rms"))
                    for c,mode in zip(pair,("learned_router","fixed_router")):
                        result[f"{mode}_minus_{label}_bpb"]=c[key]-baseline[key]
                return result
            measured,reference=differences("final_heldout_bpb"),differences("reference_final_heldout_bpb")
            rows.append(dict(seed=seed,metrics=list(metrics),measured=measured,reference=reference,
                absolute_torch_discrepancy={k:abs(v-reference[k]) for k,v in measured.items()},
                same_sign_as_torch={k:((v>0)-(v<0))==((reference[k]>0)-(reference[k]<0)) for k,v in measured.items()}))
    return rows
