"""Independent contextual study validation; Rust alone prepares the request."""
import copy
import importlib.util
import json
from pathlib import Path


def study():
    spec = importlib.util.spec_from_file_location(
        "byte_corpus_study", Path(__file__).with_name("byte_corpus_study.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contextual(case):
    return "query_projection" in case.get("mixture", {})


def validate_request(request):
    s = study()
    meta = request.get("query_controls")
    require(type(meta) is dict and set(meta) == {"source_request_sha256"}
            and type(meta["source_request_sha256"]) is str
            and len(meta["source_request_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in meta["source_request_sha256"]),
            "contextual source identity")
    cases = request["cases"]
    require(25 <= len(cases) <= 75 and len({c["name"] for c in cases}) == len(cases),
            "contextual case count/names")
    base = copy.deepcopy(request)
    base["schema"] = s.REQUEST_V7
    del base["query_controls"]
    base["cases"] = [c for c in base["cases"] if not contextual(c)]
    s.request_version(base)
    bits = s.load_reference("byte_corpus_mixture").parameter_bits
    sources = [c for c in base["cases"] if c.get("mixture", {}).get("router_update") == "train"]
    require(sum(contextual(c) for c in cases) == 2 * len(sources), "contextual coverage")
    for source in sources:
        for update in ("train", "frozen"):
            found = [c for c in cases if contextual(c) and c["seed"] == source["seed"]
                     and c["mixture"].get("metrics") == source["mixture"]["metrics"]
                     and type(c["mixture"]["query_projection"]) is dict
                     and c["mixture"]["query_projection"].get("weight_update") == update]
            require(len(found) == 1, "missing/repeated contextual arm")
            c = found[0]
            m = c["mixture"]
            require(set(m) == {"metrics", "scales", "router_update", "query_projection"}
                    and m["query_projection"] == {"weight_update": update, "bias": "frozen_zero"},
                    "contextual update policy")
            stripped = copy.deepcopy(c)
            del stripped["mixture"]["query_projection"]
            stripped["parameters"] = [p for p in stripped["parameters"]
                                      if not p["name"].startswith("geometry.query_projection.")]
            stripped["name"] = source["name"]
            require(stripped == source and bits(stripped["parameters"]) == bits(source["parameters"]),
                    "contextual arm changed shared source")
            expected = copy.deepcopy(source["parameters"])
            output = request["config"]["heads"] * len(m["metrics"])
            cols = request["config"]["geometry_cols"]
            at = next(i for i, p in enumerate(expected) if p["name"] == "geometry.mixture_logits.0") + 1
            expected[at:at] = [
                dict(name="geometry.query_projection.0.weight", shape=[cols, output], values=[0.]*(cols*output)),
                dict(name="geometry.query_projection.0.bias", shape=[output], values=[0.]*output)]
            require(bits(c["parameters"]) == bits(expected), "contextual zero initialization/order")


def contrasts(summaries):
    rows = []
    for shared in summaries:
        m = shared.get("mixture", {})
        if m.get("router_update") != "train" or contextual(shared):
            continue
        twins = [next(c for c in summaries if c["seed"] == shared["seed"] and contextual(c)
                      and c["mixture"]["metrics"] == m["metrics"]
                      and c["mixture"]["query_projection"]["weight_update"] == u)
                 for u in ("train", "frozen")]
        def differences(key):
            return dict(contextual_minus_fixed_query_bpb=twins[0][key]-twins[1][key],
                        contextual_minus_shared_bpb=twins[0][key]-shared[key],
                        fixed_query_minus_shared_bpb=twins[1][key]-shared[key])
        actual, reference = differences("final_heldout_bpb"), differences("reference_final_heldout_bpb")
        rows.append(dict(seed=shared["seed"], metrics=m["metrics"], measured=actual, reference=reference,
                         absolute_torch_discrepancy={k:abs(v-reference[k]) for k,v in actual.items()},
                         same_sign_as_torch={k:((v>0)-(v<0)) == ((reference[k]>0)-(reference[k]<0))
                                             for k,v in actual.items()}))
    return rows


def verify_preparation(source_raw, packet):
    s = study()
    source = s.decode_json(source_raw)
    require(s.request_version(source) == 7, "preparation source must be v7")
    require(set(packet) == {"schema", "request_json", "report_json"}
            and packet["schema"] == "spiraltorch.byte_corpus.prepared.v1", "preparation envelope")
    raw = packet["request_json"].encode()
    request = s.decode_json(raw)
    require(s.request_version(request) == 8, "wrong contextual version")
    report = s.decode_json(packet["report_json"])
    require(report["schema"] == "spiraltorch.byte_corpus.query_preparation.v1"
            and report["source_request_sha256"] == s.digest(source_raw)
            and request["query_controls"]["source_request_sha256"] == s.digest(source_raw)
            and report["request_sha256"] == s.digest(raw)
            and report["no_fitting"] is True and report["no_updates_consumed"] is True,
            "preparation identity or claim mismatch")
    inherited = copy.deepcopy(request)
    inherited["schema"] = s.REQUEST_V7
    del inherited["query_controls"]
    inherited["cases"] = [c for c in inherited["cases"] if not contextual(c)]
    # Sorting keys tolerates serializer object order, never float/signed-zero changes.
    require(json.dumps(inherited, sort_keys=True, allow_nan=False)
            == json.dumps(source, sort_keys=True, allow_nan=False), "source request changed")
    return raw, dict(schema="spiraltorch.byte_corpus.query_preparation_check.v1",
                     source_request_sha256=s.digest(source_raw), request_sha256=s.digest(raw),
                     passed=True, source_cases=len(source["cases"]), prepared_cases=len(request["cases"]),
                     no_fitting=True, no_updates_consumed=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("packet", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    s = study()
    raw, result = verify_preparation(args.source.read_bytes(), s.decode_json(args.packet.read_bytes()))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "request.json").open("xb") as output:
        output.write(raw)
    s.write_new(args.output / "verification.json", result)


if __name__ == "__main__":
    main()
