#!/usr/bin/env python3
"""Bind a Rust mixture preparation to unchanged v5 and training-only RMS."""
import argparse
import copy
import importlib.util
import math
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "verify_byte_corpus_headwise_preparation",
    Path(__file__).with_name("verify_byte_corpus_headwise_preparation.py"))
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
study, resume, require, verify_topology = (
    HELPERS.study, HELPERS.resume, HELPERS.require, HELPERS.verify_topology)


def verify(source_raw, packet):
    require(type(packet) is dict and set(packet)=={"schema","request_json","report_json"}
            and packet["schema"]=="spiraltorch.byte_corpus.prepared.v1", "preparation envelope")
    source=study.decode_json(source_raw)
    raw=packet["request_json"].encode()
    request=study.decode_json(raw)
    report=study.decode_json(packet["report_json"])
    require(study.request_version(source)==5 and study.request_version(request)==7, "preparation versions")
    require(request["mixture_controls"]["source_request_sha256"]==study.digest(source_raw)
            and set(request)==set(source)|{"mixture_controls"}, "source contract")
    for key in source:
        if key not in ("schema","cases"):
            require(resume.exact(request[key],source[key]), "source recipe changed")
    def initial(c):
        c=copy.deepcopy(c)
        c["parameters"]=[(p["name"],p["shape"],[resume.f32(v) for v in p["values"]]) for p in c["parameters"]]
        return c
    require(len(request["cases"])==len(source["cases"])*17//9 and
            [initial(c) for c in request["cases"][:len(source["cases"])]]==[initial(c) for c in source["cases"]], "inherited cases changed")
    require(report["schema"]=="spiraltorch.byte_corpus.mixture_preparation.v1"
            and report["source_request_sha256"]==study.digest(source_raw)
            and report["request_sha256"]==study.digest(raw)
            and report.get("no_updates_consumed") is True and type(report.get("adapter")) is str and report["adapter"], "preparation identity/state")
    require(resume.exact(report["train_batch_indices"],request["bias_calibration"]["train_batch_indices"])
            and type(report["relative_tolerance"]) in (int,float) and report["relative_tolerance"]==1e-5, "frozen RMS recipe")
    expected=[c for seed in sorted({c["seed"] for c in request["cases"]}) for c in request["cases"]
              if c["seed"]==seed and "mixture" in c and c["mixture"]["router_update"]=="train"]
    require(type(report["cases"]) is list and len(report["cases"])==len(expected), "mixture coverage")
    cfg=request["config"]
    count=len(report["train_batch_indices"])*cfg["batch"]*cfg["steps"]*(cfg["steps"]+1)//2
    targets,summaries={},[]
    for row,case in zip(report["cases"],expected):
        require(all(resume.exact(row[k],case[k]) for k in ("name","seed","mixture")) and row.get("no_updates_consumed") is True, "mixture record identity")
        source_case=next(c for c in source["cases"] if c["seed"]==case["seed"] and c.get("pair_metric")==study.POINCARE and c["geometry_update"]=="train")
        before_case=copy.deepcopy(case)
        before_case["parameters"][6]["values"]=source_case["parameters"][6]["values"]
        for key,expected_case in (("before_checkpoint_json",before_case),("fitted_checkpoint_json",case)):
            model=study.decode_json(row[key])
            parameters=resume.stored_parameters(model,0,mixture=case["mixture"])
            verify_topology(cfg,model)
            require(len(parameters)==len(expected_case["parameters"]), "checkpoint tensor coverage")
            for p,e in zip(parameters,expected_case["parameters"]):
                require(resume.exact(p["shape"],e["shape"]) and list(map(resume.f32,p["values"]))==list(map(resume.f32,e["values"])), "checkpoint initial bits")
        require(type(row["raw_gains"]) is list and len(row["raw_gains"])==1 and
                list(map(resume.f32,row["raw_gains"][0]))==list(map(resume.f32,case["parameters"][6]["values"])), "fitted gain binding")
        require(type(row["valid_pairs_per_head"]) is int and row["valid_pairs_per_head"]==count, "RMS pair coverage")
        for key in ("reference_rms","candidate_rms","fitted_rms","realized_relative_errors"):
            values=row[key]
            require(type(values) is list and len(values)==1 and type(values[0]) is list and len(values[0])==2
                    and all(type(v) in (int,float) and math.isfinite(v) and (v>=0 if key=="realized_relative_errors" else v>0) for v in values[0]), "RMS head coverage")
        require(row["reference_rms"]==targets.setdefault(case["seed"],row["reference_rms"]), "shared Poincare target")
        softplus=lambda x:max(x,0.)+math.log1p(math.exp(-abs(x)))
        for h,(a,b,c,claimed) in enumerate(zip(row["reference_rms"][0],row["candidate_rms"][0],row["fitted_rms"][0],row["realized_relative_errors"][0])):
            error=abs(c/a-1.)
            require(error<=1e-5 and abs(error-claimed)<=1e-14, "RMS arithmetic or gate")
            old=softplus(study.f32(before_case["parameters"][6]["values"][h]))
            new=softplus(study.f32(case["parameters"][6]["values"][h]))
            require(old>0. and abs((new/old)/(a/b)-1.)<=1e-5, "gain transformation differs from RMS scale")
        summaries.append({k:row[k] for k in ("name","seed","mixture","reference_rms","candidate_rms","fitted_rms","realized_relative_errors","valid_pairs_per_head")})
    return dict(schema="spiraltorch.byte_corpus.mixture_preparation_verification.v1",passed=True,
        source_request_sha256=study.digest(source_raw),request_sha256=study.digest(raw),inherited_cases=len(source["cases"]),
        all_initial_parameter_bits_checked=True,gain_only_fitting=True,no_updates_consumed=True,cases=summaries,
        scope="Artifact binding and reported RMS/gain arithmetic; not independent runtime attestation or quality evidence")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","packet","output"): parser.add_argument(name,type=Path)
    args=parser.parse_args()
    packet_raw=args.packet.read_bytes()
    packet=study.decode_json(packet_raw)
    result=verify(args.source.read_bytes(),packet)
    result["packet_sha256"]=study.digest(packet_raw)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,key in (("request.json","request_json"),("preparation.json","report_json")):
        with (args.output/name).open("xb") as output: output.write(packet[key].encode())
    study.write_new(args.output/"verification.json",result)
    print(f"Verified {len(result['cases'])} mixture preparations")


if __name__=="__main__":
    main()
