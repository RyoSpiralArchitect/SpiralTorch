"""Offline read-back. No model import, fitting, tokenization, or decoder execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b):
    return math.isclose(a,b,abs_tol=2e-12,rel_tol=2e-12)


def verify(root, originals):
    def read(name):
        return json.loads((root/name).read_bytes())
    result, manifest, protocol = [read(n+".json") for n in ("result","manifest","protocol")]
    spec=protocol["spec"]
    arms=["lora","frozen_wave_lora","joint_wave_lora"]
    check(sha(root/"manifest.json")==result["manifest_sha256"] and sha(root/"protocol.json")==manifest["protocol_sha256"]
          and sha(Path(__file__).with_name("pilot.py"))==protocol["script_sha256"],"source/manifest/protocol pin")
    required={"attempt.json","initial-owners.json","steps.jsonl","scores-000.json","scores-096.json",
              "cohort-032.json","cohort-064.json","cohort-096.json"}
    check(set(result["receipt_sha256"])==required and all(sha(root/n)==h for n,h in result["receipt_sha256"].items()),
          "raw receipt inventory/pin")
    check(protocol["installed"]==result["installed"] and result["source_commit"]==protocol["source_commit"]
          and spec["arms"]==result["arms"]==arms and spec["score_updates"]==[0,96],"source, owners or endpoints")
    check(manifest["distinct_records"]==144 and [manifest["splits"][k]["documents"] for k in ("train","eval")]==[96,48],"split sizes")
    check(set(manifest["files"])=={"train-text.json","eval-text.json","train.json","eval.json"},"data file inventory")
    data={}
    if originals:
        check(all(sha(root/n)==h for n,h in manifest["files"].items()),"data file pin")
        keys=set()
        for k,n in (("train",96),("eval",48)):
            rows=read(k+"-text.json")
            data[k]=read(k+".json")
            check(len(rows)==len(data[k]["labels"])==len(data[k]["token_documents"])==n,"encoded split count")
            for row,ids,labels in zip(rows,data[k]["token_documents"],data[k]["labels"],strict=True):
                values=tuple(sorted(row["values"].items()))
                first=next(i for i,x in enumerate(labels) if x!=-100)
                check(values not in keys and row["answer"]==" "+row["values"][row["field"]]
                      and 0<first<len(ids)<=64 and labels==[-100]*first+ids[first:],"source, mask or record overlap")
                keys.add(values)
            check(sum(x!=-100 for row in data[k]["labels"] for x in row)==manifest["splits"][k]["target_tokens"],"split targets")

    starts=read("initial-owners.json")
    first=[(starts["lora"],g) for g in (None,starts["wave"],starts["wave"])]
    previous=first
    steps=[json.loads(line) for line in (root/"steps.jsonl").read_bytes().splitlines()]
    check(len(steps)==result["updates_per_arm"]==96,"update count")
    targets=0
    for i,reports in enumerate(steps,1):
        expected=reports[0]["target_tokens"]
        check(expected>0 and len(reports)==3,"step roster or targets")
        if originals:
            check(expected==sum(t!=-100 for t in data["train"]["labels"][i-1]),"training denominator")
        for j,r in enumerate(reports):
            check((r["pre_lora_artifact_id"],r["pre_geometry_artifact_id"])==previous[j] and
                  (r["epoch"],r["update"],r["window_start"],r["window_end"],r["target_tokens"])==(0,i,i-1,i,expected)
                  and r["mode"]==("lora","lora","joint")[j] and math.isfinite(r["loss"]),"step lineage/order")
        check(reports[0]["post_geometry_artifact_id"] is None and reports[1]["post_geometry_artifact_id"]==starts["wave"],"frozen owner")
        if i==1:
            check(reports[1]["loss"]==reports[2]["loss"] and reports[1]["post_lora_artifact_id"]==reports[2]["post_lora_artifact_id"],
                  "first update not matched")
        previous=[(r["post_lora_artifact_id"],r["post_geometry_artifact_id"]) for r in reports]
        targets+=expected
        if i in (32,64,96):
            entries=read(f"cohort-{i:03}.json")
            check(len(entries)==3,"missing checkpoint arm")
            for j,e in enumerate(entries):
                check((e["post_lora_id"],e["post_geometry_id"])==previous[j] and e["completed_updates"]==i
                      and e["recorded_data_id"]==starts["data_id"] and e["file"]==f"{arms[j]}-{i:03}.ck"
                      and e["boundary"]==f"{arms[j]}-{i:03}-boundary.json","checkpoint boundary")
                if originals:
                    raw=read(e["file"])
                    check("sha256:"+sha(root/e["file"])==e["checkpoint_id"] and (raw["lora_id"],raw["geometry_id"])==previous[j]
                          and raw["updates"]==i and raw["data_id"]==starts["data_id"],"private checkpoint bytes")
                    check(read(e["boundary"])=={k:v for k,v in e.items() if k not in {"file","boundary"}},"private boundary")
    check(targets==result["training_targets_per_arm"]==manifest["splits"]["train"]["target_tokens"],"total training targets")

    eval_ids=set()
    for i,owners in [(0,first),(96,previous)]:
        receipt=read(f"scores-{i:03}.json")
        scored=receipt["scores"]
        rows=scored["answers"]
        ev=scored["comparison"]["evaluations"]
        check(len(ev)==len(arms) and len(rows)==scored["scored_documents"]==scored["document_count"]==48
              and scored["comparison"]["shared_prefix_layers"]==11,"scoring coverage")
        sizes=[]
        for k,row in enumerate(rows):
            a,b=row["start"],row["end"]
            check(row["document"]==k and 0<a<b<=64 and len(row["scores"])==3,"answer bounds")
            if originals:
                mask=data["eval"]["labels"][k]
                check(len(mask)==b and mask[:a]==[-100]*a and all(x!=-100 for x in mask[a:]),"answer mask mismatch")
            sizes.append(b-a)
            check(all(math.isfinite(s["nll"]) and s["nll"]>=0 and 0<=s["top1_correct"]<=b-a for s in row["scores"]),"answer score")
        n=sum(sizes)
        check(n==manifest["splits"]["eval"]["target_tokens"],"evaluation denominator")
        for j,v in enumerate(ev):
            check((v["lora_artifact_id"],v["geometry_artifact_id"])==owners[j] and v["base_model_identity"]==result["base_id"]
                  and v["target_tokens"]==n and v["window_count"]==48 and v["skipped_windows"]==0 and v["readout_accumulator"]=="f32",
                  "scored snapshot/policy")
            micro=math.fsum(s*r["scores"][j]["nll"] for s,r in zip(sizes,rows,strict=True))/n
            check(math.isclose(v["nll"],micro,rel_tol=1e-6,abs_tol=1e-7),"legacy f32 reduction")
            eval_ids.add((v["data_id"],v["plan_id"]))
        check([v["nll"] for v in ev]==result["start_probe_nll" if i==0 else "final_probe_nll"],"headline corpus NLL")
        for label,a,b in [("joint_minus_frozen",1,2),("joint_minus_lora",0,2),("frozen_minus_lora",0,1)]:
            pairs=[(r["scores"][a],r["scores"][b]) for r in rows]
            deltas=[y["nll"]-x["nll"] for x,y in pairs]
            c=receipt["contrasts"][label]
            check((c["reference"],c["candidate"],c["answer_count"],c["target_tokens"])==(a,b,48,n),"contrast roster")
            check((c["improved"],c["tied"],c["regressed"])==tuple(sum(fn(d) for d in deltas) for fn in
                  (lambda d:d<0,lambda d:d==0,lambda d:d>0)),"per-answer direction")
            check(close(c["answer_weighted_delta_nll"],math.fsum(deltas)/len(rows))
                  and close(c["token_weighted_delta_nll"],math.fsum(s*d for s,d in zip(sizes,deltas,strict=True))/n),"paired reduction")
            for k,owner in enumerate(["reference","candidate"]):
                check(c[owner+"_top1_correct"]==sum(p[k]["top1_correct"] for p in pairs)
                      and c[owner+"_all_correct_answers"]==sum(p[k]["top1_correct"]==s for s,p in zip(sizes,pairs,strict=True)),
                      "top1 counts")
        if i==0:
            check(receipt["contrasts"]["joint_minus_frozen"]["tied"]==48,"nonidentical primary start")
        else:
            check(receipt["contrasts"]["joint_minus_frozen"]==result["primary"],"primary mismatch")
    check(len(eval_ids)==1,"scored different evaluation data")
    c=result["primary"]
    check(result["pilot_positive"]==(c["answer_weighted_delta_nll"]<0 and c["candidate_all_correct_answers"]>=c["reference_all_correct_answers"]),
          "prespecified outcome mismatch")
    attempt=read("attempt.json")
    check(attempt["no_retry"] and attempt["manifest_sha256"]==result["manifest_sha256"] and
          attempt["started_epoch"]+result["wall_seconds"]<=attempt["deadline_epoch"]==protocol["deadline_epoch"] and
          manifest["prep_cpu_seconds"]+result["train_cpu_seconds"]<=3600,"budget or attempt binding")
    return dict(steps_verified=len(steps)*3, paired_endpoints=2, contrasted_pairs=6, evaluation_answers_per_endpoint=48,
                checkpoint_boundaries_verified=9, originals_checked=originals, model_executions=0)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    ap.add_argument("--originals",action="store_true",help="also check all encoded/text files and raw checkpoints")
    args=ap.parse_args()
    print(json.dumps(verify(args.directory,args.originals),sort_keys=True))
