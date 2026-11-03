"""Join generated text to references only after inference. Pure read-back, no model execution."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as d  # noqa: E402
from native_answer_generation import score_line  # noqa: E402


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def decode_endpoint(directory, rows, decode):
    gens=lines(directory/"generated.jsonl")
    d.check(len(gens)==4*len(rows),"incomplete generations")
    with (directory/"lines.jsonl").open("x") as out:
        for j,arm in enumerate(d.ARMS):
            for i,r in enumerate(rows):
                g=gens[j*len(rows)+i]
                d.check((g["arm"],g["index"],g["key"])==(arm,i,r["key"]),"generation/reference join")
                text=decode(g["generated_ids"])
                flags=score_line(text,r["answer"])
                out.write(json.dumps(dict(arm=arm,index=i,key=r["key"],generated_text=text,**flags),
                                     separators=(",",":"),allow_nan=False)+"\n")


def describe(directory, rows):
    receipt=d.old.read(directory/"scores.json")
    answers=receipt["scores"]["answers"]
    ev=receipt["scores"]["comparison"]["evaluations"]
    gens=lines(directory/"generated.jsonl")
    decoded=lines(directory/"lines.jsonl")
    n=len(rows)
    d.check(len(answers)==n and len(ev)==4 and len(gens)==len(decoded)==4*n,"endpoint cardinality")
    macro,hit_counts,line_counts,arms=[],[],[],{}
    bits=[]
    for j,arm in enumerate(d.ARMS):
        h=tokens=allcorrect=0
        nlls=[]
        scores=[]
        for i,r in enumerate(rows):
            a=answers[i]
            k=j*n+i
            g,line=gens[k],decoded[k]
            d.check((a["document"],g["arm"],g["index"],g["key"])==(i,arm,i,r["key"])
                    and (line["arm"],line["index"],line["key"])==(arm,i,r["key"])
                    and g["owners"]==[ev[j]["lora_artifact_id"],ev[j]["geometry_artifact_id"]],"answer or owner mismatch")
            nt=a["end"]-a["start"]
            score=a["scores"][j]
            d.check(nt>0 and 0<=score["top1_correct"]<=nt and math.isfinite(score["nll"]),"invalid answer scores")
            flags=score_line(line["generated_text"],r["answer"])
            d.check(all(line[k] is v for k,v in flags.items()),"incorrect exact line adjudication")
            tokens+=nt
            h+=score["top1_correct"]
            allcorrect+=int(score["top1_correct"]==nt)
            nlls.append(score["nll"])
            scores.append(flags)
        hits=[s["exact"] for s in scores]
        bits.append(hits)
        macro.append(math.fsum(nlls)/n)
        hit_counts.append(allcorrect)
        line_counts.append(sum(hits))
        arms[arm]=dict(mean_answer_nll=macro[-1],content_tokens=tokens,teacher_token_hits=h,
                       teacher_all_correct=allcorrect,greedy_line_exact=sum(hits),
                       greedy_terminated=sum(s["terminated"] for s in scores),greedy_literal_prefix=sum(s["prefix_exact"] for s in scores))
    contrasts=receipt["contrasts"]
    d.check(set(contrasts)==set(d.PAIRS),"contrast roster")
    for name,(a,b) in d.PAIRS.items():
        c=contrasts[name]
        dis=[r["scores"][b]["nll"]-r["scores"][a]["nll"] for r in answers]
        d.check((c["reference"],c["candidate"],c["answer_count"],c["target_tokens"])==(a,b,n,arms[d.ARMS[a]]["content_tokens"])
                and math.isclose(math.fsum(dis)/n,c["answer_weighted_delta_nll"],abs_tol=2e-12,rel_tol=2e-12)
                and math.isclose(math.fsum(v*(r["end"]-r["start"]) for v,r in zip(dis,answers,strict=True))/c["target_tokens"],
                                 c["token_weighted_delta_nll"],abs_tol=2e-12,rel_tol=2e-12)
                and (c["improved"],c["tied"],c["regressed"])==(sum(x<0 for x in dis),sum(x==0 for x in dis),sum(x>0 for x in dis))
                and (c["reference_all_correct_answers"],c["candidate_all_correct_answers"])==(hit_counts[a],hit_counts[b])
                and (c["reference_top1_correct"],c["candidate_top1_correct"])==(arms[d.ARMS[a]]["teacher_token_hits"],arms[d.ARMS[b]]["teacher_token_hits"]),
                "inconsistent paired NLL or hit counts")
    return dict(arms=arms,nll_contrasts=contrasts,line_contrasts={k:d.paired_lines(bits[a],bits[b]) for k,(a,b) in d.PAIRS.items()},
                factorial_nll=d.factorial_effects(*macro),factorial_lines=d.factorial_effects(*line_counts))
