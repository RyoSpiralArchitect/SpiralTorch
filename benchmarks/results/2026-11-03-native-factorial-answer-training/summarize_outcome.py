"""Describe the frozen archive, including empty emitted lines. No model, weights or tokenizer."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify as v


def summarize(archive):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        v.open_archive(archive,root)
        verified=v.verify(root)
        outcome=v.read(root/"result.json")
        rows={}
        for point in v.POINTS:
            loc=root/point
            text=v.r.lines(loc/"lines.jsonl")
            gen=v.r.lines(loc/"generated.jsonl")
            reference=[x["generated_ids"] for x in gen if x["arm"]=="affine"]
            endpoint=outcome["endpoints"][point]
            arms={}
            for arm in v.d.ARMS:
                lines=[x for x in text if x["arm"]==arm]
                streams=[x["generated_ids"] for x in gen if x["arm"]==arm]
                counts=dict(exact=0,empty_first_line=0,other_terminated=0,unterminated=0)
                for line in lines:
                    first, lf, _ = line["generated_text"].partition("\n")
                    key=("unterminated" if not lf else "empty_first_line" if not first else
                         "exact" if line["exact"] else "other_terminated")
                    counts[key]+=1
                score=endpoint["arms"][arm]
                v.check(sum(counts.values())==v.d.EVAL and counts["exact"]==score["greedy_line_exact"] and
                        v.d.EVAL-counts["unterminated"]==score["greedy_terminated"],"line count disagreement")
                arms[arm]=dict(answer_nll=score["mean_answer_nll"],teacher_token_hits=score["teacher_token_hits"],
                               teacher_all_correct=score["teacher_all_correct"],lines=counts,
                               same_generated_ids_as_affine=sum(x==y for x,y in zip(streams,reference,strict=True)))
            rows[point]=dict(arms=arms,primary_nll=endpoint["nll_contrasts"][v.d.PRIMARY],
                             primary_lines=endpoint["line_contrasts"][v.d.PRIMARY],
                             nll_interaction=endpoint["factorial_nll"]["interaction"])
        return dict(archive_sha256=v.sha(archive),decision=verified["decision"],rows=rows,
                    model_executions=0,content_id_recomputed=False)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    print(json.dumps(summarize(ap.parse_args().archive),indent=2,sort_keys=True))
