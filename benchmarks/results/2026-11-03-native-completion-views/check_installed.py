"""Print a comparable two-update synthetic signature from one pinned installed CPU wheel."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"tools"))
import byte_release_artifacts as wheels


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def run(which, wheel, pin):
    inventory = wheels.wheel_archive_inventory(wheel)
    identity = wheels.installed_identity(inventory)
    require(identity["wheel_sha256"] == pin, "unexpected wheel")
    banned = {"torch","numpy","transformers","safetensors","peft","accelerate","tokenizers"}
    class Guard:
        def find_spec(self,name,path=None,target=None):
            if name.partition(".")[0] in banned:
                raise AssertionError("unexpected foreign runtime: "+name)
    sys.meta_path.insert(0,Guard())
    import spiraltorch as st
    require(wheels.sha256(Path(st._rs.__file__)) == identity["native_sha256"], "wrong loaded extension")
    owners = runpy.run_path(str(ROOT/"bindings/st-py/tests/test_native_causal_lm_session.py"))["owners"]
    base, model = owners()
    fixture = json.loads((ROOT/"crates/st-nn/tests/fixtures/causal_completion.json").read_bytes())
    spec,expected = fixture["input"],fixture["expected"]
    prompt = spec["prompt_ids"]
    if which == "views":
        view = st.CausalCompletion(prompt,spec["prompt_answer_ids"],spec["training_ids"])
        ids,labels = view.training_document()
        content,mask = view.evaluation_document()
        prompt = view.prompt_ids
    else:
        ids,labels = expected["training_ids"],expected["training_labels"]
        content,mask = ids[:expected["answer_end"]],labels[:expected["answer_end"]]
    session = st.NativeCausalLmTrainingSession(model,[ids],9,mode="joint",labels=[labels],epochs=2,
                                                learning_rate=.01,window_policy="covering_targets")
    steps = [session.step().to_json() for _ in range(2)]
    require(session.exhausted and all(json.loads(s)["target_tokens"] == 5 for s in steps), "training coverage")
    loaded = st.NativeCausalLm.from_session_checkpoint_at(base,session.to_json(),session.checkpoint_id,
                 update=2,recorded_data_id=session.data_id,post_lora_id=session.model.lora_artifact_id,
                 post_geometry_id=session.model.geometry_artifact_id)
    ev = st.NativeCausalLm.evaluate_variants_by_answer([model,loaded],[content],9,labels=[mask])
    require([e.target_tokens for e in ev.evaluations] == [4,4], "evaluation includes the terminator")
    cfg = json.dumps(dict(max_new_tokens=3,execution=dict(kind="kv_cache",max_values=1024)))
    gen = st.NativeCausalLmGenerator(loaded,prompt,cfg)
    require(gen.token_ids == spec["prompt_ids"], "generation includes the reference")
    decisions = [gen.step().to_json() for _ in range(3)]
    require(not banned.intersection(n.partition(".")[0] for n in sys.modules), "foreign runtime imported")
    signature = dict(data_id=session.data_id,step_reports=digest("\n".join(steps)),checkpoint=session.checkpoint_id,
                     evaluation=digest(ev.to_json()),prompt=prompt,generated_ids=gen.generated_ids,
                     decisions=digest("\n".join(decisions)),cursor=gen.checkpoint_id)
    return dict(identity=identity,signature=signature)


if __name__ == "__main__":
    require(sys.flags.isolated and sys.dont_write_bytecode,"use python -I -B")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("manual","views"))
    p.add_argument("wheel",type=Path)
    p.add_argument("sha256")
    p.add_argument("--expect",type=Path,help="also require the archived synthetic signature")
    args=p.parse_args()
    result=run(args.mode,args.wheel,args.sha256)
    if args.expect is not None:
        expected=json.loads(args.expect.read_bytes())
        require(result["signature"] == expected["matched_signature"], "synthetic training/evaluation/generation differs")
    print(json.dumps(result,sort_keys=True))
