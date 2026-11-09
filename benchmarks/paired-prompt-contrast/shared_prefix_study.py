"""CPU wheel exercise: learn to stop or continue after an identical first answer token.

No pretrained inputs. Preparation, training, label-free scoring, and model-free
readback are separate commands. Each writes only to a new output directory.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys


ARMS = {"gold_ce": 0., "line_mix": .5}
SITE = "transformer.h.1.attn"
STOP, VOCAB, RATE = 1, 64, .035


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, data):
    with path.open("xb") as f:
        f.write(data)
    return sha(data)


def put(path, value):
    return write(path, pack(value))


def read(path, pin, limit=262144):
    with path.open("rb") as f:
        data = f.read(limit+1)
    require(len(data) <= limit and sha(data) == pin, "changed or oversized input: "+path.name)
    return data


def obj(path, pin, limit=262144):
    return json.loads(read(path,pin,limit))


def record(f, item):
    f.write(pack(item)+b"\n")
    f.flush()


def roster():
    train, queries, truth = [], [], []
    contents = ([23], [23, 2], [23, 14])
    for i in range(4):
        for j in range(3):
            cue = (i+j)%3
            slot = len(train)
            train.append(dict(slot=slot, prompt=[50,32+i,24+cue,29+i%3], answer=contents[cue]))
            queries.append(dict(slot=slot, prompt=[51,24+cue,32+i,29+i%3]))
            truth.append(dict(slot=slot, answer=contents[cue]))
    return train, queries, truth


def prepare(out):
    src = Path(__file__).with_name("projection_language_fixture.py")
    spec = importlib.util.spec_from_file_location("synthetic_weights",src)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out.mkdir(mode=0o700)
    train, qs, truth = roster()
    cfg = dict(model_type="gpt2", vocab_size=VOCAB, n_positions=16, n_embd=32,
               n_head=4, n_layer=2, n_inner=48, tie_word_embeddings=False)
    pins = {name:put(out/(name+".json"),v) for name,v in
            (("config",cfg),("train",train),("queries",qs),("truth",truth))}
    pins["weights"] = write(out/"model.safetensors",gen.weights())
    return put(out/"fixture.json",dict(kind="shared_content_stop_or_continue.v1", files=pins,
        arms=ARMS, rate=RATE, stop=STOP, training=12, evaluation=12,
        support="Distinct sorted training contents only. All share content ID 23.",
        control="Same anchored geometry, frozen base/LoRA, prompt order, menu and per-step length. No endpoint rescaling.",
        gates="On 12 different synthetic prompts: report lower gold conditional NLL separately from strictly more full-line winners without fewer short-answer winners.",
        cap=dict(updates=24, gradients=24, pre_and_post_reports=48, score_reports=36, prefix_rows=144,
                 wall_and_cpu_seconds=120, retries=0), generator=sha(src.read_bytes())))


def model(st, folder, pins):
    raw = read(folder/"model.safetensors",pins["weights"],100000)
    base = st.NativeCausalLm.from_gpt2(read(folder/"config.json",pins["config"]).decode(),
                                      st.SafeTensorArchive.from_bytes(raw)).with_readout_accumulator("f32_wide_backward")
    lora = json.loads(base.with_lora(1,2,3.).lora_to_json())
    lora["b"] = [(i%9-4)/512 for i in range(64)]
    tensors = [dict(name=k,shape=[32],role="parameter",requires_grad=True,
                    values=[.5+(i%7-3)/128 if k=="gate" else (i%5-2)/128 for i in range(32)])
               for k in ("gate","bias")]
    projection = dict(schema="spiraltorch.wave_stage_adapter.v1", features=32, strength=.5, max_values=4096,
                      stage=dict(kind="projection",curvature=-1.,log_radius=0.))
    art = st.GeometryAdapterArtifact.create_json(json.dumps(dict(schema="spiraltorch.geometry_adapter_artifact.v1",
        base_model_identity=base.base_model_identity,policy="full_prefix_f32_no_cache.v1",training=True,
        placements=[dict(path=SITE,training=True,recipe=projection,tensors=tensors)])))
    return base.with_lora_json(json.dumps(lora)).with_geometry(art.anchor_wave_projection(SITE))


def save_geo(out, name, m):
    art=m.geometry_artifact()
    require("sha256:"+write(out/(name+".geometry.json"),art.to_json().encode())==art.artifact_id,"geometry identity")


def fit(samples):
    menu=sorted({tuple(s["answer"]) for s in samples})
    require(menu==[(23,),(23,2),(23,14)] and len(samples)==12 and
            all(sum(tuple(s["answer"])==c for s in samples)==4 for c in menu),"unexpected train-only support")
    return [list(c) for c in menu]


def train(folder,pin,out):
    import spiraltorch as st
    fx=obj(folder/"fixture.json",pin);pins=fx["files"]
    samples=obj(folder/"train.json",pins["train"]);menu=fit(samples)
    out.mkdir(mode=0o700)
    initial=model(st,folder,pins);frozen=initial.lora_to_json()
    save_geo(out,"start",initial);arms={a:initial for a in ARMS}
    with (out/"steps.jsonl").open("xb") as f:
        for i,s in enumerate(samples):
            radius=None;target=menu.index(s["answer"]);prompt=s["prompt"]
            for a,mix in ARMS.items():
                m=arms[a];kw=dict(gold_index=target,stop_id=STOP,mix=mix);pre=m.geometry_artifact_id
                record(f,dict(event="start",step=i+1,arm=a,pre=pre))
                before=m.line_menu_loss(prompt,menu,**kw)
                g=m.geometry_line_menu_gradients(prompt,menu,**kw)
                require(before==g.line_menu_report() and before["loss"]==g.loss and g.target_tokens==8 and g.window_shape==[3,7] and
                        g.geometry_artifact_id==pre and g.lora_artifact_id==m.lora_artifact_id and
                        g.base_model_identity==m.base_model_identity and g.readout_accumulator=="f32_wide_backward" and
                        before["shared_content_tokens"]==1,"wrong objective or gradient owner")
                values=g.parameter_gradients()
                require(set(values)=={SITE} and set(values[SITE])=={"bias","gate"},"unexpected parameter group")
                bits=[v for k in ("bias","gate") for v in values[SITE][k]]
                require(len(bits)==64 and all(map(math.isfinite,bits)),"invalid gradients")
                if a=="gold_ce":
                    nxt=m.step_adaptation_sgd(g,"geometry",RATE);match=None
                    radius=m.geometry_artifact().trainable_delta_l2(nxt.geometry_artifact())
                    require(radius>0.,"no reference step")
                else:
                    nxt,match=m.step_geometry_sgd_at_distance(g,RATE,radius)
                    require(match["requested_l2"]==radius and match["relative_vector_error"]<=1e-4 and
                            abs(match["realized_l2"]/radius-1.)<=1e-4,"unmatched step")
                require(nxt.lora_to_json()==frozen and nxt.base_model_identity==initial.base_model_identity,"frozen owner changed")
                after=nxt.line_menu_loss(prompt,menu,**kw)
                save_geo(out,f"{a}-{i+1}",nxt)
                record(f,dict(event="return",step=i+1,arm=a,pre=pre,post=nxt.geometry_artifact_id,reference=radius,
                    match=match,before=before,after=after,gradient_bits=struct.unpack("<64I",struct.pack("<64f",*bits))))
                arms[a]=nxt
    menu_pin=put(out/"menu.json",menu)
    return put(out/"training.json",dict(fixture=pin,base=initial.base_model_identity,lora=initial.lora_artifact_id,
        origin=initial.geometry_artifact_id, endpoints={a:m.geometry_artifact_id for a,m in arms.items()},
        menu=menu_pin,journal=sha((out/"steps.jsonl").read_bytes()),gradients=24,updates=24,objective_forwards=48))


def score(folder,pin,trained,training_pin,out):
    import spiraltorch as st
    fx=obj(folder/"fixture.json",pin);qs=obj(folder/"queries.json",fx["files"]["queries"])
    tr=obj(trained/"training.json",training_pin);menu=obj(trained/"menu.json",tr["menu"])
    origin=model(st,folder,fx["files"]);states={"start":origin}
    require(tr["fixture"]==pin and (origin.base_model_identity,origin.lora_artifact_id,origin.geometry_artifact_id)==
            (tr["base"],tr["lora"],tr["origin"]),"wrong evaluation origin")
    for a in ARMS:
        owner=tr["endpoints"][a]
        art=st.GeometryAdapterArtifact.from_json(read(trained/(a+"-12.geometry.json"),owner[7:]).decode(),owner,tr["base"])
        states[a]=origin.with_geometry(art)
    parts=sorted({tuple(line[:j]) for line in menu for j in range(len(line)+1)})
    require(len(states)==3 and len(parts)==4 and len(qs)==12,"score budget")
    out.mkdir(mode=0o700);nr=reports=0
    with (out/"rows.jsonl").open("xb") as f:
        for i,q in enumerate(qs):
            require(set(q)=={"slot","prompt"} and q["slot"]==i,"label leaked into scorer")
            prompt=q["prompt"]
            for name,m in states.items():
                ident=dict(state=name,owner=m.geometry_artifact_id,slot=i)
                record(f,dict(event="start",**ident))
                # This fixed index is not the answer key. All lines and their costs are returned.
                r=m.line_menu_loss(prompt,menu,gold_index=0,stop_id=STOP,mix=1.);reports+=1
                rows=[]
                for part in parts:
                    ids=[*prompt,*part]
                    buf=bytes(m.forward(ids,[1,len(ids)],last_only=True).values_buffer())
                    rows.append(dict(prefix=part,bits=struct.unpack("<64I",buf)));nr+=1
                record(f,dict(event="return",**ident,report=r,rows=rows))
    require((reports,nr)==(36,144),"missing scored cells")
    return put(out/"scoring.json",dict(fixture=pin,training=training_pin,journal=sha((out/"rows.jsonl").read_bytes()),
        reports=reports,prefix_forwards=nr,owners={a:m.geometry_artifact_id for a,m in states.items()}))


def main(args):
    if len(args)==2 and args[0]=="prepare":
        return prepare(Path(args[1]))
    if len(args)==4 and args[0]=="train":
        return train(Path(args[1]),args[2],Path(args[3]))
    if len(args)==6 and args[0]=="score":
        return score(Path(args[1]),args[2],Path(args[3]),args[4],Path(args[5]))
    raise SystemExit("prepare NEW_FIXTURE | train FIXTURE PIN NEW_TRAIN | score FIXTURE PIN TRAIN TRAIN_PIN NEW_SCORES")


if __name__=="__main__":
    print(main(sys.argv[1:]))
