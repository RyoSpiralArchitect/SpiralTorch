"""One-shot, label-blind transfer comparison on new matched symbolic contexts. No training."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_interference import signature
from alias_subset_study import MENU, STOP, native, open_model
from shared_prefix_study import obj, pack, put, read, record, require, sha

POOLS = ("familiar", "novel")
ORDERS = ("original", "swapped")
STARTS = ("initial", "subset_rank")
ROLES = ("origin", "four_group", "individual_guard")
PHASES = ("reference", "native")
EPS, EFFECT, TOKEN_TOL, WALL = 5e-6, .01, 2e-5, 120
BUDGET = dict(prompts=72, triplets=24, blocks=6, distinct_weights=5, reference_menu=360,
              reference_tokens=4680, native_menus=360, weight_updates=0, free_tokens=0)


def sources():
    names = ("alias_fresh_transfer", "alias_fresh_transfer_audit", "alias_subset_study",
             "alias_interference", "shared_prefix_study", "projection_language_fixture")
    return {n: signature(Path(__file__).with_name(n+".py")) for n in names}


def dataset():
    queries, key = [], []
    for pool in POOLS:
        for order in ORDERS:
            for block in range(6):
                entity = (32 if pool == "familiar" else 53) + block
                side = 27 + (1 - block % 2)
                for c in range(3):
                    a, b = 24+c, 29+c
                    if order == "swapped":
                        a, b = b, a
                    prompt = [49, entity, a, side, b]
                    qid = sha(pack(prompt))
                    queries.append(dict(id=qid, prompt=prompt))
                    key.append(dict(id=qid, pool=pool, order=order, block=block, meaning=c))
    queries.sort(key=lambda q: q["id"])
    require(len({q["id"] for q in queries}) == BUDGET["prompts"], "nonunique prompt")
    return queries, key


def freeze(trained, fixture, prior_pin, extension, wheel, out):
    ref = obj(trained/"readback.json", prior_pin)
    roster = obj(trained/"endpoint-roster.json", ref["endpoint_roster"])
    p = obj(trained/"protocol/plan.json", ref["pins"]["plan"])
    m = obj(trained/"measured/complete.json", ref["pins"]["measurement"])
    part = obj(trained/"measured/part1/complete.json", m["parts"][1])
    require(p["kind"] == "spiraltorch.alias_guarded_chain.v1" and p["horizon"] == 128 and
            part["plan"] == m["plan"] == roster["plan"] == ref["pins"]["plan"] and
            (roster["base"],roster["lora"],roster["native"]) == (p["base"],p["lora"],p["native"]), "foreign training run")
    expected = {s+"/origin":p["origins"][s] for s in STARTS}
    for s in STARTS:
        for a in ROLES[1:]:
            boundary = part["endpoints"][s+"/"+a]
            require((boundary["state"],boundary["arm"],boundary["cursor"],boundary["plan"]) ==
                    (s,a,128,m["plan"]), "not the frozen final endpoint")
            expected[s+"/"+a] = boundary["owner"]
    owners = sorted(set(expected.values()))
    require(roster["roles"] == expected and set(roster["artifacts"]) == set(owners) and len(owners) == 5 and
            expected["subset_rank/four_group"] == expected["subset_rank/individual_guard"] and
            expected["initial/four_group"] != expected["initial/individual_guard"], "different role roster")
    for digest in (extension,wheel):
        require(len(digest)==64 and all(k in "0123456789abcdef" for k in digest), "invalid binary pin")
    fx = obj(fixture/"fixture.json",p["fixture"])
    train = obj(fixture/"train.json",fx["files"]["train.json"])
    roster18 = {(49,32+b,24+c,27+b%2,29+c) for b in range(6) for c in range(3)}
    require(fx["menu"]==MENU and fx["stop"]==STOP and len(train)==18 and
            {tuple(row["prompt"]) for row in train}==roster18 and
            all(r["prompt"][2] == 24+r["meaning"] for r in train), "different training roster")
    queries, key = dataset()
    require(all(tuple(r["prompt"]) not in roster18 for r in queries), "train/transfer overlap")
    out.mkdir(mode=0o700)
    inputs = out/"inputs"
    inputs.mkdir(mode=0o700)
    for filename,k in (("model.safetensors","weights"),("config.json","config.json")):
        put_bytes(inputs/filename, read(fixture/filename,fx["files"][k],100000))
    for owner in owners:
        require(owner.startswith("sha256:") and len(owner)==71 and all(c in "0123456789abcdef" for c in owner[7:]),
                "invalid owner")
        name = owner[7:]+".json"
        require(roster["artifacts"][owner]=="measured/geometry/"+name, "noncanonical artifact path")
        put_bytes(inputs/name, read(trained/"measured/geometry"/name,owner[7:],100000))
    qp, kp = put(inputs/"queries.json",queries), put(out/"answer-key.json",key)
    return put(out/"plan.json",dict(kind="spiraltorch.alias_matched_context_transfer.v1", code=sources(),
        prior_readback=prior_pin, endpoint_roster=ref["endpoint_roster"], training_plan=m["plan"],
        roles=expected, owners=owners, base=p["base"], lora=p["lora"],
        files={k:fx["files"][k] for k in ("weights","config.json")}, train_sha=fx["files"]["train.json"],
        query_pin=qp, key_pin=kp, menu=MENU, stop=STOP, target=5,
        runtimes=dict(reference=dict(extension=p["native"], wheel=p["wheel"]), native=dict(extension=extension,wheel=wheel)),
        budget=BUDGET, wall=WALL, retries=0, eps=EPS, effect=EFFECT, token_tolerance=TOKEN_TOL,
        cells="Two identity pools (32..37 familiar, 53..58 novel) x original/swapped state cue order. "
              "All use the opposite side code to the matched training block. Both codes still denote the same state. "
              "No training prompt is repeated. Legacy eval files are not used. Six crossed blocks are not 24 independent seeds.",
        primary="Initial guard vs initial four-group: in EVERY cell the positive-context extra log gain B must exceed "
                "5e-6 and the difference D=B-mean(wrong-context extra gains) must exceed 0.01 nats for BOTH absolute "
                "and within-menu P(standby); at least 4/6 SAME blocks per cell must improve both gaps by >5e-6. "
                "0.01 is a practical threshold, not a significance level.",
        containment="Separate: correct gains positive and no mean extra log gain >5e-6 in either wrong class, in both readouts.",
        broad_lift="Separate: in every cell, correct AND pooled incorrect gains >0.01, and wrong gain >=75% of correct gain. "
                   "Report absolute and menu separately. Mixed, null and adverse results are retained.",
        scoring="Five distinct frozen geometries; no labels/key at scoring. Reference uses a fixed dummy partition [0,3] "
                "and 13 independent target-token calls. New native API uses one label-free shared-trie readout. "
                "The stored shared-trie line totals, union NLL and winner must match exactly between wheels on this fixture. "
                "Independent per-token and summed full-line costs must agree within 2e-5. "
                "Open the key only after both journals are closed. No new training or generation."))


def put_bytes(path, data):
    with path.open("xb") as f:
        f.write(data)


def hide_key(protocol):
    key = (protocol/"answer-key.json").resolve()
    def deny(event, args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.name=="answer-key.json" and path.resolve()==key:
                raise PermissionError("answer key cannot be opened by either scorer")
    sys.addaudithook(deny)


def reference(model, prompt):
    report = model.line_subset_loss(prompt, MENU, accepted_indices=[0,3], stop_id=STOP)
    cache, terms = {}, []
    for line in MENU:
        prefix, v = list(prompt), []
        for token in [*line, STOP]:
            k = (tuple(prefix),token)
            if k not in cache:
                r = model.next_token_score(prefix,token)
                require((r["target_id"],r["vocabulary"])==(token,64), "foreign target-token score")
                cache[k] = r["nll"]
            v.append(cache[k])
            prefix.append(token)
        terms.append(v)
    require(len(cache)==13 and report["accepted_indices"]==[0,3], "reference domain changed")
    return dict(line_nlls=report["line_nlls"],menu_probabilities=report["menu_probabilities"],token_nlls=terms,
                winner_index=report["winner_index"],menu_set_nll=report["menu_set_nll"],stop_id=report["stop_id"],
                target_counts=report["target_counts"],prefix_nodes=report["prefix_nodes"],decoder_lanes=report["decoder_lanes"])


def worker(protocol, pp, root, phase):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(WALL//2,WALL//2))
    hide_key(protocol)
    p = obj(protocol/"plan.json",pp)
    require(phase in PHASES and p["code"]==sources(), "unexpected runtime/protocol")
    att = dict(plan=pp,query_pin=p["query_pin"],phases=list(PHASES),attempt=1)
    require(json.loads((protocol/"consumed.json").read_bytes())==att==json.loads((root/"attempt.json").read_bytes()),
            "no admission for this scoring run")
    if phase=="native":
        require(json.loads((root/"reference-exit.json").read_bytes())["exit"]==0 and
                (root/"reference/complete.json").is_file(), "reference did not finish")
    inputs = protocol/"inputs"
    queries = obj(inputs/"queries.json",p["query_pin"])
    require(len(queries)==72 and all(set(q)=={"id","prompt"} and sha(pack(q["prompt"]))==q["id"] for q in queries) and
            [q["id"] for q in queries]==sorted({q["id"] for q in queries}), "invalid blind query deck")
    st = native()
    ext = signature(st._rs.__file__)
    require(ext==p["runtimes"][phase]["extension"], "different installed extension")
    base = open_model(st,inputs,p["files"])
    require((base.base_model_identity,base.lora_artifact_id,base.readout_accumulator,base.geometry_artifact_id)==
            (p["base"],p["lora"],"f32_wide_backward",p["roles"]["initial/origin"]),
            "frozen base or policy mismatch")
    part = root/phase
    part.mkdir(mode=0o700)
    n=0
    with (part/"scores.jsonl").open("xb") as f:
        for owner in p["owners"]:
            raw = read(inputs/(owner[7:]+".json"),owner[7:],100000)
            model = base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(),owner,p["base"]))
            for q in queries:
                tags = dict(owner=owner,id=q["id"])
                record(f,dict(event="start",**tags))
                if phase=="native":
                    r = model.score_lines(q["prompt"],MENU,stop_id=STOP)
                else:
                    r = reference(model,q["prompt"])
                require(model.geometry_artifact_id==owner and (model.lora_artifact_id,model.base_model_identity)==(p["lora"],p["base"]),
                        "frozen model changed")
                record(f,dict(event="return",**tags,report=r))
                n+=1
    require(n==360, "wrong number of scored prompts")
    return put(part/"complete.json",dict(plan=pp,phase=phase,extension=ext,pid=os.getpid(),query_pin=p["query_pin"],
        owners=p["owners"],menus=n,separate_tokens=(13*n if phase=="reference" else 0),journal=signature(part/"scores.jsonl")))


def claim(protocol, pp, query_pin, out):
    attempt=dict(plan=pp,query_pin=query_pin,phases=PHASES,attempt=1)
    put(protocol/"consumed.json",attempt)
    put(out/"attempt.json",attempt)


def run(protocol, pp, old_py, new_py, out):
    p = obj(protocol/"plan.json",pp)
    require(p["code"]==sources() and p["budget"]==BUDGET and (p["wall"],p["retries"])==(WALL,0), "unfrozen study")
    out.mkdir(mode=0o700)
    claim(protocol, pp, p["query_pin"], out)
    env = dict(os.environ, SPIRALTORCH_BACKEND="cpu", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    deadline = time.monotonic()+WALL
    phases={}
    for phase,exe in zip(PHASES,(old_py,new_py),strict=True):
        log=out/(phase+".log")
        cmd=[str(exe),"-I","-B",str(Path(__file__).resolve()),"_worker",str(protocol),pp,str(out),phase]
        env["SPIRALTORCH_AUTOTUNE_STORE"]=str(out/(phase+"-tuning.json"))
        with log.open("xb") as f:
            try:
                status=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,env=env,
                                      timeout=max(0.,deadline-time.monotonic())).returncode
            except subprocess.TimeoutExpired:
                status="timeout"
        elapsed=WALL-(deadline-time.monotonic())
        put(out/(phase+"-exit.json"),dict(exit=status,elapsed=elapsed,log=signature(log)))
        require(status==0 and elapsed<=WALL,"first failure retained, no retry: "+phase)
        phases[phase]=signature(out/phase/"complete.json")
        require(log.read_text().splitlines()[-1:]==[phases[phase]],"worker receipt mismatch")
    return put(out/"complete.json",dict(plan=pp,phases=phases))


if __name__=="__main__":
    from alias_fresh_transfer_audit import audit
    cmd,*a=sys.argv[1:]
    if cmd=="freeze" and len(a)==6:
        print(freeze(Path(a[0]),Path(a[1]),*a[2:5],Path(a[5])))
    elif cmd=="run" and len(a)==5:
        # Preserve venv executable paths: resolving the python symlink would escape its venv.
        print(run(Path(a[0]).resolve(),a[1],Path(a[2]).absolute(),Path(a[3]).absolute(),Path(a[4]).resolve()))
    elif cmd=="_worker" and len(a)==4:
        print(worker(Path(a[0]),a[1],Path(a[2]),a[3]))
    elif cmd=="audit" and len(a)==5:
        print(audit(Path(a[0]),a[1],Path(a[2]),a[3],Path(a[4])))
    else:
        raise SystemExit("freeze TRAINED FIXTURE READBACK_PIN NEW_EXT NEW_WHEEL OUT | run PLAN PIN OLD_PY NEW_PY OUT | audit PLAN PIN MEASURED MP OUT")
