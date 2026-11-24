"""Dense vs two bounded complete-answer vocabulary VJPs; only the pinned artificial model."""
import json
import math
import os
from pathlib import Path
import platform
import runpy
import statistics
import subprocess
import sys
import time

common = runpy.run_path(str(Path(__file__).with_name("stream_line_bench.py")))
V, PROMPT, LINES = (common[k] for k in ("VOCAB","PROMPT","LINES"))
save, peak, sha = (common[k] for k in ("save","peak_bytes","digest"))
ARMS = ("old_dense","auto","one_row")
OBJECTIVES = {"set":{}, "menu":{"gold_index":2,"mix":0.5}, "subset":{"accepted_indices":[0,2]}}
WEIGHTS = "ea368f69c6e7474a9d8eb9cf28f4b2353f18e22eb1a537e91576a72a79602770"
GEOMETRY = "sha256:f6ab09c248ffbd54341433381964623b43185bd43cde03c9eb5f462be9019d67"


def worker(fixture, out, arm, kind):
    sys.addaudithook(common["no_foreign"])
    import spiraltorch as st
    f = json.loads((fixture/"fixture.json").read_bytes())
    m = st.NativeCausalLm.from_gpt2(f["config"],st.SafeTensorArchive(str(fixture/"synthetic.safetensors")))
    m = m.with_geometry(st.GeometryAdapterArtifact.from_json(
        f["geometry"], expected_id=GEOMETRY, expected_base_model_identity=f["base_id"]))
    assert m.geometry_artifact_id == GEOMETRY and m.base_model_identity == f["base_id"]
    kw = dict(OBJECTIVES[kind],stop_id=0)
    if arm == "one_row":
        kw["max_tile_values"] = V
    before = peak()
    start = time.perf_counter_ns()
    g = getattr(m,"geometry_line_"+kind+"_gradients")(PROMPT,LINES,**kw)
    elapsed = time.perf_counter_ns()-start
    after = peak()
    if arm == "old_dense":
        assert not hasattr(g,"line_training_work")
        work = None
    else:
        work = g.line_training_work()
        tile = 5 if arm=="auto" else 1
        batches = (129+tile-1)//tile
        assert work == dict(prefix_rows=129,vocabulary=V,tile_rows=tile,forward_head_calls=2*batches-1,
                            backward_head_calls=batches,replayed_head_rows=129-tile,
                            peak_logit_values=tile*V,peak_extra_seed_values=V)
    r = getattr(g,"line_"+kind+"_report")()
    assert r["target_counts"] == [33]*4 and r["prefix_nodes"] == 129 and r["decoder_lanes"] == 4
    assert m.geometry_artifact_id == g.geometry_artifact_id == GEOMETRY
    assert g.target_tokens == 132 and r["loss"] == g.loss
    params = g.parameter_gradients()
    assert sum(v != 0 for group in params.values() for block in group.values() for v in block) > 0
    save(out,dict(arm=arm,kind=kind,report=r,parameters=params,work=work,ns=elapsed,
                  rss_before=before,rss_after=after,rss_growth=after-before))


def error(a,b):
    if isinstance(a,dict):
        assert a.keys() == b.keys()
        return max(error(a[k],b[k]) for k in a)
    if isinstance(a,list):
        assert len(a) == len(b)
        return max(error(i,j) for i,j in zip(a,b))
    if isinstance(a,int):
        assert a == b
    d = abs(a-b)
    assert math.isfinite(d) and d <= 3e-6, (a,b)
    return d


def compare(old,new,fixture,out):
    out.mkdir(mode=0o700)
    assert sha((fixture/"synthetic.safetensors").read_bytes()) == WEIGHTS
    env = dict(os.environ,RAYON_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    fields=("ns","rss_before","rss_after","rss_growth")
    measurements, medians = [], {}
    for kind in OBJECTIVES:
        ref = None
        for i in range(3):
            order=ARMS[i:]+ARMS[:i]
            for arm in order:
                f=out/("%s-%d-%s.json"%(kind,i,arm))
                subprocess.run([old if arm=="old_dense" else new,"-IB",str(Path(__file__).resolve()),
                                "worker",str(fixture),str(f),arm,kind],check=True,env=env,timeout=30)
                r=json.loads(f.read_bytes())
                assert r["arm"] == arm and r["kind"] == kind
                if ref is None:
                    assert arm == "old_dense"
                    ref = r
                record={k:r[k] for k in fields}
                record.update(kind=kind,arm=arm,rotation=i,raw_sha256=sha(f.read_bytes()))
                for k in ("report","parameters"):
                    record[k+"_max_abs"]=error(ref[k],r[k])
                    record[k+"_exact"]=json.dumps(ref[k],sort_keys=True)==json.dumps(r[k],sort_keys=True)
                measurements.append(record)
        medians[kind]={a:{k:statistics.median(r[k] for r in measurements if r["kind"]==kind and r["arm"]==a)
                          for k in fields} for a in ARMS}
    save(out/"summary.json",dict(schema="spiraltorch.synthetic_line_vjp_memory.v1",
         platform=platform.platform(),weights_sha256=WEIGHTS,geometry_id=GEOMETRY,vocabulary=V,prefix_rows=129,
         objectives=OBJECTIVES,repetitions=3,threads_env=1,updates=0,medians=medians,measurements=measurements))


def verify(out):
    """Read back all raw records; never import the extension or call any decoder."""
    summary = json.loads((out/"summary.json").read_bytes())
    assert summary["weights_sha256"] == WEIGHTS and summary["geometry_id"] == GEOMETRY
    assert len(summary["measurements"]) == 27
    rows = {(r["kind"],r["rotation"],r["arm"]):r for r in summary["measurements"]}
    for kind in OBJECTIVES:
        reference = json.loads((out/(kind+"-0-old_dense.json")).read_bytes())
        for arm in ARMS:
            observed = []
            for i in range(3):
                data = (out/("%s-%d-%s.json"%(kind,i,arm))).read_bytes()
                r = json.loads(data)
                claim = rows[kind,i,arm]
                assert r["arm"] == arm and r["kind"] == kind and sha(data) == claim["raw_sha256"]
                assert r["rss_after"]-r["rss_before"] == r["rss_growth"] >= 0 and r["ns"] > 0
                for k in ("report","parameters"):
                    assert claim[k+"_max_abs"] == error(reference[k],r[k])
                    assert claim[k+"_exact"] == (json.dumps(reference[k],sort_keys=True)==json.dumps(r[k],sort_keys=True))
                for key in ("ns","rss_before","rss_after","rss_growth"):
                    assert r[key] == claim[key]
                observed.append(r)
            assert {k:statistics.median(r[k] for r in observed) for k in
                    ("ns","rss_before","rss_after","rss_growth")} == summary["medians"][kind][arm]
    print("Verified all 27 raw records, losses, parameter gradients, hashes and medians.")


if __name__=="__main__":
    if len(sys.argv)==6 and sys.argv[1]=="worker" and sys.argv[4] in ARMS and sys.argv[5] in OBJECTIVES:
        worker(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5])
    elif len(sys.argv)==6 and sys.argv[1]=="compare":
        compare(sys.argv[2],sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]))
    elif len(sys.argv)==3 and sys.argv[1]=="verify":
        verify(Path(sys.argv[2]))
    else:
        sys.exit("compare OLD_PY NEW_PY FIXTURE NEW_OUTPUT | verify OUTPUT")
