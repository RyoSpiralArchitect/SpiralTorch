"""No models or optimizers. Check every factual receipt and recompute the factorial contrasts."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as d  # noqa: E402
m = d.m
STUDY = "c82c55b429a0f661435d3acf4bb3f86d2d7880a5"
FILES = ("result.json","manifest.json","attempt.json","worker.json","exit.json")


def lines(root, archived):
    path = root/("observations.jsonl.gz" if archived else "observations.jsonl")
    m.check(path.is_file() and not path.is_symlink() and path.stat().st_size<=576*8192, "journal file")
    with (gzip.open(path,"rb") if archived else path.open("rb")) as f:
        for _ in range(576):
            line=f.readline(8193)
            m.check(len(line)<=8192 and line.endswith(b"\n"),"incomplete journal")
            yield line
        m.check(f.read(1)==b"","extra observations")


def state_vs_operator(orders):
    """An algebraic product, not an assertion that hidden states are equal across owners."""
    out={}
    for o,arms in orders.items():
        out[o]={}
        for key in d.MEASURES:
            a,w=(arms[name][key]["pooled_relative_rms"] for name in d.SOURCES)
            out[o][key]=dict(factual_wave_over_affine=d.fraction(w["both"],a["affine"]),
                same_wave_owner_operator=d.fraction(w["both"],w["affine"]),
                same_plain_map_owner_difference=d.fraction(w["affine"],a["affine"]),
                same_affine_owner_operator=d.fraction(a["both"],a["affine"]),
                same_full_map_owner_difference=d.fraction(w["both"],a["both"]))
    return out


def verify(root, archived=False, originals=None):
    res,mf,attempt,worker,done=(m.read(root/n) for n in FILES)
    m.check(res["schema"]=="spiraltorch.local_wave_factorial_study.v1" and mf["schema"]=="spiraltorch.wave_factorial_inputs.v1"
            and (res["model_calls"],res["local_maps"],res["updates"],res["eval_windows"])==(576,2304,0,0)
            and res["manifest_sha256"]==m.sha(root/"manifest.json") and mf["study_commit"]==attempt["commit"]==STUDY,
            "study scope or manifest")
    m.check(attempt["limit_seconds"]==1800 and attempt["retries"]==0 and done["status"]==0
            and attempt["started_epoch"]<=worker["started_epoch"]<=res["finished_epoch"]<=done["finished_epoch"]
            and done["finished_epoch"]-attempt["started_epoch"]<=1800 and 0<res["cpu_seconds"]<=worker["cpu_limit"]<=1800,
            "attempt or deadline")
    build=m.read(d.HERE/"build.json")
    m.check(mf["build"]==build and build["core_commit"]==d.CORE and mf["installed"]["all_original_files_match"]
            and all(build[key]==mf["installed"][key] for key in ("native_sha256","wheel_sha256"))
            and mf["config"]==d.CFG and mf["factual_archive_sha256"]==d.ANCHOR and mf["base"]==m.BASE
            and mf["base_sha256"]==m.MODEL and mf["train_sha256"]==m.TRAIN,"wheel, base or recipe identity")
    paths=[d.HERE/n for n in ("study.py","design.py","build.json")]+[
            d.PREV/"observe.py",d.PREV/"metrics.py",m.ROOT/"tools/owned_process.py",m.ROOT/"tools/byte_release_artifacts.py"]
    m.check(mf["source_sha256"]=={str(p.relative_to(m.ROOT)):m.sha(p) for p in paths},"instrument source")
    entries=m.boundaries()
    m.check(mf["original_checkpoint_ids"]=={o:{a:entries[o,a]["checkpoint_id"] for a in d.SOURCES} for o in m.ORDERS},
            "wrong checkpoint roster")
    expected=d.anchors()
    m.check(len(mf["plan"])==m.N and all([r["key"],r["observation"]["window_shape"][1],r["observation"]["target_tokens"]]
            ==mf["plan"][i] for (_,_,i),r in expected.items()),"different training keys, lengths or target masks")
    if originals is not None:
        train=originals/"train.json"
        m.check(m.sha(train)==m.TRAIN and d.old.train_plan(m.read(train))==mf["plan"]
                and all("sha256:"+m.sha(originals/o/(a+".ck"))==entries[o,a]["checkpoint_id"] for o in m.ORDERS for a in d.SOURCES),
                "original train file or checkpoint bytes")
    if archived:
        meta=m.read(root/"archive.json")
        m.check(meta["file_sha256"]=={n:m.sha(root/n) for n in (*FILES,"observations.jsonl.gz")}
                and meta["uncompressed_sha256"]==res["journal_sha256"],"archive pins")
    h=hashlib.sha256()
    def feed():
        for line in lines(root,archived):
            h.update(line)
            yield line
    orders=d.summarize(feed(),expected)
    m.check(orders==res["orders"] and h.hexdigest()==res["journal_sha256"],"factorial summary or journal changed")
    for arms in orders.values():
        for stats in arms.values():
            for measure in d.MEASURES:
                v=stats[measure]
                c=v["decomposition"]
                if c is not None:
                    m.check(math.isclose(c["saturation_average"]+c["projection_average"],c["total_reduction"],abs_tol=2e-12)
                            and math.isclose(c["total_reduction"],1.-c["rms_relative_to_affine"]["both"]**2,abs_tol=2e-12),
                            "energy versus RMS attribution")
    products=state_vs_operator(orders)
    for both in products.values():
        for v in both.values():
            for path in (("same_wave_owner_operator","same_plain_map_owner_difference"),("same_affine_owner_operator","same_full_map_owner_difference")):
                if all(v[k] is not None for k in (*path,"factual_wave_over_affine")):
                    m.check(math.isclose(v[path[0]]*v[path[1]],v["factual_wave_over_affine"],rel_tol=2e-12),"owner/operator product")
    return dict(model_executions_by_verifier=0,factual_matches=576,local_maps=2304,unique_train_docs=96,
                original_files_checked=7 if originals else 0,updates=0,heldout=0,owner_operator=products)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    ap.add_argument("--archive",action="store_true")
    ap.add_argument("--originals",type=Path)
    ap.add_argument("--export",type=Path)
    a=ap.parse_args()
    result=verify(a.directory,a.archive,a.originals)
    if a.export is not None:
        m.check(not a.archive and a.originals is not None,"verify original bytes first")
        a.export.mkdir(exist_ok=False)
        for name in FILES:
            shutil.copyfile(a.directory/name,a.export/name)
        with (a.export/"observations.jsonl.gz").open("xb") as out, gzip.GzipFile(filename="",mode="wb",fileobj=out,mtime=0) as z:
            for line in lines(a.directory,False):
                z.write(line)
        m.write(a.export/"archive.json",dict(file_sha256={n:m.sha(a.export/n) for n in (*FILES,"observations.jsonl.gz")},
                uncompressed_sha256=m.read(a.export/"result.json")["journal_sha256"]))
        verify(a.export,True)
    print(json.dumps(result,sort_keys=True))
