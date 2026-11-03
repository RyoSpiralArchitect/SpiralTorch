"""File-only admission, read-back and bounded export of the endpoint measurements."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as m  # noqa: E402
from observe import CORE, LIMIT, NATIVE, WHEEL, train_plan  # noqa: E402

FILES = ("manifest.json", "result.json", "attempt.json", "worker.json", "exit.json")
STUDY = "89778f7680a0c21e844843a30e3377e6dcb697cf"


def scalar_payload(r):
    """Fail closed before exporting any unknown fields from a future instrument."""
    obs = r["observation"]
    m.check(set(r) == {"arm", "order", "index", "key", "observation"} and set(obs) == {
            "schema", "base_model_identity", "lora_artifact_id", "geometry_artifact_id", "readout_accumulator",
            "mode", "window_shape", "target_tokens", "loss", "sites"} and set(obs["sites"]) == {m.SITE},
            "unexpected observation payload")
    s = obs["sites"][m.SITE]
    m.check(set(s) == {"elements", "input_rms", "residual_rms", "relative_residual_rms", "parameter_gradient_l2",
                       "pullback", "log_radius_gradient", "wave"} and set(s["pullback"]) ==
            {"input_adjoint_rms","correction_rms","upstream_rms","relative_correction_rms"}, "unexpected site payload")
    w = s["wave"]
    if w is not None:
        keys = {"schema","features","rows","log_radius","projection_radius","gate_outside_values","affine_outside_values",
                "affine_nonfinite_values","affine_abs_slope_mean","dimensionless_norm_mean","dimensionless_norm_max",
                "relative_radial_gain_mean","relative_radial_gain_min","relative_tangential_gain_mean","relative_tangential_gain_min"}
        m.check(set(w) == keys and all(type(w[k]) in (float, int) and math.isfinite(w[k]) for k in keys - {"schema"}),
                "unexpected conditioning payload")


def measurements(root, compressed):
    path = root / ("observations.jsonl.gz" if compressed else "observations.jsonl")
    m.check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 864 * 8192, "journal size/type")
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as f:
        for _ in range(864):
            line = f.readline(8193)
            m.check(0 < len(line) <= 8192 and line.endswith(b"\n"), "missing, partial or oversized observation")
            yield line
        m.check(f.read(1) == b"", "more than 864 observed windows")


def verify(root, originals=None, *, archived=False):
    mf, res, attempt, worker, done = (m.read(root / f) for f in FILES)
    m.check(res["manifest_sha256"] == m.sha(root / "manifest.json")
            and res["schema"] == "spiraltorch.native_endpoint_pullback.v1" and res["observations"] == 864
            and res["updates"] == mf["updates"] == res["eval_windows"] == 0 and res["unique_train_windows"] == 96,
            "invalid model execution budget or manifest")
    m.check(mf["schema"] == "spiraltorch.native_endpoint_pullback_manifest.v1" and mf["core_commit"] == attempt["core_commit"] == CORE
            and mf["study_commit"] == attempt["commit"] == STUDY
            and mf["train_sha256"] == attempt["train_sha256"] == m.TRAIN and mf["base_sha256"] == m.MODEL
            and mf["boundary_sha256"] == attempt["checkpoint_sets"] == list(m.BOUNDARIES) and mf["original_files"] == 10,
            "input ownership or source")
    files = (m.HERE / "observe.py", m.HERE / "metrics.py", m.ROOT / "tools/owned_process.py",
             m.ROOT / "tools/byte_release_artifacts.py")
    m.check(mf["source_sha256"] == {str(f.relative_to(m.ROOT)): m.sha(f) for f in files}
            and mf["installed"]["native_sha256"] == NATIVE and mf["installed"]["wheel_sha256"] == WHEEL
            and mf["installed"]["all_original_files_match"], "execution source or wheel")
    m.check(attempt["limit_seconds"] == LIMIT and attempt["retries"] == 0 and worker["retry"] is False and done["status"] == 0
            and attempt["started_epoch"] <= worker["started_epoch"] <= res["finished_epoch"] <= done["finished_epoch"]
            and done["finished_epoch"] - attempt["started_epoch"] <= LIMIT
            and 0 < res["cpu_seconds"] <= worker["limit_cpu"] <= LIMIT, "attempt or duration")
    if archived:
        meta = m.read(root / "archive.json")
        m.check(meta["file_sha256"] == {n:m.sha(root / n) for n in (*FILES,"observations.jsonl.gz")}
                and meta["uncompressed_sha256"] == res["journal_sha256"], "archive identity")
    if originals is not None:
        f = originals / "train.json"
        m.check(m.sha(f) == m.TRAIN and train_plan(m.read(f)) == mf["plan"], "train masks or plan")
        for (order, arm), entry in m.boundaries().items():
            m.check("sha256:" + m.sha(originals / order / (arm+".ck")) == entry["checkpoint_id"], "checkpoint bytes")

    hasher = hashlib.sha256()
    by_owner = {}
    def checked_lines():
        for line in measurements(root, archived):
            hasher.update(line)
            r = json.loads(line)
            scalar_payload(r)
            by_owner.setdefault((r["order"],r["arm"]), []).append(r["observation"]["sites"][m.SITE])
            yield line
    recomputed = m.summarize(checked_lines(), mf["plan"])
    m.check(hasher.hexdigest() == res["journal_sha256"] and res["orders"] == recomputed, "journal or summary mismatch")
    descriptions = {}
    for order in m.ORDERS:
        a, w = (by_owner[order,arm] for arm in m.ARMS[:2])
        for arm in m.ARMS:
            rows = by_owner[order,arm]
            # Independent identity: relative RMS is the RMS of per-document gains
            # weighted by its denominator energy, not a simple mean of gains.
            for metric, denom, back in (("relative_residual_rms","input_rms",False),
                                        ("relative_correction_rms","upstream_rms",True)):
                fields = [s["pullback"] if back else s for s in rows]
                weights = [s["elements"] * f[denom] ** 2 for s,f in zip(rows,fields)]
                pooled = math.sqrt(math.fsum(weight * f[metric] ** 2 for weight,f in zip(weights,fields)) / math.fsum(weights))
                m.check(math.isclose(pooled, recomputed[order][arm][metric], rel_tol=2e-12), "independent pooled gain")
        fixed, affine = (recomputed[order][arm] for arm in ("wave_fixed","affine"))
        crosses = [s["wave"]["affine_outside_values"] for s in w]
        descriptions[order] = dict(
            fixed_over_affine_residual=fixed["relative_residual_rms"]/affine["relative_residual_rms"],
            fixed_over_affine_correction=fixed["relative_correction_rms"]/affine["relative_correction_rms"],
            smaller_fixed_residual_docs=sum(x["relative_residual_rms"] < y["relative_residual_rms"] for x,y in zip(w,a)),
            smaller_fixed_correction_docs=sum(x["pullback"]["relative_correction_rms"] < y["pullback"]["relative_correction_rms"]
                                              for x,y in zip(w,a)),
            min_max_fixed_affine_threshold_crossings_per_doc=[min(crosses), max(crosses)])
    return dict(calls_verified=864, checkpoint_owners=9, unique_training_windows=96, evaluation_windows=0, updates=0,
                model_executions_by_verifier=0, original_files_verified=10 if originals is not None else 0,
                per_order=descriptions)


def export(source, dest):
    dest.mkdir(exist_ok=False)
    for n in FILES:
        shutil.copyfile(source / n, dest / n)
    with (dest / "observations.jsonl.gz").open("xb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as compressed:
        for line in measurements(source, False):
            compressed.write(line)
    m.write(dest / "archive.json", dict(uncompressed_sha256=m.read(source / "result.json")["journal_sha256"],
            file_sha256={n:m.sha(dest / n) for n in (*FILES,"observations.jsonl.gz")}))
    verify(dest, archived=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--originals", type=Path, help="check the previous private training file and checkpoint bytes, never eval")
    ap.add_argument("--archive", action="store_true", help="read the compressed receipt export")
    ap.add_argument("--export", type=Path, help="write checked scalar receipts into a new directory")
    a = ap.parse_args()
    result = verify(a.directory, a.originals, archived=a.archive)
    if a.export is not None:
        m.check(a.originals is not None and not a.archive, "export only after checking original files and raw journal")
        export(a.directory,a.export)
    print(json.dumps(result, sort_keys=True))
