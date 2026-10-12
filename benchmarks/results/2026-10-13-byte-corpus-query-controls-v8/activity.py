"""Post-screen CPU diagnostic of final routers on declared training windows.

No updates, fitting or held-out access; these are Torch reconstructions from
retained weights, not measured resident kernel activations.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "report", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("study", a.root / "tools/byte_corpus_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    raw, report_raw = a.request.read_bytes(), a.report.read_bytes()
    r, result = study.decode_json(raw), study.decode_json(report_raw)
    if study.request_version(r) != 8 or result["request_sha256"] != study.digest(raw):
        raise ValueError("wrong request identity")
    if result["schema"] != "spiraltorch.byte_corpus.result.v8" or len(result["cases"]) != len(r["cases"]):
        raise ValueError("incomplete result")
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(1)
    wave = study.load_reference("generate_causal_zspace_wave_torch_fixture").wave
    cfg = r["config"]
    b, t, h = (cfg[k] for k in ("batch", "steps", "heads"))
    records = []
    for case, row in zip(r["cases"], result["cases"]):
        if (row["name"] != case["name"] or row.get("mixture") != case.get("mixture")
                or [s["revision"] for s in row["training"]] != list(range(1,len(r["train_batches"])+1))
                or len(row["final_parameters"]) != len(case["parameters"])):
            raise ValueError("incomplete case identity or updates")
        for d, values in zip(case["parameters"], row["final_parameters"]):
            if len(values) != len(d["values"]):
                raise ValueError("parameter coverage")
            study.close(values, values, d["name"])
        q = case.get("mixture", {}).get("query_projection")
        if q is None:
            continue
        values = {d["name"]:torch.tensor(v,dtype=torch.float32).reshape(d["shape"])
                  for d,v in zip(case["parameters"],row["final_parameters"])}
        w, bias = (values["geometry.query_projection.0."+k] for k in ("weight","bias"))
        if bias.count_nonzero() or (q["weight_update"] == "frozen" and w.count_nonzero()):
            raise ValueError("frozen query parameters changed")
        k = len(case["mixture"]["metrics"])
        logits = values["geometry.mixture_logits.0"]
        masses, contrasts = [], []
        with torch.no_grad():
            for i in r["bias_calibration"]["train_batch_indices"]:
                ids = torch.tensor([r["train_documents"][d][s:s+t] for d,s in r["train_batches"][i]])
                x = f.embedding(ids,values["token_embedding"]) + values["position_embedding"]
                coords,_ = wave(x @ values["geometry.projection.weight"] + values["geometry.projection.bias"],
                                values["geometry.raw_decay"],values["geometry.raw_phase"],
                                torch.zeros((b,cfg["geometry_cols"])),cfg["curvature"])
                residual = (coords @ w + bias).reshape(b,t,h,k)
                masses.append((residual+logits).softmax(-1).double().reshape(-1,h,k))
                contrasts.append(residual.double().reshape(-1,h,k))
        masses, contrasts = torch.cat(masses), torch.cat(contrasts)
        shared = logits.softmax(-1).double()
        records.append(dict(name=case["name"],seed=case["seed"],mixture=case["mixture"],
            positions=int(masses.shape[0]),weight_l2=w.double().norm().item(),
            max_mass_shift_from_own_shared=(masses-shared).abs().max().item(),
            max_position_mass_std=masses.std(0,correction=0).max().item(),
            centered_context_logit_rms=(contrasts-contrasts.mean(-1,keepdim=True)).square().mean().sqrt().item()))
    study.write_new(a.output,dict(schema="spiraltorch.byte_corpus.query_activity.v1",
        request_sha256=study.digest(raw),report_sha256=hashlib.sha256(report_raw).hexdigest(),
        source_engine=result["engine"],reconstruction="independent_cpu_float32_torch",
        train_batch_indices=r["bias_calibration"]["train_batch_indices"],cases=records,
        no_updates_consumed=True,heldout_accessed=False,qualification=False,
        scope="Post-screen descriptive reconstruction, not direct GPU activations or a quality gate."))


if __name__ == "__main__":
    main()
