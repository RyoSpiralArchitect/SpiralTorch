"""Model-free definitions and read-back of a fixed native GPT-2 endpoint diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "2026-11-03-native-wave-affine-extraction" / "receipts"
ORDERS = ("order-109", "order-113", "order-127")
ARMS = ("affine", "wave_fixed", "wave_learned")
TRAIN = "165a5779fce5d70609c91335c9e9475ed0d22676c71bb813e19021087fc0d28d"
BASE = "sha256:0c552e7a3d7fae610edd7a22831f91e0fa718b212d38d15a07ff875e3979fb3c"
MODEL = {"config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
         "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"}
BOUNDARIES = ("0d27743094de30e43383fbaefbcea46fb37b2f9cb7de1039c6f1cba639e5d5d1",
              "14d673df33058f25f624707798f1aad7977dae344b0ebd3d847408631a6f874b",
              "344cd895a58ca76abc91c1296a7c134c4172b053898fb0589554a149a7024c6d")
SITE = "transformer.h.11.attn"
N, TARGETS, ROWS, WIDTH = 96, 148, 2200, 768


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, separators=(",", ":"))
        f.write("\n")


def boundaries():
    out = {}
    for order, pin in zip(ORDERS, BOUNDARIES, strict=True):
        f = PREVIOUS / order / "checkpoints.json"
        check(sha(f) == pin, "changed checkpoint boundary: " + order)
        for a, e in zip(ARMS, read(f), strict=True):
            check(e["arm"] == a and e["update"] == 96, "checkpoint arm or cursor")
            out[order, a] = e
    return out


def ratio(num, den):
    return math.sqrt(num / den) if den else None


def reduce_sites(sites):
    n = sum(s["elements"] for s in sites)
    def ss(field, back=False):
        return math.fsum(s["elements"] * (s["pullback"] if back else s)[field] ** 2 for s in sites)
    x, res, up, dx, corr = ss("input_rms"), ss("residual_rms"), ss("upstream_rms", True), ss("input_adjoint_rms", True), ss("correction_rms", True)
    grads = {k: math.sqrt(math.fsum(s["parameter_gradient_l2"][k] ** 2 for s in sites) / len(sites))
             for k in sorted(sites[0]["parameter_gradient_l2"])}
    report = dict(elements=n, input_rms=ratio(x,n), residual_rms=ratio(res,n), relative_residual_rms=ratio(res,x),
                  upstream_rms=ratio(up,n), input_adjoint_rms=ratio(dx,n), correction_rms=ratio(corr,n),
                  relative_correction_rms=ratio(corr,up), document_rms_parameter_l2=grads, wave=None)
    waves = [s["wave"] for s in sites if s["wave"] is not None]
    if waves:
        rows = sum(w["rows"] for w in waves)
        check(len(waves) == len(sites) and all(w["gate_outside_values"] == waves[0]["gate_outside_values"]
              for w in waves), "partial or inconsistent Wave measurement")
        report["wave"] = dict(rows=rows, gate_outside=waves[0]["gate_outside_values"],
                affine_outside=sum(w["affine_outside_values"] for w in waves),
                affine_nonfinite=sum(w["affine_nonfinite_values"] for w in waves),
                max_dimensionless_norm=max(w["dimensionless_norm_max"] for w in waves),
                min_radial_gain=min(w["relative_radial_gain_min"] for w in waves),
                min_tangential_gain=min(w["relative_tangential_gain_min"] for w in waves))
        for key in ("relative_radial_gain_mean", "relative_tangential_gain_mean", "dimensionless_norm_mean", "affine_abs_slope_mean"):
            report["wave"][key] = math.fsum(w["rows"] * w[key] for w in waves) / rows
    return report


def validate_row(row, order, arm, index, e, plan):
    key, length, targets = plan
    check((row["order"], row["arm"], row["index"], row["key"]) == (order, arm, index, key), "unpaired window")
    o = row["observation"]
    check(o["schema"] == "spiraltorch.native_causal_lm_adaptation_observation.v1" and o["base_model_identity"] == BASE
          and [o["lora_artifact_id"], o["geometry_artifact_id"]] == e["owners"] and o["readout_accumulator"] == "f32"
          and o["mode"] == "joint" and o["window_shape"] == [1, length] and o["target_tokens"] == targets
          and math.isfinite(o["loss"]) and set(o["sites"]) == {SITE}, "observation provenance/shape")
    s = o["sites"][SITE]
    d = s["pullback"]
    check(s["elements"] == (length - 1) * WIDTH and s["input_rms"] > 0 and d["upstream_rms"] > 0
          and set(s["parameter_gradient_l2"]) == ({"bias", "gate", "log_radius"} if arm == ARMS[2] else {"bias", "gate"})
          and all(math.isfinite(v) and v >= 0 for v in (s["input_rms"], s["residual_rms"], *d.values(),
                                                        *s["parameter_gradient_l2"].values()))
          and math.isclose(s["relative_residual_rms"], s["residual_rms"] / s["input_rms"], rel_tol=1e-12)
          and math.isclose(d["relative_correction_rms"], d["correction_rms"] / d["upstream_rms"], rel_tol=1e-12),
          "residual/pullback denominator or parameter roster")
    if arm == "wave_learned":
        # The derivative is serialized as f32, while its norm was accumulated in f64.
        radius_grad, = struct.unpack("<f", struct.pack("<f", s["log_radius_gradient"]))
        check(abs(radius_grad) == s["parameter_gradient_l2"]["log_radius"], "radius gradient")
    else:
        check(s["log_radius_gradient"] is None, "spurious radius gradient")
    w = s["wave"]
    if arm == "affine":
        check(w is None, "spurious projection")
    else:
        # Checkpoint JSON uses the shortest f32 string; the f64 exp is not that same serialization.
        raw_radius, = struct.unpack("<f", struct.pack("<f", e["log_radius"]))
        check(w["schema"] == "spiraltorch.wave_gate_conditioning.v2" and w["rows"] == length-1 and w["features"] == WIDTH
              and w["log_radius"] == e["log_radius"] and math.isclose(w["projection_radius"], math.exp(raw_radius), rel_tol=1e-14)
              and 0 <= w["gate_outside_values"] <= WIDTH and 0 <= w["affine_outside_values"] <= s["elements"]
              and w["affine_nonfinite_values"] == 0 and 0 <= w["dimensionless_norm_mean"] <= w["dimensionless_norm_max"] + 1e-12
              and all(0 <= w["relative_" + g + "_gain_min"] <= w["relative_" + g + "_gain_mean"] + 1e-12 <= 1.00000000001
                      for g in ("radial","tangential")), "Wave conditioning")
    return s


def summarize(lines, plan):
    check(len(plan) == N and len({k for k, _, _ in plan}) == N and sum(t for _, _, t in plan) == TARGETS
          and all(0 < t < n <= 64 for _, n, t in plan) and sum(n-1 for _,n,_ in plan) == ROWS, "training plan")
    entries, results = boundaries(), {}
    for order in ORDERS:
        cohort = {}
        for a in ARMS:
            sites = []
            for i, row_plan in enumerate(plan):
                line = next(lines, None)
                check(line is not None and len(line) <= 8192, "incomplete/oversized journal")
                sites.append(validate_row(json.loads(line), order, a, i, entries[order,a], row_plan))
            cohort[a] = reduce_sites(sites)
            check(cohort[a]["elements"] == WIDTH * ROWS, "wrong input denominator")
        results[order] = cohort
    check(next(lines, None) is None, "unexpected additional execution")
    return results
