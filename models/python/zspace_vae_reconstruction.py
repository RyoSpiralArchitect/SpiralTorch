from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import sys
import time
from typing import Any

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

FORMAT = "st-zspace-vae-recon-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"


def _timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_run_dir() -> pathlib.Path:
    return _ROOT / "models" / "runs" / _timestamp_slug()


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_vae_reconstruction.py "
            "[--input-dim N] [--latent-dim N] [--seed N] [--steps N] [--lr F] "
            "[--input \"0.35,-0.12,...\"] [--exponents \"1.0,0.5,...\"] "
            "[--events PATH] [--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--run-dir PATH]"
        )
        return

    run_dir: pathlib.Path | None = None
    events_path: pathlib.Path | None = None
    atlas = False
    atlas_bound = 512
    atlas_district = "Coherence"

    input_dim = 8
    latent_dim = 3
    seed = 42
    steps = 12
    lr = 1e-2

    input_vec = [0.35, -0.12, 0.77, 0.05, -0.28, 0.44, 0.10, -0.06]
    exponents = [1.0, 0.5, 2.0, 1.25, 0.75, 1.5, 1.0, 0.9]

    args = list(sys.argv[1:])
    it = iter(args)
    for flag in it:
        if flag == "--run-dir":
            run_dir = pathlib.Path(next(it))
        elif flag == "--events":
            events_path = pathlib.Path(next(it))
        elif flag == "--atlas":
            atlas = True
        elif flag == "--atlas-bound":
            atlas_bound = int(next(it))
        elif flag == "--atlas-district":
            atlas_district = str(next(it))
        elif flag == "--input-dim":
            input_dim = int(next(it))
        elif flag == "--latent-dim":
            latent_dim = int(next(it))
        elif flag == "--seed":
            seed = int(next(it))
        elif flag == "--steps":
            steps = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--input":
            input_vec = _parse_floats(str(next(it)))
        elif flag == "--exponents":
            exponents = _parse_floats(str(next(it)))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    if input_dim <= 0:
        raise ValueError("--input-dim must be > 0")
    if latent_dim <= 0:
        raise ValueError("--latent-dim must be > 0")
    if steps < 0:
        raise ValueError("--steps must be >= 0")
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("--lr must be positive and finite")

    if len(input_vec) != input_dim:
        raise ValueError(f"--input length mismatch (expected {input_dim}, got {len(input_vec)})")
    if len(exponents) != input_dim:
        raise ValueError(
            f"--exponents length mismatch (expected {input_dim}, got {len(exponents)})"
        )

    basis = st.nn.MellinBasis([float(v) for v in exponents])
    projected = basis.project([float(v) for v in input_vec])

    vae = st.nn.ZSpaceVae(input_dim, latent_dim, seed=seed)

    run_meta = {
        "schema": RUN_SCHEMA,
        "arch": "zspace_vae_reconstruction",
        "format": FORMAT,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "events_path": str(events_path) if events_path is not None else None,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
        "input": input_vec,
        "exponents": exponents,
        "projected": projected,
    }
    _write_json(run_dir / "run.json", run_meta)

    print(f"input_dim={input_dim} latent_dim={latent_dim} steps={steps} lr={lr:.3e} run_dir={run_dir}")
    print(f"input_norm={_l2_norm(input_vec):.6f} projected_norm={_l2_norm(projected):.6f}")

    start_ts = time.time()
    last_recon: float | None = None
    for step in range(steps):
        state = vae.forward(projected)
        stats = state.stats
        recon = float(stats.recon_loss)
        kl = float(stats.kl_loss)
        elbo = float(stats.evidence_lower_bound)
        delta = recon - last_recon if last_recon is not None else None
        delta_txt = f"{delta:+.6f}" if delta is not None else "—"
        print(
            f"step={step:02d} recon_loss={recon:.6f} kl_loss={kl:.6f} elbo={elbo:.6f} delta={delta_txt}"
        )
        if events_path is not None:
            payload = {
                "event_type": "TrainerStep",
                "ts": start_ts + float(step) * 0.01,
                "payload": {
                    "epoch": 0,
                    "step": step,
                    "metrics": {
                        "extra": {
                            "recon_loss": recon,
                            "kl_loss": kl,
                            "elbo": elbo,
                        }
                    },
                },
            }
            _append_jsonl(events_path, payload)
        last_recon = recon
        vae.refine_decoder(state, lr)

    final_state = vae.forward(projected)
    final_recon = [float(v) for v in final_state.reconstruction]
    final_error_norm = _l2_norm([final_recon[i] - projected[i] for i in range(len(projected))])
    _write_json(
        run_dir / "final.json",
        {
            "projected": projected,
            "reconstruction": final_recon,
            "error_norm": final_error_norm,
            "stats": {
                "recon_loss": float(final_state.stats.recon_loss),
                "kl_loss": float(final_state.stats.kl_loss),
                "elbo": float(final_state.stats.evidence_lower_bound),
            },
        },
    )
    print(f"final_error_norm={final_error_norm:.6f}")

    if atlas and events_path is not None:
        try:
            route = st.trainer_events_to_atlas_route(
                events_path,
                district=atlas_district,
                bound=atlas_bound,
            )
            _write_json(run_dir / "atlas_summary.json", route.summary())
        except Exception as exc:
            _write_json(run_dir / "atlas_summary.json", {"error": str(exc)})


if __name__ == "__main__":
    main()
