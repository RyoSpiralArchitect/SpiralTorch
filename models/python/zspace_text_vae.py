from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

FORMAT = "st-zspace-text-vae-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"

_TEXT_EXTS = {".txt"}


def _timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_run_dir() -> pathlib.Path:
    return _ROOT / "models" / "runs" / _timestamp_slug()


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _collect_text_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()

    for raw in paths:
        path = raw.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.suffix.lower() not in _TEXT_EXTS:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)

    return files


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pick_window(text: str, window_chars: int, rng: random.Random) -> str:
    if window_chars <= 0:
        return ""
    if len(text) <= window_chars:
        return text
    start = rng.randrange(0, len(text) - window_chars)
    return text[start : start + window_chars]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae.py <text_or_dir> [<text_or_dir> ...] "
            "[--window-chars N] [--latent-dim N] [--epochs N] [--batches N] [--lr F] "
            "[--curvature F] [--temperature F] [--seed N] "
            "[--mellin none|constant|ramp] [--mellin-exponent F] [--mellin-start F] [--mellin-end F] "
            "[--load PATH] [--save PATH] [--checkpoint-every N] "
            "[--events PATH] [--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--run-dir PATH]"
        )
        return

    args = list(sys.argv[1:])
    data_args: list[str] = []
    while args and not str(args[0]).startswith("--"):
        data_args.append(str(args.pop(0)))
    if not data_args:
        raise ValueError("expected at least one <text.txt|dir> before flags")
    data_paths = [pathlib.Path(p) for p in data_args]

    run_dir: pathlib.Path | None = None
    events_path: pathlib.Path | None = None
    atlas = False
    atlas_bound = 512
    atlas_district = "Coherence"

    load_path: pathlib.Path | None = None
    save_path: pathlib.Path | None = None
    checkpoint_every = 0

    window_chars = 256
    latent_dim = 32
    epochs = 10
    batches_per_epoch = 256
    lr = 1e-2
    curvature = -1.0
    temperature = 1.0
    seed = 42

    mellin_mode = "none"
    mellin_exponent = 1.0
    mellin_start = 0.8
    mellin_end = 1.2

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
        elif flag == "--load":
            load_path = pathlib.Path(next(it))
        elif flag == "--save":
            save_path = pathlib.Path(next(it))
        elif flag == "--checkpoint-every":
            checkpoint_every = int(next(it))
        elif flag == "--window-chars":
            window_chars = int(next(it))
        elif flag == "--latent-dim":
            latent_dim = int(next(it))
        elif flag == "--epochs":
            epochs = int(next(it))
        elif flag == "--batches":
            batches_per_epoch = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--curvature":
            curvature = float(next(it))
        elif flag == "--temperature":
            temperature = float(next(it))
        elif flag == "--seed":
            seed = int(next(it))
        elif flag == "--mellin":
            mellin_mode = str(next(it)).strip().lower()
        elif flag == "--mellin-exponent":
            mellin_exponent = float(next(it))
        elif flag == "--mellin-start":
            mellin_start = float(next(it))
        elif flag == "--mellin-end":
            mellin_end = float(next(it))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if window_chars <= 0:
        raise ValueError("--window-chars must be > 0")
    if latent_dim <= 0:
        raise ValueError("--latent-dim must be > 0")
    if epochs < 0:
        raise ValueError("--epochs must be >= 0")
    if batches_per_epoch <= 0:
        raise ValueError("--batches must be > 0")
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("--lr must be positive and finite")
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if mellin_mode not in {"none", "constant", "ramp"}:
        raise ValueError("--mellin must be one of: none|constant|ramp")

    data_files = _collect_text_files(data_paths)
    if not data_files:
        raise ValueError("no .txt files found in inputs")

    text_parts = [_read_text(path) for path in data_files]
    text = "\n\n".join(part for part in text_parts if part)
    if not text:
        raise ValueError("empty text")

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    (run_dir / "data_files.txt").write_text(
        "\n".join(str(path) for path in data_files) + "\n",
        encoding="utf-8",
    )

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    if save_path is None:
        save_path = run_dir / "best_weights.bin"
    checkpoint_suffix = save_path.suffix if save_path.suffix else ".bin"

    if load_path is not None:
        model = st.nn.ZSpaceTextVae.load(str(load_path))
        window_chars = int(model.window_chars)
        latent_dim = int(model.latent_dim)
        curvature = float(model.curvature)
        temperature = float(model.temperature)
    else:
        model = st.nn.ZSpaceTextVae(
            window_chars,
            latent_dim,
            curvature=curvature,
            temperature=temperature,
            seed=seed,
        )

    basis: st.nn.MellinBasis | None = None
    if mellin_mode == "constant":
        basis = st.nn.MellinBasis.constant(model.input_dim, mellin_exponent)
    elif mellin_mode == "ramp":
        basis = st.nn.MellinBasis.ramp(model.input_dim, mellin_start, mellin_end)

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "zspace_text_vae",
        "format": FORMAT,
        "data_paths": [str(path) for path in data_paths],
        "data_file_count": len(data_files),
        "data_files_manifest": str(run_dir / "data_files.txt"),
        "window_chars": window_chars,
        "input_dim": int(model.input_dim),
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "lr": lr,
        "curvature": curvature,
        "temperature": temperature,
        "seed": seed,
        "mellin": {
            "mode": mellin_mode,
            "exponent": mellin_exponent if mellin_mode == "constant" else None,
            "start": mellin_start if mellin_mode == "ramp" else None,
            "end": mellin_end if mellin_mode == "ramp" else None,
        },
        "checkpoint": {
            "load_path": str(load_path) if load_path is not None else None,
            "save_path": str(save_path) if save_path is not None else None,
            "checkpoint_every": checkpoint_every,
        },
        "events_path": str(events_path) if events_path is not None else None,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    print(
        f"files={len(data_files)} chars={len(text)} window_chars={window_chars} input_dim={model.input_dim} latent_dim={latent_dim} "
        f"epochs={epochs} batches={batches_per_epoch} lr={lr:.3e} curvature={curvature} temp={temperature} mellin={mellin_mode} run_dir={run_dir}"
    )

    metrics_path = run_dir / "metrics.jsonl"
    start_ts = time.time()

    best_metric = float("inf")
    best_epoch: int | None = None

    for epoch in range(max(0, epochs)):
        rng = random.Random(seed + epoch * 10_000)
        recon_sum = 0.0
        kl_sum = 0.0
        elbo_sum = 0.0

        for step in range(batches_per_epoch):
            window = _pick_window(text, window_chars, rng)
            if basis is None:
                state = model.forward_text(window)
            else:
                state = model.forward_text_with_mellin(window, basis)
            stats = state.stats
            recon = float(stats.recon_loss)
            kl = float(stats.kl_loss)
            elbo = float(stats.evidence_lower_bound)
            recon_sum += recon
            kl_sum += kl
            elbo_sum += elbo

            if events_path is not None:
                _append_jsonl(
                    events_path,
                    {
                        "event_type": "TrainerStep",
                        "ts": start_ts + epoch * 1.0 + float(step) * 0.001,
                        "payload": {
                            "epoch": epoch,
                            "step": step,
                            "metrics": {
                                "extra": {
                                    "recon_loss": recon,
                                    "kl_loss": kl,
                                    "elbo": elbo,
                                }
                            },
                        },
                    },
                )
            model.refine_decoder(state, lr)

        denom = float(batches_per_epoch)
        avg_recon = recon_sum / denom
        avg_kl = kl_sum / denom
        avg_elbo = elbo_sum / denom
        print(
            f"epoch[{epoch}] avg_recon_loss={avg_recon:.6f} avg_kl_loss={avg_kl:.6f} avg_elbo={avg_elbo:.6f}",
            flush=True,
        )
        _append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "batches": batches_per_epoch,
                "avg_recon_loss": avg_recon,
                "avg_kl_loss": avg_kl,
                "avg_elbo": avg_elbo,
            },
        )

        tracked = avg_recon + avg_kl
        if tracked < best_metric:
            best_metric = tracked
            best_epoch = epoch
            print(f"epoch[{epoch}] saving {save_path.name}...", flush=True)
            model.save(str(save_path))

        if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0):
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}{checkpoint_suffix}"
            model.save(str(ckpt_path))

    if best_epoch is None:
        # Save the initial model if epochs=0.
        model.save(str(save_path))

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
