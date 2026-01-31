from __future__ import annotations

import datetime as _dt
import contextlib
import json
import math
import pathlib
import random
import sys
from typing import Any

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

FORMAT_V1 = "st-char-lm-v1"
FORMAT_V2 = "st-char-lm-v2"
DEFAULT_UNK = "\uFFFD"

RUN_SCHEMA = "st.modelzoo.run.v1"


def _meta_path_for_weights(weights_path: pathlib.Path) -> pathlib.Path:
    name = weights_path.name
    if name.endswith(".json"):
        return weights_path.with_name(name[: -len(".json")] + ".meta.json")
    return weights_path.with_name(name + ".meta.json")


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

def _timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _default_run_dir() -> pathlib.Path:
    return _ROOT / "models" / "runs" / _timestamp_slug()


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _native_feature_flags() -> dict[str, bool]:
    try:
        info = st.build_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    features = info.get("features")
    if not isinstance(features, dict):
        return {}
    out: dict[str, bool] = {}
    for key, value in features.items():
        out[str(key)] = bool(value)
    return out


def _require_backend_available(backend: str) -> None:
    backend = str(backend).strip().lower()
    if backend in {"cpu"}:
        return

    flags = _native_feature_flags()
    if backend in {"wgpu", "webgpu", "auto"}:
        if flags.get("wgpu") or flags.get("wgpu-rt"):
            return
        raise RuntimeError(
            "backend=wgpu requested but this SpiralTorch build lacks the 'wgpu' feature. "
            "Rebuild the extension with `--features wgpu` (or `wgpu-rt`)."
        )
    if backend == "cuda":
        if flags.get("cuda"):
            return
        raise RuntimeError(
            "backend=cuda requested but this SpiralTorch build lacks the 'cuda' feature. "
            "Rebuild the extension with `--features cuda`."
        )
    if backend in {"hip", "rocm"}:
        if flags.get("hip") or flags.get("hip-real"):
            return
        raise RuntimeError(
            "backend=hip requested but this SpiralTorch build lacks the 'hip' feature. "
            "Rebuild the extension with `--features hip` (or `hip-real`)."
        )
    raise ValueError(f"unknown --backend: {backend} (expected cpu|wgpu|cuda|hip|auto)")


def _build_vocab(text: str, unk: str) -> tuple[str, list[str], dict[str, int]]:
    symbols = sorted({ch for ch in text if ch != unk})
    symbols = [unk, *symbols]
    index = {ch: i for i, ch in enumerate(symbols)}
    return unk, symbols, index


def _encode(text: str, index: dict[str, int]) -> list[int]:
    return [index.get(ch, 0) for ch in text]


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for idx, value in enumerate(values):
        if value == value and value > best_val:  # NaN-safe
            best_val = value
            best_idx = idx
    return best_idx


def _sample_topk(values: list[float], top_k: int, rng: random.Random) -> int:
    if not values:
        return 0
    if top_k <= 1:
        return _argmax(values)

    candidates: list[tuple[int, float]] = []
    for idx, value in enumerate(values):
        if value == value and value > 0.0:
            candidates.append((idx, float(value)))
        else:
            candidates.append((idx, 0.0))

    if 0 < top_k < len(candidates):
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        candidates = candidates[:top_k]

    total = sum(weight for _idx, weight in candidates)
    if total <= 0.0 or total != total:
        return _argmax(values)

    threshold = rng.random() * total
    for idx, weight in candidates:
        threshold -= weight
        if threshold <= 0.0:
            return idx
    return candidates[-1][0]

def _logits_from_probs(values: list[float], eps: float = 1e-9) -> list[float]:
    out: list[float] = []
    for value in values:
        v = float(value)
        if v == v and v > 0.0:
            out.append(math.log(v))
        else:
            out.append(math.log(eps))
    return out

def _apply_desire_offsets_to_probs(
    probs: list[float],
    desire_step: Any,
    *,
    max_offset: float = 8.0,
) -> list[float]:
    if not isinstance(desire_step, dict):
        return probs
    indices = desire_step.get("indices")
    offsets = desire_step.get("logit_offsets")
    if not isinstance(indices, list) or not isinstance(offsets, list):
        return probs
    if len(indices) != len(offsets):
        return probs

    out: list[float] = []
    for value in probs:
        v = float(value)
        out.append(v if v == v and v > 0.0 else 0.0)

    for idx, offset in zip(indices, offsets):
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(out):
            continue
        try:
            off = float(offset)
        except Exception:
            continue
        if off != off or off == float("inf") or off == float("-inf"):
            continue
        if max_offset > 0.0:
            off = max(-max_offset, min(max_offset, off))
        out[idx] *= math.exp(off)

    total = sum(out)
    if total <= 0.0 or total != total:
        return probs
    inv = 1.0 / total
    return [value * inv for value in out]


def _build_model(
    vocab_size: int,
    steps: int,
    hidden: int,
    curvature: float,
    temperature: float,
    *,
    embed_dim: int | None,
) -> st.nn.Sequential:
    model = st.nn.Sequential()
    if embed_dim is not None:
        model.add(st.nn.Embedding("embed", vocab_size, embed_dim))
        model.add(st.nn.SpiralRnn("char_rnn", embed_dim, hidden, steps))
    else:
        model.add(st.nn.SpiralRnn("char_rnn", vocab_size, hidden, steps))
    model.add(st.nn.Linear("head", hidden, vocab_size))
    model.add(st.nn.ZSpaceSoftmax(curvature, temperature))
    return model


def _build_random_batch(
    tokens: list[int],
    vocab_size: int,
    steps: int,
    batch: int,
    rng: random.Random,
    *,
    embed_dim: int | None,
) -> tuple[st.Tensor, st.Tensor]:
    max_start = len(tokens) - steps
    if max_start <= 0:
        raise ValueError(f"text too short for steps={steps}: len={len(tokens)}")

    if embed_dim is not None:
        x_data: list[float] = []
        y_data = [0.0] * (batch * vocab_size)
        for row in range(batch):
            start = rng.randrange(0, max_start)
            for t in range(steps):
                x_data.append(float(tokens[start + t]))
            target = tokens[start + steps]
            if 0 <= target < vocab_size:
                y_data[row * vocab_size + target] = 1.0
        x = st.Tensor(batch, steps, x_data)
        y = st.Tensor(batch, vocab_size, y_data)
        return x, y

    x_cols = vocab_size * steps
    x_data = [0.0] * (batch * x_cols)
    y_data = [0.0] * (batch * vocab_size)
    for row in range(batch):
        start = rng.randrange(0, max_start)
        for t in range(steps):
            idx = tokens[start + t]
            if 0 <= idx < vocab_size:
                x_data[row * x_cols + t * vocab_size + idx] = 1.0
        target = tokens[start + steps]
        if 0 <= target < vocab_size:
            y_data[row * vocab_size + target] = 1.0
    x = st.Tensor(batch, x_cols, x_data)
    y = st.Tensor(batch, vocab_size, y_data)
    return x, y


def _generate(
    model: st.nn.Sequential,
    symbols: list[str],
    index: dict[str, int],
    steps: int,
    prompt: str,
    gen_len: int,
    top_k: int,
    seed: int,
    *,
    embed_dim: int | None,
    desire: st.nn.DesirePipeline | None = None,
) -> str:
    rng = random.Random(seed)
    vocab_size = len(symbols)

    context = [index.get(ch, 0) for ch in prompt]
    if len(context) < steps:
        context = [0] * (steps - len(context)) + context
    elif len(context) > steps:
        context = context[-steps:]

    out = prompt
    for _ in range(gen_len):
        if embed_dim is not None:
            x = st.Tensor(1, steps, [float(i) for i in context])
        else:
            x_data = [0.0] * (vocab_size * steps)
            for t, idx in enumerate(context):
                if 0 <= idx < vocab_size:
                    x_data[t * vocab_size + idx] = 1.0
            x = st.Tensor(1, vocab_size * steps, x_data)

        probs = model.forward(x)
        row = probs.tolist()[0]
        if desire is not None:
            step = desire.step(_logits_from_probs([float(v) for v in row]), context[-1])
            row = _apply_desire_offsets_to_probs([float(v) for v in row], step)
        else:
            row = [float(v) for v in row]
        next_idx = _sample_topk(row, top_k, rng)
        out += symbols[next_idx]
        context = context[1:] + [next_idx]
    return out


def _load_meta(meta_path: pathlib.Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_meta(meta_path: pathlib.Path, meta: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text.txt> "
            "[--load weights.json] [--save weights.json] [--steps N] [--embed-dim N] [--hidden N] "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] "
            "[--gen N] [--topk N] [--seed N] [--prompt STR] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--desire] [--desire-concepts N] [--desire-prime N] [--desire-blend F] [--desire-drift-gain F] "
            "[--run-dir PATH]"
        )
        return

    args = list(sys.argv[1:])
    text_path = pathlib.Path(args.pop(0))

    load_weights: pathlib.Path | None = None
    save_weights: pathlib.Path | None = None
    run_dir: pathlib.Path | None = None
    steps = 32
    embed_dim = 32
    hidden = 64
    epochs = 6
    batches_per_epoch = 24
    batch = 8
    lr = 2e-2
    curvature = -1.0
    temperature = 1.0
    gen_len = 200
    top_k = 32
    seed = 42
    prompt: str | None = None
    backend = "cpu"
    events_path: pathlib.Path | None = None
    events_types = [
        "EpochStart",
        "EpochEnd",
        "RoundtablePlanned",
        "TrainerStep",
        "TrainerPhase",
    ]
    atlas = False
    atlas_bound = 512
    atlas_district = "Training"
    desire = False
    desire_concepts = 3
    desire_prime = 16
    desire_blend = 0.35
    desire_drift_gain = 0.35

    it = iter(args)
    for flag in it:
        if flag == "--load":
            load_weights = pathlib.Path(next(it))
        elif flag == "--save":
            save_weights = pathlib.Path(next(it))
        elif flag == "--steps":
            steps = int(next(it))
        elif flag == "--embed-dim":
            embed_dim = int(next(it))
        elif flag == "--hidden":
            hidden = int(next(it))
        elif flag == "--epochs":
            epochs = int(next(it))
        elif flag == "--batches":
            batches_per_epoch = int(next(it))
        elif flag == "--batch":
            batch = int(next(it))
        elif flag == "--lr":
            lr = float(next(it))
        elif flag == "--curvature":
            curvature = float(next(it))
        elif flag == "--temperature":
            temperature = float(next(it))
        elif flag == "--gen":
            gen_len = int(next(it))
        elif flag == "--topk":
            top_k = int(next(it))
        elif flag == "--seed":
            seed = int(next(it))
        elif flag == "--prompt":
            prompt = str(next(it))
        elif flag == "--backend":
            backend = str(next(it)).strip().lower()
        elif flag == "--events":
            events_path = pathlib.Path(str(next(it)))
        elif flag == "--events-types":
            raw = str(next(it))
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if not parts:
                raise ValueError("empty --events-types")
            events_types = parts
        elif flag == "--atlas":
            atlas = True
        elif flag == "--atlas-bound":
            atlas_bound = int(next(it))
        elif flag == "--atlas-district":
            atlas_district = str(next(it))
        elif flag == "--desire":
            desire = True
        elif flag == "--desire-concepts":
            desire_concepts = int(next(it))
        elif flag == "--desire-prime":
            desire_prime = int(next(it))
        elif flag == "--desire-blend":
            desire_blend = float(next(it))
        elif flag == "--desire-drift-gain":
            desire_drift_gain = float(next(it))
        elif flag == "--run-dir":
            run_dir = pathlib.Path(str(next(it)))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if desire_concepts <= 0:
        raise ValueError("--desire-concepts must be >= 1")
    if desire_prime < 0:
        raise ValueError("--desire-prime must be >= 0")

    _require_backend_available(backend)

    text = _read_text(text_path)
    if not text:
        raise ValueError("empty text")

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    if load_weights is not None:
        meta_path = _meta_path_for_weights(load_weights)
        meta = _load_meta(meta_path)
        fmt = str(meta.get("format", ""))
        if fmt not in {FORMAT_V1, FORMAT_V2}:
            raise ValueError(f"unexpected meta format: {fmt}")
        steps = int(meta["steps"])
        hidden = int(meta["hidden"])
        curvature = float(meta["curvature"])
        temperature = float(meta["temperature"])
        unk = str(meta["unk"])
        symbols = [str(ch) for ch in meta["symbols"]]
        embed_dim_meta = meta.get("embed_dim")
        embed_dim_opt = int(embed_dim_meta) if embed_dim_meta is not None else None
        if fmt == FORMAT_V2 and embed_dim_opt is None:
            raise ValueError("meta format st-char-lm-v2 requires embed_dim")
        index = {ch: i for i, ch in enumerate(symbols)}
        vocab_size = len(symbols)
        model = _build_model(
            vocab_size,
            steps,
            hidden,
            curvature,
            temperature,
            embed_dim=embed_dim_opt,
        )
        st.nn.load(str(load_weights), model)
        embed_dim_runtime = embed_dim_opt
    else:
        unk, symbols, index = _build_vocab(text, DEFAULT_UNK)
        vocab_size = len(symbols)
        model = _build_model(
            vocab_size,
            steps,
            hidden,
            curvature,
            temperature,
            embed_dim=embed_dim,
        )
        embed_dim_runtime = embed_dim

    tokens = _encode(text, index)
    if len(tokens) <= steps:
        raise ValueError(f"text too short for steps={steps}: len={len(tokens)}")

    if prompt is None:
        prompt = "".join(list(text)[:steps])

    mode = (
        f"embedding({embed_dim_runtime})" if embed_dim_runtime is not None else "one_hot"
    )
    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "llm_char_finetune",
        "text_path": str(text_path),
        "format": FORMAT_V2 if embed_dim_runtime is not None else FORMAT_V1,
        "steps": steps,
        "embed_dim": embed_dim_runtime,
        "hidden": hidden,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "lr": lr,
        "curvature": curvature,
        "temperature": temperature,
        "backend": backend,
        "gen_len": gen_len,
        "top_k": top_k,
        "seed": seed,
        "prompt": prompt,
        "vocab_size": vocab_size,
        "symbols_count": len(symbols),
        "mode": mode,
        "events_path": str(events_path) if events_path is not None else None,
        "events_types": events_types,
        "atlas": atlas,
        "atlas_bound": atlas_bound if atlas else None,
        "atlas_district": atlas_district if atlas else None,
        "desire": desire,
        "desire_concepts": desire_concepts if desire else None,
        "desire_prime": desire_prime if desire else None,
        "desire_blend": desire_blend if desire else None,
        "desire_drift_gain": desire_drift_gain if desire else None,
        "weights_loaded_from": str(load_weights) if load_weights is not None else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=curvature,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    desire_pipeline: st.nn.DesirePipeline | None = None
    if desire:
        desire_bundle = st.nn.DesireTelemetryBundle(
            blend=desire_blend,
            drift_gain=desire_drift_gain,
        )
        trainer.enable_desire_telemetry(desire_bundle)
        desire_pipeline = st.nn.DesirePipeline(
            vocab_size,
            concepts=desire_concepts,
            bundle=desire_bundle,
        )
    loss = st.nn.CategoricalCrossEntropy()

    print(
        f"mode={mode} vocab={vocab_size} steps={steps} hidden={hidden} epochs={epochs} "
        f"batch={batch} lr={lr:.3e} curvature={curvature} temp={temperature} backend={backend} run_dir={run_dir}"
    )

    metrics_path = run_dir / "metrics.jsonl"
    record_ctx = (
        st.plugin.record(str(events_path), event_types=events_types)
        if events_path is not None
        else contextlib.nullcontext()
    )
    with record_ctx:
        schedule = trainer.roundtable(
            batch,
            vocab_size,
            st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
        )

        for epoch in range(epochs):
            rng = random.Random(seed + epoch * 10_000)
            if desire_pipeline is not None and desire_prime > 0:
                _generate(
                    model,
                    symbols,
                    index,
                    steps,
                    prompt,
                    desire_prime,
                    top_k,
                    seed + 123 + epoch,
                    embed_dim=embed_dim_runtime,
                    desire=desire_pipeline,
                )
            batches = []
            for _ in range(batches_per_epoch):
                batches.append(
                    _build_random_batch(
                        tokens,
                        vocab_size,
                        steps,
                        batch,
                        rng,
                        embed_dim=embed_dim_runtime,
                    )
                )
            stats = trainer.train_epoch(model, loss, batches, schedule)
            avg_loss = float(stats.average_loss)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"epoch": epoch, "batches": int(stats.batches), "average_loss": avg_loss},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            sample = _generate(
                model,
                symbols,
                index,
                steps,
                prompt,
                gen_len,
                top_k,
                seed + 999 + epoch,
                embed_dim=embed_dim_runtime,
                desire=desire_pipeline,
            )
            (samples_dir / f"epoch_{epoch:03d}.txt").write_text(sample, encoding="utf-8")
            print(f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f}")

    if atlas and events_path is not None:
        try:
            from spiraltorch.zspace_atlas import trainer_events_to_atlas_route

            route = trainer_events_to_atlas_route(
                events_path,
                district=atlas_district,
                bound=atlas_bound,
            )
            _write_json(run_dir / "atlas_summary.json", route.summary())
        except Exception as exc:
            _write_json(run_dir / "atlas_summary.json", {"error": str(exc)})

    weights_path = run_dir / "weights.json"
    st.nn.save(str(weights_path), model)

    meta = {
        "format": FORMAT_V2 if embed_dim_runtime is not None else FORMAT_V1,
        "steps": steps,
        "hidden": hidden,
        "curvature": curvature,
        "temperature": temperature,
        "embed_dim": embed_dim_runtime,
        "unk": symbols[0],
        "symbols": symbols,
    }
    _save_meta(_meta_path_for_weights(weights_path), meta)

    if save_weights is not None:
        save_weights = pathlib.Path(save_weights)
        save_weights.parent.mkdir(parents=True, exist_ok=True)
        st.nn.save(str(save_weights), model)
        _save_meta(_meta_path_for_weights(save_weights), meta)

    if epochs > 0:
        sample = (samples_dir / f"epoch_{epochs - 1:03d}.txt").read_text(encoding="utf-8")
    else:
        sample = _generate(
            model,
            symbols,
            index,
            steps,
            prompt,
            gen_len,
            top_k,
            seed + 999,
            embed_dim=embed_dim_runtime,
            desire=desire_pipeline,
        )
        (samples_dir / "init.txt").write_text(sample, encoding="utf-8")
    print("--- sample (prompt + gen) ---")
    print(sample)


if __name__ == "__main__":
    main()
