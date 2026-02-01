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

FORMAT = "st-char-lm-wave-rnn-mixer-v1"
DEFAULT_UNK = "\uFFFD"

RUN_SCHEMA = "st.modelzoo.run.v1"

_PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "steps": 32,
        "embed_dim": 32,
        "hidden": 64,
        "kernel": 3,
        "stride": 1,
        "mixer_depth": 1,
        "epochs": 6,
        "batches_per_epoch": 24,
        "batch": 8,
        "lr": 2e-2,
        "top_k": 32,
        "temperature": 1.0,
        "weights_format": "json",
        "checkpoint_every": 0,
        "val_split": 0.1,
        "val_batches": 0,
    },
    "small": {
        "steps": 64,
        "embed_dim": 64,
        "hidden": 128,
        "kernel": 3,
        "stride": 1,
        "mixer_depth": 2,
        "epochs": 8,
        "batches_per_epoch": 32,
        "batch": 8,
        "lr": 1.5e-2,
        "top_k": 32,
        "temperature": 1.0,
        "weights_format": "auto",
        "checkpoint_every": 0,
        "val_split": 0.1,
        "val_batches": 0,
    },
    "base": {
        "steps": 128,
        "embed_dim": 128,
        "hidden": 256,
        "kernel": 5,
        "stride": 1,
        "mixer_depth": 4,
        "epochs": 10,
        "batches_per_epoch": 48,
        "batch": 8,
        "lr": 1.0e-2,
        "top_k": 48,
        "temperature": 1.0,
        "weights_format": "auto",
        "checkpoint_every": 1,
        "val_split": 0.1,
        "val_batches": 4,
    },
    "large": {
        "steps": 256,
        "embed_dim": 256,
        "hidden": 512,
        "kernel": 7,
        "stride": 1,
        "mixer_depth": 6,
        "epochs": 12,
        "batches_per_epoch": 64,
        "batch": 4,
        "lr": 8.0e-3,
        "top_k": 64,
        "temperature": 1.0,
        "weights_format": "bincode",
        "checkpoint_every": 1,
        "val_split": 0.1,
        "val_batches": 4,
    },
}


def _meta_path_for_weights(weights_path: pathlib.Path) -> pathlib.Path:
    name = weights_path.name
    if name.endswith(".json"):
        return weights_path.with_name(name[: -len(".json")] + ".meta.json")
    return weights_path.with_name(name + ".meta.json")


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_TEXT_EXTS = {".txt"}


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
    embed_dim: int,
    hidden: int,
    kernel: int,
    stride: int,
    padding: int,
    mixer_depth: int,
    curvature: float,
    temperature: float,
) -> st.nn.Sequential:
    mixer_depth = max(1, int(mixer_depth))

    model = st.nn.Sequential()
    model.add(st.nn.Embedding("embed", vocab_size, embed_dim))
    # Embedding emits (batch, steps * channels) laid out like NHWC (w-major, then channels).
    # WaveRnn/Conv1d expects channels-major, so reorder NHWC -> NCHW before the 1D conv stack.
    model.add(
        st.nn.FeatureReorder2d(
            embed_dim,
            1,
            steps,
            layout="NHWC",
            direction="to_canonical",
        )
    )
    model.add(
        st.nn.WaveRnn(
            "wrnn",
            embed_dim,
            hidden,
            kernel,
            curvature,
            temperature,
            stride=stride,
            padding=padding,
        )
    )
    for depth in range(mixer_depth):
        model.add(st.nn.ZSpaceMixer(f"mixer_{depth}", hidden))
        model.add(st.nn.WaveGate(f"gate_{depth}", hidden, curvature, temperature))
    model.add(st.nn.Linear("head", hidden, vocab_size))
    model.add(st.nn.ZSpaceSoftmax(curvature, temperature))
    return model


def _estimate_parameter_count(module: Any) -> int:
    state = getattr(module, "state_dict", None)
    if state is None:
        return 0
    try:
        entries = state()
    except Exception:
        return 0

    total = 0
    for _name, tensor in entries:
        try:
            shape = tensor.shape()
            if isinstance(shape, tuple) and len(shape) >= 2:
                rows = int(shape[0])
                cols = int(shape[1])
            else:
                rows = int(getattr(tensor, "rows", 0) or 0)
                cols = int(getattr(tensor, "cols", 0) or 0)
            if rows > 0 and cols > 0:
                total += rows * cols
        except Exception:
            continue
    return total


def _resolve_weights_format(requested: str, *, parameter_count: int) -> str:
    requested = str(requested).strip().lower()
    if requested in {"json", "bincode"}:
        return requested
    if requested != "auto":
        raise ValueError("--weights-format must be json|bincode|auto")
    # JSON is convenient for tiny models; bincode avoids huge JSON payloads.
    if parameter_count >= 250_000:
        return "bincode"
    return "json"


def _weights_format_for_path(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".bin", ".bincode", ".bc"}:
        return "bincode"
    raise ValueError(
        "unsupported weights extension (expected .json, .bin, .bincode, or .bc)"
    )


def _weights_path_for_run(run_dir: pathlib.Path, weights_format: str) -> pathlib.Path:
    weights_format = str(weights_format).strip().lower()
    if weights_format == "json":
        return run_dir / "weights.json"
    if weights_format == "bincode":
        return run_dir / "weights.bin"
    raise ValueError(f"unsupported weights format: {weights_format}")


def _evaluate_loss(
    model: st.nn.Sequential,
    loss: Any,
    batches: list[tuple[st.Tensor, st.Tensor]],
) -> float:
    if not batches:
        return float("nan")
    total = 0.0
    for x, y in batches:
        pred = model.forward(x)
        value = loss.forward(pred, y).tolist()
        total += float(value[0][0])
    return total / float(len(batches))


def _apply_preset(
    preset: str,
    *,
    steps: int,
    embed_dim: int,
    hidden: int,
    kernel: int,
    stride: int,
    mixer_depth: int,
    epochs: int,
    batches_per_epoch: int,
    batch: int,
    lr: float,
    top_k: int,
    temperature: float,
    weights_format: str,
    checkpoint_every: int,
    val_split: float,
    val_batches: int,
) -> tuple[int, int, int, int, int, int, int, int, int, float, int, float, str, int, float, int]:
    preset = str(preset).strip().lower()
    cfg = _PRESETS.get(preset)
    if cfg is None:
        raise ValueError(f"unknown --preset: {preset} (available: {', '.join(sorted(_PRESETS))})")

    return (
        int(cfg.get("steps", steps)),
        int(cfg.get("embed_dim", embed_dim)),
        int(cfg.get("hidden", hidden)),
        int(cfg.get("kernel", kernel)),
        int(cfg.get("stride", stride)),
        int(cfg.get("mixer_depth", mixer_depth)),
        int(cfg.get("epochs", epochs)),
        int(cfg.get("batches_per_epoch", batches_per_epoch)),
        int(cfg.get("batch", batch)),
        float(cfg.get("lr", lr)),
        int(cfg.get("top_k", top_k)),
        float(cfg.get("temperature", temperature)),
        str(cfg.get("weights_format", weights_format)),
        int(cfg.get("checkpoint_every", checkpoint_every)),
        float(cfg.get("val_split", val_split)),
        int(cfg.get("val_batches", val_batches)),
    )


def _build_random_batch(
    tokens: list[int],
    vocab_size: int,
    steps: int,
    batch: int,
    rng: random.Random,
) -> tuple[st.Tensor, st.Tensor]:
    max_start = len(tokens) - steps
    if max_start <= 0:
        raise ValueError(f"text too short for steps={steps}: len={len(tokens)}")

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


def _generate(
    model: st.nn.Sequential,
    symbols: list[str],
    index: dict[str, int],
    steps: int,
    prompt: str,
    gen_len: int,
    top_k: int,
    seed: int,
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
        x = st.Tensor(1, steps, [float(i) for i in context])
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
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_wave_rnn_mixer.py <text_or_dir> [<text_or_dir> ...] "
            "[--preset tiny|small|base|large] "
            "[--load weights.json] [--save weights.json] [--steps N] [--embed-dim N] [--hidden N] "
            "[--kernel N] [--stride N] [--padding N] "
            "[--mixer-depth N] "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] "
            "[--gen N] [--topk N] [--seed N] [--prompt STR] "
            "[--infuse STR] [--infuse-every once|epoch|batch] [--infuse-mode blend|separate] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--weights-format json|bincode|auto] [--checkpoint-every N] "
            "[--val-split F] [--val-batches N] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--desire] [--desire-concepts N] [--desire-prime N] [--desire-blend F] [--desire-drift-gain F] "
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

    load_weights: pathlib.Path | None = None
    save_weights: pathlib.Path | None = None
    run_dir: pathlib.Path | None = None
    preset: str | None = None
    steps = 32
    embed_dim = 32
    hidden = 64
    kernel = 3
    stride = 1
    padding: int | None = None
    mixer_depth = 1
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
    infuse: str | None = None
    infuse_every = "once"
    infuse_mode: str | None = None
    backend = "cpu"
    weights_format = "json"
    checkpoint_every = 0
    val_split = 0.1
    val_batches = 0
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

    # Apply presets before parsing other flags so explicit args override defaults.
    if "--preset" in args:
        idx = args.index("--preset")
        if idx + 1 >= len(args):
            raise ValueError("--preset requires a value")
        preset = str(args[idx + 1]).strip().lower()
        del args[idx : idx + 2]
        (
            steps,
            embed_dim,
            hidden,
            kernel,
            stride,
            mixer_depth,
            epochs,
            batches_per_epoch,
            batch,
            lr,
            top_k,
            temperature,
            weights_format,
            checkpoint_every,
            val_split,
            val_batches,
        ) = _apply_preset(
            preset,
            steps=steps,
            embed_dim=embed_dim,
            hidden=hidden,
            kernel=kernel,
            stride=stride,
            mixer_depth=mixer_depth,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            batch=batch,
            lr=lr,
            top_k=top_k,
            temperature=temperature,
            weights_format=weights_format,
            checkpoint_every=checkpoint_every,
            val_split=val_split,
            val_batches=val_batches,
        )
        if checkpoint_every == 0 and weights_format == "bincode":
            checkpoint_every = 1

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
        elif flag == "--kernel":
            kernel = int(next(it))
        elif flag == "--stride":
            stride = int(next(it))
        elif flag == "--padding":
            padding = int(next(it))
        elif flag == "--mixer-depth":
            mixer_depth = int(next(it))
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
        elif flag == "--infuse":
            infuse = str(next(it))
        elif flag == "--infuse-every":
            infuse_every = str(next(it)).strip().lower()
        elif flag == "--infuse-mode":
            infuse_mode = str(next(it)).strip().lower()
        elif flag == "--backend":
            backend = str(next(it)).strip().lower()
        elif flag == "--weights-format":
            weights_format = str(next(it)).strip().lower()
        elif flag == "--checkpoint-every":
            checkpoint_every = int(next(it))
        elif flag == "--val-split":
            val_split = float(next(it))
        elif flag == "--val-batches":
            val_batches = int(next(it))
        elif flag == "--events":
            events_path = pathlib.Path(next(it))
        elif flag == "--events-types":
            raw = str(next(it))
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if parts:
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
            run_dir = pathlib.Path(next(it))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if kernel <= 0:
        raise ValueError("--kernel must be > 0")
    if stride <= 0:
        raise ValueError("--stride must be > 0")
    if padding is None:
        padding = kernel // 2
    if padding < 0:
        raise ValueError("--padding must be >= 0")
    if embed_dim <= 0:
        raise ValueError("--embed-dim must be > 0")
    if hidden <= 0:
        raise ValueError("--hidden must be > 0")
    if steps <= 0:
        raise ValueError("--steps must be > 0")
    if mixer_depth <= 0:
        raise ValueError("--mixer-depth must be > 0")
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if val_batches < 0:
        raise ValueError("--val-batches must be >= 0")
    if val_split <= 0.0 or val_split >= 0.95:
        raise ValueError("--val-split must be within (0, 0.95)")
    if infuse_every not in {"once", "epoch", "batch"}:
        raise ValueError(
            f"invalid --infuse-every: {infuse_every} (expected once|epoch|batch)"
        )
    if infuse_every != "once" and infuse is None:
        raise ValueError("--infuse-every requires --infuse")
    if infuse_mode is not None and infuse_mode not in {"blend", "separate"}:
        raise ValueError(
            f"invalid --infuse-mode: {infuse_mode} (expected blend|separate)"
        )
    if infuse_mode is not None and infuse is None:
        raise ValueError("--infuse-mode requires --infuse")

    _require_backend_available(backend)

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
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    (run_dir / "data_files.txt").write_text(
        "\n".join(str(path) for path in data_files) + "\n",
        encoding="utf-8",
    )

    if atlas and events_path is None:
        events_path = run_dir / "events.jsonl"

    if load_weights is not None:
        meta_path = _meta_path_for_weights(load_weights)
        meta = _load_meta(meta_path)
        fmt = str(meta.get("format", ""))
        if fmt != FORMAT:
            raise ValueError(f"unexpected meta format: {fmt}")
        steps = int(meta["steps"])
        embed_dim = int(meta["embed_dim"])
        hidden = int(meta["hidden"])
        kernel = int(meta["kernel"])
        stride = int(meta["stride"])
        padding = int(meta["padding"])
        mixer_depth = int(meta.get("mixer_depth", 1))
        curvature = float(meta["curvature"])
        temperature = float(meta["temperature"])
        symbols = [str(ch) for ch in meta["symbols"]]
        index = {ch: i for i, ch in enumerate(symbols)}
        vocab_size = len(symbols)
        model = _build_model(
            vocab_size,
            steps,
            embed_dim,
            hidden,
            kernel,
            stride,
            padding,
            mixer_depth,
            curvature,
            temperature,
        )
        st.nn.load(str(load_weights), model)
    else:
        _unk, symbols, index = _build_vocab(text, DEFAULT_UNK)
        vocab_size = len(symbols)
        model = _build_model(
            vocab_size,
            steps,
            embed_dim,
            hidden,
            kernel,
            stride,
            padding,
            mixer_depth,
            curvature,
            temperature,
        )

    tokens = _encode(text, index)
    if len(tokens) <= steps:
        raise ValueError(f"text too short for steps={steps}: len={len(tokens)}")

    if prompt is None:
        prompt = "".join(list(text)[:steps])

    parameter_count = _estimate_parameter_count(model)
    resolved_weights_format = _resolve_weights_format(weights_format, parameter_count=parameter_count)
    weights_suffix = _weights_path_for_run(run_dir, resolved_weights_format).suffix

    train_tokens = tokens
    val_tokens: list[int] = []
    if val_batches > 0:
        split_idx = int(float(len(tokens)) * (1.0 - float(val_split)))
        split_idx = max(0, min(len(tokens), split_idx))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        if len(train_tokens) <= steps + 1 or len(val_tokens) <= steps + 1:
            # If the text is too short, just disable validation to avoid confusing errors.
            train_tokens = tokens
            val_tokens = []
            val_batches = 0

    meta_base = {
        "format": FORMAT,
        "preset": preset,
        "steps": steps,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "mixer_depth": mixer_depth,
        "curvature": curvature,
        "temperature": temperature,
        "parameter_count": parameter_count,
        "unk": symbols[0],
        "symbols": symbols,
    }

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "llm_char_wave_rnn_mixer",
        "data_paths": [str(path) for path in data_paths],
        "data_file_count": len(data_files),
        "data_files_manifest": str(run_dir / "data_files.txt"),
        "format": FORMAT,
        "preset": preset,
        "steps": steps,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "mixer_depth": mixer_depth,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "lr": lr,
        "curvature": curvature,
        "temperature": temperature,
        "backend": backend,
        "parameter_count": parameter_count,
        "weights_format": weights_format,
        "resolved_weights_format": resolved_weights_format,
        "checkpoint_every": checkpoint_every,
        "val_split": val_split if val_batches > 0 else None,
        "val_batches": val_batches if val_batches > 0 else None,
        "gen_len": gen_len,
        "top_k": top_k,
        "seed": seed,
        "prompt": prompt,
        "vocab_size": vocab_size,
        "symbols_count": len(symbols),
        "infuse": infuse,
        "infuse_every": infuse_every,
        "infuse_mode": infuse_mode,
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
    if infuse is not None:
        if infuse_mode is None:
            if infuse_every == "once":
                infuse_mode = "separate"
            else:
                infuse_mode = "blend"
        trainer.set_text_infusion(infuse, every=infuse_every, mode=infuse_mode)

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
        f"vocab={vocab_size} files={len(data_files)} chars={len(text)} steps={steps} embed={embed_dim} hidden={hidden} kernel={kernel} depth={mixer_depth} "
        f"epochs={epochs} batch={batch} lr={lr:.3e} curvature={curvature} temp={temperature} "
        f"backend={backend} params={parameter_count} weights={resolved_weights_format} run_dir={run_dir}"
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

        best_metric = float("inf")
        best_epoch = None

        val_batches_cache: list[tuple[st.Tensor, st.Tensor]] = []
        if val_batches > 0 and val_tokens:
            val_rng = random.Random(seed + 987_654)
            for _ in range(val_batches):
                val_batches_cache.append(
                    _build_random_batch(val_tokens, vocab_size, steps, batch, val_rng)
                )

        for epoch in range(max(0, epochs)):
            print(
                f"epoch[{epoch}] start batches={batches_per_epoch} batch={batch} backend={backend}",
                flush=True,
            )
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
                    desire=desire_pipeline,
                )
            batches = []
            for _ in range(batches_per_epoch):
                batches.append(_build_random_batch(train_tokens, vocab_size, steps, batch, rng))
            stats = trainer.train_epoch(model, loss, batches, schedule)
            avg_loss = float(stats.average_loss)

            val_loss = (
                _evaluate_loss(model, loss, val_batches_cache)
                if val_batches_cache
                else float("nan")
            )

            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batches": int(stats.batches),
                            "average_loss": avg_loss,
                            "val_loss": val_loss if val_loss == val_loss else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if gen_len > 0:
                print(f"epoch[{epoch}] sampling gen_len={gen_len}...", flush=True)
            sample = _generate(
                model,
                symbols,
                index,
                steps,
                prompt,
                gen_len,
                top_k,
                seed + 999 + epoch,
                desire=desire_pipeline,
            )
            (samples_dir / f"epoch_{epoch:03d}.txt").write_text(sample, encoding="utf-8")
            if val_loss == val_loss:
                print(
                    f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f} val_loss={val_loss:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f}",
                    flush=True,
                )

            tracked = val_loss if val_loss == val_loss else avg_loss
            if tracked < best_metric:
                best_metric = tracked
                best_epoch = epoch
                best_weights_path = run_dir / f"best_weights{weights_suffix}"
                print(f"epoch[{epoch}] saving {best_weights_path.name}...", flush=True)
                st.nn.save(str(best_weights_path), model)
                best_meta = dict(meta_base)
                best_meta["weights_format"] = resolved_weights_format
                best_meta["best_epoch"] = epoch
                best_meta["epoch"] = epoch
                _save_meta(_meta_path_for_weights(best_weights_path), best_meta)
                print(f"epoch[{epoch}] saved {best_weights_path.name}", flush=True)

            if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0):
                ckpt_dir = run_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}{weights_suffix}"
                print(f"epoch[{epoch}] saving {ckpt_path.name}...", flush=True)
                st.nn.save(str(ckpt_path), model)
                ckpt_meta = dict(meta_base)
                ckpt_meta["weights_format"] = resolved_weights_format
                ckpt_meta["epoch"] = epoch
                _save_meta(_meta_path_for_weights(ckpt_path), ckpt_meta)
                print(f"epoch[{epoch}] saved {ckpt_path.name}", flush=True)

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

    weights_path = _weights_path_for_run(run_dir, resolved_weights_format)
    st.nn.save(str(weights_path), model)

    meta = dict(meta_base)
    meta.update(
        {
        "weights_format": resolved_weights_format,
        "checkpoint_every": checkpoint_every,
        "best_epoch": best_epoch,
        "val_split": val_split if val_batches > 0 else None,
        "val_batches": val_batches if val_batches > 0 else None,
        }
    )
    _save_meta(_meta_path_for_weights(weights_path), meta)

    if save_weights is not None:
        save_weights = pathlib.Path(save_weights)
        save_weights.parent.mkdir(parents=True, exist_ok=True)
        st.nn.save(str(save_weights), model)
        save_meta = dict(meta)
        save_meta["weights_format"] = _weights_format_for_path(save_weights)
        _save_meta(_meta_path_for_weights(save_weights), save_meta)

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
            desire=desire_pipeline,
        )
        (samples_dir / "init.txt").write_text(sample, encoding="utf-8")
    print("--- sample (prompt + gen) ---", flush=True)
    print(sample, flush=True)


if __name__ == "__main__":
    main()
