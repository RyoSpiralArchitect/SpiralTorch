from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
import random
import shlex
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

FORMAT = "st-llm-char-vae-context-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"
DEFAULT_UNK = "\uFFFD"
FEATURE_RAW = "raw"
FEATURE_RECONSTRUCTION = "reconstruction"
FEATURE_LATENT = "latent"
FEATURE_RAW_LATENT = "raw_latent"
FEATURE_RECONSTRUCTION_LATENT = "reconstruction_latent"
FEATURE_FAMILY_HYBRID_LATENT = "hybrid_latent"
FEATURE_CHOICES = (
    FEATURE_RAW,
    FEATURE_RECONSTRUCTION,
    FEATURE_LATENT,
    FEATURE_RAW_LATENT,
    FEATURE_RECONSTRUCTION_LATENT,
)
NORMALIZE_CHOICES = ("none", "vector", "blocks")
HEAD_INIT_CHOICES = ("legacy", "xavier")
FOLLOW_UP_VERDICTS = ("improved", "confirmed", "regressed", "unknown")
FOLLOW_UP_CHAIN_MAX_ANCESTORS = 8
RUN_BUDGET_KEYS = (
    "window_chars",
    "latent_dim",
    "hidden",
    "epochs",
    "batches",
    "batch_size",
    "eval_samples",
    "vae_epochs",
    "vae_batches",
    "vae_batch_size",
)

_TEXT_EXTS = {".txt"}


@dataclass(frozen=True)
class _WindowSample:
    window: str
    target: int


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
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in _TEXT_EXTS:
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


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_vocab(text: str, unk: str = DEFAULT_UNK) -> tuple[list[str], dict[str, int]]:
    symbols = sorted({ch for ch in text if ch != unk})
    symbols = [unk, *symbols]
    return symbols, {ch: idx for idx, ch in enumerate(symbols)}


def _encode_tokens(text: str, index: dict[str, int]) -> list[int]:
    return [index.get(ch, 0) for ch in text]


def _split_text(text: str, val_ratio: float, window_chars: int) -> tuple[str, str]:
    if val_ratio <= 0.0 or len(text) <= window_chars + 1:
        return text, text
    split_at = int(len(text) * (1.0 - val_ratio))
    split_at = min(max(split_at, window_chars + 1), len(text) - 1)
    train_text = text[:split_at]
    val_text = text[split_at:]
    if len(val_text) <= window_chars:
        val_text = text
    return train_text, val_text


def _pick_window(text: str, window_chars: int, index: dict[str, int], rng: random.Random) -> _WindowSample:
    if len(text) <= window_chars:
        raise ValueError(
            f"text too short for window_chars={window_chars}: len(text)={len(text)}"
        )
    start = rng.randrange(0, len(text) - window_chars)
    end = start + window_chars
    return _WindowSample(window=text[start:end], target=index.get(text[end], 0))


def _pick_text_window(text: str, window_chars: int, rng: random.Random) -> str:
    if len(text) <= window_chars:
        raise ValueError(
            f"text too short for window_chars={window_chars}: len(text)={len(text)}"
        )
    start = rng.randrange(0, len(text) - window_chars)
    return text[start : start + window_chars]


def _sample_windows(
    text: str,
    window_chars: int,
    index: dict[str, int],
    count: int,
    rng: random.Random,
) -> list[_WindowSample]:
    return [_pick_window(text, window_chars, index, rng) for _ in range(max(0, count))]


def _one_hot_targets(samples: list[_WindowSample], vocab_size: int) -> st.Tensor:
    data = [0.0] * (len(samples) * vocab_size)
    for row, sample in enumerate(samples):
        if 0 <= sample.target < vocab_size:
            data[row * vocab_size + sample.target] = 1.0
    return st.Tensor(len(samples), vocab_size, data)


def _feature_dim(model: Any, feature: str) -> int:
    if feature == FEATURE_LATENT:
        return int(model.latent_dim)
    if feature in {FEATURE_RAW, FEATURE_RECONSTRUCTION}:
        return int(model.input_dim)
    if feature in {FEATURE_RAW_LATENT, FEATURE_RECONSTRUCTION_LATENT}:
        return int(model.input_dim) + int(model.latent_dim)
    raise ValueError(f"unknown feature: {feature}")


def _feature_family(feature: str) -> str:
    if feature in {FEATURE_RAW_LATENT, FEATURE_RECONSTRUCTION_LATENT}:
        return FEATURE_FAMILY_HYBRID_LATENT
    return feature


def _feature_family_members(family: str) -> list[str]:
    if family == FEATURE_FAMILY_HYBRID_LATENT:
        return [FEATURE_RAW_LATENT, FEATURE_RECONSTRUCTION_LATENT]
    if family in FEATURE_CHOICES:
        return [family]
    return []


def _feature_family_focused_features(features: list[str], family: str) -> list[str]:
    focused = list(dict.fromkeys(str(feature) for feature in features))
    if family != FEATURE_RAW and FEATURE_RAW not in focused:
        focused.insert(0, FEATURE_RAW)
    for feature in _feature_family_members(family):
        if feature not in focused:
            focused.append(feature)
    return focused


def _build_mellin_basis(model: Any, args: argparse.Namespace) -> Any | None:
    if args.mellin == "none":
        return None
    if args.mellin == "constant":
        return st.nn.MellinBasis.constant(int(model.input_dim), float(args.mellin_exponent))
    if args.mellin == "ramp":
        return st.nn.MellinBasis.ramp(
            int(model.input_dim),
            float(args.mellin_start),
            float(args.mellin_end),
        )
    raise ValueError("--mellin must be one of: none|constant|ramp")


def _l2_normalize(values: list[float], epsilon: float = 1e-12) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= epsilon:
        return values
    return [value / norm for value in values]


def _normalise_feature_values(
    model: Any,
    feature: str,
    values: list[float],
    mode: str,
    hybrid_latent_scale: float,
) -> list[float]:
    if feature in {FEATURE_RAW_LATENT, FEATURE_RECONSTRUCTION_LATENT}:
        split = int(model.input_dim)
        base_values = values[:split]
        latent_values = values[split:]
    else:
        base_values = values
        latent_values = []

    if mode == "none":
        if latent_values:
            return base_values + [value * hybrid_latent_scale for value in latent_values]
        return values
    if mode == "vector":
        if latent_values:
            values = base_values + [value * hybrid_latent_scale for value in latent_values]
        return _l2_normalize(values)
    if mode == "blocks":
        if latent_values:
            return _l2_normalize(base_values) + [
                value * hybrid_latent_scale for value in _l2_normalize(latent_values)
            ]
        return _l2_normalize(values)
    raise ValueError("--feature-normalize must be one of: none|vector|blocks")


def _feature_vector(
    model: Any,
    basis: Any | None,
    feature: str,
    text: str,
    normalize: str = "none",
    hybrid_latent_scale: float = 1.0,
) -> list[float]:
    if feature in {FEATURE_RAW, FEATURE_RAW_LATENT}:
        if basis is None:
            raw_values = model.encode_text(text)
        else:
            raw_values = model.encode_text_with_mellin(text, basis)
        if feature == FEATURE_RAW:
            return _normalise_feature_values(
                model,
                feature,
                [float(value) for value in raw_values],
                normalize,
                hybrid_latent_scale,
            )

    if basis is None:
        state = model.forward_mean_text(text)
    else:
        state = model.forward_mean_text_with_mellin(text, basis)
    if feature == FEATURE_LATENT:
        values = state.latent
    elif feature == FEATURE_RECONSTRUCTION:
        values = state.reconstruction
    elif feature == FEATURE_RAW_LATENT:
        values = [*raw_values, *state.latent]
    elif feature == FEATURE_RECONSTRUCTION_LATENT:
        values = [*state.reconstruction, *state.latent]
    else:
        raise ValueError(f"unknown feature: {feature}")
    return _normalise_feature_values(
        model,
        feature,
        [float(value) for value in values],
        normalize,
        hybrid_latent_scale,
    )


def _feature_tensor(
    model: Any,
    basis: Any | None,
    feature: str,
    samples: list[_WindowSample],
    normalize: str = "none",
    hybrid_latent_scale: float = 1.0,
) -> st.Tensor:
    dim = _feature_dim(model, feature)
    data: list[float] = []
    for sample in samples:
        values = _feature_vector(
            model,
            basis,
            feature,
            sample.window,
            normalize,
            hybrid_latent_scale,
        )
        if len(values) != dim:
            raise ValueError(f"{feature} feature length mismatch: expected {dim}, got {len(values)}")
        data.extend(values)
    return st.Tensor(len(samples), dim, data)


def _build_batches(
    vae: Any,
    basis: Any | None,
    feature: str,
    normalize: str,
    hybrid_latent_scale: float,
    text: str,
    index: dict[str, int],
    vocab_size: int,
    window_chars: int,
    batch_size: int,
    batches: int,
    rng: random.Random,
) -> list[tuple[st.Tensor, st.Tensor]]:
    out: list[tuple[st.Tensor, st.Tensor]] = []
    for _ in range(batches):
        samples = _sample_windows(text, window_chars, index, batch_size, rng)
        out.append((
            _feature_tensor(vae, basis, feature, samples, normalize, hybrid_latent_scale),
            _one_hot_targets(samples, vocab_size),
        ))
    return out


def _build_head(feature_dim: int, hidden: int, vocab_size: int, curvature: float, temperature: float) -> Any:
    model = st.nn.Sequential()
    if hidden > 0:
        model.add(st.nn.Linear("feature_proj", feature_dim, hidden))
        model.add(st.nn.Relu())
        model.add(st.nn.Linear("head", hidden, vocab_size))
    else:
        model.add(st.nn.Linear("head", feature_dim, vocab_size))
    model.add(st.nn.ZSpaceSoftmax(curvature, temperature))
    return model


def _xavier_limit(rows: int, cols: int) -> float:
    fan_sum = max(1, int(rows) + int(cols))
    return math.sqrt(6.0 / float(fan_sum))


def _rescale_values_to_abs_limit(values: Iterable[float], limit: float) -> list[float]:
    parsed = [float(value) for value in values]
    max_abs = max((abs(value) for value in parsed), default=0.0)
    if max_abs <= 0.0:
        return parsed
    scale = float(limit) / max_abs
    return [value * scale for value in parsed]


def _flatten_tensor_rows(tensor: Any) -> list[float]:
    rows = tensor.tolist()
    return [float(value) for row in rows for value in row]


def _rescale_head_init(head: Any, mode: str) -> dict[str, Any]:
    if mode == "legacy":
        return {"mode": mode, "rescaled": False, "layers": []}
    if mode != "xavier":
        raise ValueError("--head-init must be one of: legacy|xavier")

    next_state = []
    layers = []
    for name, tensor in head.state_dict():
        rows, cols = tensor.shape()
        if name.endswith("::weight"):
            limit = _xavier_limit(int(rows), int(cols))
            values = _flatten_tensor_rows(tensor)
            rescaled = _rescale_values_to_abs_limit(values, limit)
            next_state.append((name, st.Tensor(int(rows), int(cols), rescaled)))
            layers.append(
                {
                    "name": name,
                    "rows": int(rows),
                    "cols": int(cols),
                    "limit": float(limit),
                    "max_abs_before": max(
                        (abs(value) for value in values),
                        default=0.0,
                    ),
                    "max_abs_after": max(
                        (abs(value) for value in rescaled),
                        default=0.0,
                    ),
                }
            )
        else:
            next_state.append((name, tensor))
    head.load_state_dict(next_state)
    return {"mode": mode, "rescaled": True, "layers": layers}


def _normalise_grad_clip(raw: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"none", "off", "0"}:
        return None
    parsed = float(value)
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise ValueError("--vae-grad-clip must be positive, or one of none|off|0")
    return parsed


def _train_vae(
    vae: Any,
    basis: Any | None,
    train_text: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for epoch in range(max(0, int(args.vae_epochs))):
        rng = random.Random(int(args.seed) + 500_000 + epoch * 10_000)
        recon_sum = 0.0
        weighted_sum = 0.0
        grad_sum = 0.0
        update_sum = 0.0
        for _ in range(int(args.vae_batches)):
            windows = [
                _pick_text_window(train_text, int(args.window_chars), rng)
                for _ in range(int(args.vae_batch_size))
            ]
            if basis is None:
                stats = vae.train_text_batch(windows, float(args.vae_lr), float(args.vae_kl_weight))
            else:
                stats = vae.train_text_batch_with_mellin(
                    windows,
                    basis,
                    float(args.vae_lr),
                    float(args.vae_kl_weight),
                )
            recon_sum += float(stats.recon_loss)
            weighted_sum += float(stats.weighted_loss)
            grad_sum += float(stats.gradient_l2)
            update_sum += float(stats.update_l2)
        denom = float(args.vae_batches)
        item = {
            "epoch": epoch,
            "avg_recon_loss": recon_sum / denom,
            "avg_weighted_loss": weighted_sum / denom,
            "avg_gradient_l2": grad_sum / denom,
            "avg_update_l2": update_sum / denom,
        }
        history.append(item)
        print(
            "vae_epoch[{epoch}] avg_recon_loss={recon:.6f} avg_weighted_loss={weighted:.6f} "
            "avg_grad_l2={grad:.6f} avg_update_l2={update:.6f}".format(
                epoch=epoch,
                recon=item["avg_recon_loss"],
                weighted=item["avg_weighted_loss"],
                grad=item["avg_gradient_l2"],
                update=item["avg_update_l2"],
            ),
            flush=True,
        )
    return history


def _nll_from_prob(prob: float, epsilon: float = 1e-9) -> float:
    return -math.log(max(float(prob), epsilon))


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _squared_l2(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _cosine(left: list[float], right: list[float]) -> float | None:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        left_norm += fa * fa
        right_norm += fb * fb
    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    return dot / denom if denom > 0.0 else None


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_val = float("-inf")
    for idx, value in enumerate(values):
        if value == value and value > best_val:
            best_idx = idx
            best_val = value
    return best_idx


def _sample_topk(values: list[float], top_k: int, rng: random.Random) -> int:
    if not values:
        return 0
    if top_k <= 1:
        return _argmax(values)
    candidates = [(idx, max(0.0, float(value))) for idx, value in enumerate(values)]
    if top_k < len(candidates):
        candidates.sort(key=lambda item: item[1], reverse=True)
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


def _feature_variance(vectors: list[list[float]], threshold: float = 1e-8) -> dict[str, Any]:
    if not vectors:
        return {
            "dims": 0,
            "variance_mean": None,
            "variance_max": None,
            "active_dim_fraction": None,
            "norm_mean": None,
        }
    dims = min(len(vector) for vector in vectors)
    if dims <= 0:
        return {
            "dims": 0,
            "variance_mean": None,
            "variance_max": None,
            "active_dim_fraction": None,
            "norm_mean": None,
        }
    means = [
        sum(float(vector[idx]) for vector in vectors) / float(len(vectors))
        for idx in range(dims)
    ]
    variances = []
    for idx in range(dims):
        variances.append(
            sum((float(vector[idx]) - means[idx]) ** 2 for vector in vectors)
            / float(len(vectors))
        )
    active = sum(1 for value in variances if value > threshold)
    return {
        "dims": dims,
        "variance_mean": sum(variances) / float(dims),
        "variance_max": max(variances) if variances else None,
        "active_dim_fraction": active / float(dims),
        "norm_mean": sum(_vector_norm(vector[:dims]) for vector in vectors) / float(len(vectors)),
    }


def _feature_diagnostics(
    vae: Any,
    basis: Any | None,
    features: list[str],
    normalize: str,
    hybrid_latent_scale: float,
    text: str,
    index: dict[str, int],
    window_chars: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    windows = _sample_windows(text, window_chars, index, max(1, samples), rng)
    vectors = {
        feature: [
            _feature_vector(
                vae,
                basis,
                feature,
                sample.window,
                normalize,
                hybrid_latent_scale,
            )
            for sample in windows
        ]
        for feature in features
    }
    raw_vectors = vectors.get(FEATURE_RAW)
    diagnostics: dict[str, Any] = {}
    for feature, feature_vectors in vectors.items():
        item = _feature_variance(feature_vectors)
        if raw_vectors is not None and raw_vectors and feature_vectors:
            dims = min(
                min(len(vector) for vector in raw_vectors),
                min(len(vector) for vector in feature_vectors),
            )
            if dims == min(len(vector) for vector in raw_vectors) == min(
                len(vector) for vector in feature_vectors
            ):
                sqs = [
                    _squared_l2(raw[:dims], vector[:dims])
                    for raw, vector in zip(raw_vectors, feature_vectors)
                ]
                cosines = [
                    value
                    for value in (
                        _cosine(raw[:dims], vector[:dims])
                        for raw, vector in zip(raw_vectors, feature_vectors)
                    )
                    if value is not None
                ]
                item["raw_l2_mean"] = sum(math.sqrt(value) for value in sqs) / float(len(sqs))
                item["raw_mse_mean"] = sum(value / float(dims) for value in sqs) / float(len(sqs))
                item["raw_cosine_mean"] = (
                    sum(cosines) / float(len(cosines)) if cosines else None
                )
            else:
                item["raw_l2_mean"] = None
                item["raw_mse_mean"] = None
                item["raw_cosine_mean"] = None
        else:
            item["raw_l2_mean"] = None
            item["raw_mse_mean"] = None
            item["raw_cosine_mean"] = None
        diagnostics[feature] = item
    return {
        "samples": len(windows),
        "features": diagnostics,
    }


def _evaluate(
    head: Any,
    vae: Any,
    basis: Any | None,
    feature: str,
    normalize: str,
    hybrid_latent_scale: float,
    text: str,
    index: dict[str, int],
    vocab_size: int,
    window_chars: int,
    eval_samples: int,
    seed: int,
) -> dict[str, Any]:
    if eval_samples <= 0:
        return {
            "windows": 0,
            "mean_nll": None,
            "perplexity": None,
            "accuracy": None,
            "mean_target_probability": None,
        }
    rng = random.Random(seed)
    samples = _sample_windows(text, window_chars, index, eval_samples, rng)
    x = _feature_tensor(vae, basis, feature, samples, normalize, hybrid_latent_scale)
    probs = head.forward(x).tolist()
    nll = 0.0
    correct = 0
    target_prob = 0.0
    for row, sample in zip(probs, samples):
        values = [float(value) for value in row]
        prob = values[sample.target] if 0 <= sample.target < len(values) else 0.0
        target_prob += prob
        nll += _nll_from_prob(prob)
        if _argmax(values) == sample.target:
            correct += 1
    mean_nll = nll / float(len(samples))
    return {
        "windows": len(samples),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 80.0 else None,
        "accuracy": correct / float(len(samples)),
        "mean_target_probability": target_prob / float(len(samples)),
    }


def _generate(
    head: Any,
    vae: Any,
    basis: Any | None,
    feature: str,
    normalize: str,
    hybrid_latent_scale: float,
    symbols: list[str],
    index: dict[str, int],
    prompt: str,
    window_chars: int,
    gen_len: int,
    top_k: int,
    seed: int,
) -> str:
    if gen_len <= 0:
        return prompt
    rng = random.Random(seed)
    out = prompt
    for _ in range(gen_len):
        context = out[-window_chars:]
        if len(context) < window_chars:
            context = (DEFAULT_UNK * (window_chars - len(context))) + context
        sample = _WindowSample(window=context, target=0)
        x = _feature_tensor(vae, basis, feature, [sample], normalize, hybrid_latent_scale)
        probs = [float(value) for value in head.forward(x).tolist()[0]]
        next_idx = _sample_topk(probs, top_k, rng)
        out += symbols[next_idx] if 0 <= next_idx < len(symbols) else DEFAULT_UNK
    return out


def _train_feature_head(
    feature: str,
    vae: Any,
    basis: Any | None,
    train_text: str,
    val_text: str,
    symbols: list[str],
    index: dict[str, int],
    args: argparse.Namespace,
    run_dir: pathlib.Path,
) -> dict[str, Any]:
    vocab_size = len(symbols)
    feature_dim = _feature_dim(vae, feature)
    head = _build_head(
        feature_dim,
        int(args.hidden),
        vocab_size,
        float(args.curvature),
        float(args.temperature),
    )
    head_init = _rescale_head_init(head, str(args.head_init))
    head.attach_hypergrad(curvature=float(args.curvature), learning_rate=float(args.lr))
    trainer = st.nn.ModuleTrainer(
        backend=str(args.backend),
        curvature=float(args.curvature),
        hyper_learning_rate=float(args.lr),
        fallback_learning_rate=float(args.lr),
    )
    loss = st.nn.CategoricalCrossEntropy()
    schedule = trainer.roundtable(
        int(args.batch_size),
        vocab_size,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    metrics_path = run_dir / f"metrics_{feature}.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    initial_validation = _evaluate(
        head,
        vae,
        basis,
        feature,
        str(args.feature_normalize),
        float(args.hybrid_latent_scale),
        val_text,
        index,
        vocab_size,
        int(args.window_chars),
        int(args.eval_samples),
        int(args.seed) + 800_000,
    )
    print(
        f"{feature}[init] val_nll={initial_validation['mean_nll']:.6f} "
        f"acc={float(initial_validation['accuracy']) * 100.0:.2f}%",
        flush=True,
    )

    best_validation = initial_validation
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(max(0, int(args.epochs))):
        rng = random.Random(int(args.seed) + epoch * 10_000)
        batches = _build_batches(
            vae,
            basis,
            feature,
            str(args.feature_normalize),
            float(args.hybrid_latent_scale),
            train_text,
            index,
            vocab_size,
            int(args.window_chars),
            int(args.batch_size),
            int(args.batches),
            rng,
        )
        stats = trainer.train_epoch(head, loss, batches, schedule)
        validation = _evaluate(
            head,
            vae,
            basis,
            feature,
            str(args.feature_normalize),
            float(args.hybrid_latent_scale),
            val_text,
            index,
            vocab_size,
            int(args.window_chars),
            int(args.eval_samples),
            int(args.seed) + 800_000 + epoch + 1,
        )
        item = {
            "feature": feature,
            "epoch": epoch,
            "batches": int(stats.batches),
            "average_loss": float(stats.average_loss),
            "validation": validation,
        }
        history.append(item)
        _append_jsonl(metrics_path, item)
        if (
            validation["mean_nll"] is not None
            and (
                best_validation["mean_nll"] is None
                or float(validation["mean_nll"]) < float(best_validation["mean_nll"])
            )
        ):
            best_validation = validation
            best_epoch = epoch
        print(
            f"{feature}[{epoch}] train_loss={float(stats.average_loss):.6f} "
            f"val_nll={float(validation['mean_nll']):.6f} "
            f"acc={float(validation['accuracy']) * 100.0:.2f}%",
            flush=True,
        )

    sample = _generate(
        head,
        vae,
        basis,
        feature,
        str(args.feature_normalize),
        float(args.hybrid_latent_scale),
        symbols,
        index,
        str(args.prompt),
        int(args.window_chars),
        int(args.gen),
        int(args.top_k),
        int(args.seed) + 900_000,
    )
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"{feature}.txt"
    sample_path.write_text(sample, encoding="utf-8")
    weights_path = run_dir / f"head_{feature}.json"
    st.nn.save(str(weights_path), head)
    best_nll = _finite_float(best_validation.get("mean_nll"))
    final_validation = history[-1]["validation"] if history else initial_validation
    final_nll = _finite_float(final_validation.get("mean_nll"))
    initial_nll = _finite_float(initial_validation.get("mean_nll"))
    validation_nlls = [
        value
        for value in (
            _finite_float(item.get("validation", {}).get("mean_nll"))
            for item in history
        )
        if value is not None
    ]
    if not validation_nlls and initial_nll is not None:
        validation_nlls.append(initial_nll)
    return {
        "feature": feature,
        "feature_dim": feature_dim,
        "head_init": head_init,
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "best_step": 0 if best_epoch is None else int(best_epoch) + 1,
        "validation_nll_mean": (
            sum(validation_nlls) / len(validation_nlls) if validation_nlls else None
        ),
        "validation_nll_initial_minus_best": (
            initial_nll - best_nll
            if initial_nll is not None and best_nll is not None
            else None
        ),
        "validation_nll_final_minus_best": (
            final_nll - best_nll
            if final_nll is not None and best_nll is not None
            else None
        ),
        "final_validation": final_validation,
        "history": history,
        "sample_path": str(sample_path),
        "weights_path": str(weights_path),
    }


def _parse_features(raw: str) -> list[str]:
    features = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not features:
        raise ValueError("--features must contain at least one feature")
    unknown = [feature for feature in features if feature not in FEATURE_CHOICES]
    if unknown:
        joined = ", ".join(FEATURE_CHOICES)
        raise ValueError(f"unknown --features entries {unknown}; expected comma-separated {joined}")
    return list(dict.fromkeys(features))


def _parse_seeds(seed: int, raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return [int(seed)]
    seeds = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return list(dict.fromkeys(seeds))


def _parse_hybrid_latent_scales(default_scale: float, raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return [float(default_scale)]
    scales = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        parsed = float(value)
        if parsed < 0.0 or not math.isfinite(parsed):
            raise ValueError("--hybrid-latent-scales values must be non-negative and finite")
        scales.append(parsed)
    if not scales:
        raise ValueError("--hybrid-latent-scales must contain at least one value")

    unique: list[float] = []
    seen: set[float] = set()
    for scale in scales:
        if scale in seen:
            continue
        seen.add(scale)
        unique.append(scale)
    return unique


def _parse_feature_normalize_modes(default_mode: str, raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return [default_mode]
    modes = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not modes:
        raise ValueError("--feature-normalize-modes must contain at least one mode")
    unknown = [mode for mode in modes if mode not in NORMALIZE_CHOICES]
    if unknown:
        joined = ", ".join(NORMALIZE_CHOICES)
        raise ValueError(
            f"unknown --feature-normalize-modes entries {unknown}; expected comma-separated {joined}"
        )
    return list(dict.fromkeys(modes))


def _clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _metric_stats(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "stddev": None,
            "stderr": None,
            "min": None,
            "max": None,
        }
    mean = sum(vals) / float(len(vals))
    variance = sum((value - mean) ** 2 for value in vals) / float(len(vals))
    return {
        "count": len(vals),
        "mean": mean,
        "stddev": math.sqrt(variance),
        "stderr": math.sqrt(variance) / math.sqrt(float(len(vals))),
        "min": min(vals),
        "max": max(vals),
    }


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _validation_mean_nll(validation: Any) -> float | None:
    if not isinstance(validation, dict):
        return None
    return _finite_float(validation.get("mean_nll"))


def _feature_result_validation_nlls(feature_result: dict[str, Any]) -> list[float]:
    values: list[float] = []
    history = feature_result.get("history", [])
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            value = _validation_mean_nll(item.get("validation"))
            if value is not None:
                values.append(value)
    if not values:
        final = _validation_mean_nll(feature_result.get("final_validation"))
        if final is not None:
            values.append(final)
    if not values:
        initial = _validation_mean_nll(feature_result.get("initial_validation"))
        if initial is not None:
            values.append(initial)
    return values


def _feature_result_best_nll(feature_result: dict[str, Any]) -> float | None:
    best = _validation_mean_nll(feature_result.get("best_validation"))
    if best is not None:
        return best
    values = _feature_result_validation_nlls(feature_result)
    return min(values) if values else None


def _feature_result_final_nll(feature_result: dict[str, Any]) -> float | None:
    final = _validation_mean_nll(feature_result.get("final_validation"))
    if final is not None:
        return final
    history = feature_result.get("history", [])
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            value = _validation_mean_nll(item.get("validation"))
            if value is not None:
                return value
    return _validation_mean_nll(feature_result.get("initial_validation"))


def _feature_result_validation_nll_mean(feature_result: dict[str, Any]) -> float | None:
    explicit = _finite_float(feature_result.get("validation_nll_mean"))
    if explicit is not None:
        return explicit
    values = _feature_result_validation_nlls(feature_result)
    return sum(values) / len(values) if values else None


def _feature_result_initial_minus_best(feature_result: dict[str, Any]) -> float | None:
    explicit = _finite_float(feature_result.get("validation_nll_initial_minus_best"))
    if explicit is not None:
        return explicit
    initial = _validation_mean_nll(feature_result.get("initial_validation"))
    best = _feature_result_best_nll(feature_result)
    return initial - best if initial is not None and best is not None else None


def _feature_result_final_minus_best(feature_result: dict[str, Any]) -> float | None:
    explicit = _finite_float(feature_result.get("validation_nll_final_minus_best"))
    if explicit is not None:
        return explicit
    final = _feature_result_final_nll(feature_result)
    best = _feature_result_best_nll(feature_result)
    return final - best if final is not None and best is not None else None


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return "-"
    return f"{parsed * 100.0:.{digits}f}%"


def _fmt_count_rate(count: Any, total: int) -> str:
    parsed = _finite_float(count)
    if parsed is None or total <= 0:
        return "-"
    return f"{int(parsed)}/{total} ({_fmt_percent(parsed / float(total), 1)})"


def _fmt_count_map(counts: Any) -> str:
    if not isinstance(counts, dict):
        return "-"
    rows = []
    for key, value in sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    ):
        rows.append(f"{key}={int(value)}")
    return ", ".join(rows) if rows else "-"


def _fmt_stat_mean(stats: dict[str, Any] | None, digits: int = 6) -> str:
    if not stats:
        return "-"
    return _fmt_float(stats.get("mean"), digits)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_" + "No rows." + "_"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _single_report(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    ranking_rows = []
    feature_results = {
        str(item.get("feature")): item
        for item in summary.get("features", [])
    }
    for item in summary.get("ranking", []):
        feature = str(item.get("feature"))
        result = feature_results.get(feature, {})
        initial = result.get("initial_validation", {})
        final = result.get("final_validation", {})
        ranking_rows.append(
            [
                feature,
                str(item.get("best_epoch") if item.get("best_epoch") is not None else "init"),
                _fmt_float(item.get("best_mean_nll")),
                _fmt_percent(item.get("best_accuracy")),
                _fmt_float(initial.get("mean_nll")),
                _fmt_float(final.get("mean_nll")),
                _fmt_float(result.get("validation_nll_mean")),
                _fmt_float(result.get("validation_nll_final_minus_best")),
            ]
        )

    diagnostics_rows = []
    diagnostics = summary.get("feature_diagnostics", {}).get("features", {})
    for feature, item in sorted(diagnostics.items()):
        diagnostics_rows.append(
            [
                str(feature),
                str(item.get("dims", "-")),
                _fmt_float(item.get("variance_mean")),
                _fmt_percent(item.get("active_dim_fraction")),
                _fmt_float(item.get("norm_mean")),
                _fmt_float(item.get("raw_mse_mean")),
                _fmt_float(item.get("raw_cosine_mean")),
            ]
        )

    lines = [
        "# Char VAE Context Report",
        "",
        f"- status: single-run",
        f"- best_feature: {summary.get('best_feature')}",
        f"- seed: {run.get('seed')}",
        f"- features: {', '.join(str(item) for item in run.get('features', []))}",
        f"- feature_normalize: {run.get('feature_normalize')}",
        f"- hybrid_latent_scale: {run.get('hybrid_latent_scale')}",
        f"- head_init: {run.get('head_init')}",
        f"- window_chars: {run.get('window_chars')}",
        f"- latent_dim: {run.get('latent_dim')}",
        f"- summary_json: `summary.json`",
        "",
        "## Ranking",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "feature",
                "best_epoch",
                "best_nll",
                "best_acc",
                "init_nll",
                "final_nll",
                "curve_nll",
                "final_gap",
            ],
            ranking_rows,
        )
    )
    lines.extend(["", "## Feature Diagnostics", ""])
    lines.extend(
        _markdown_table(
            ["feature", "dims", "var_mean", "active_dims", "norm_mean", "raw_mse", "raw_cosine"],
            diagnostics_rows,
        )
    )
    lines.append("")
    return "\n".join(lines)


def _aggregate_report(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    config_summaries = summary.get("config_summaries") or summary.get("scale_summaries", [])
    seed_total = len(summary.get("seed_summaries", []))
    if not seed_total:
        seed_count = int(run.get("seed_count") or 0)
        config_count = int(run.get("config_count") or len(config_summaries) or 1)
        seed_total = int(run.get("run_count") or seed_count * config_count)
    show_normalize_column = int(run.get("normalize_count") or 1) > 1
    show_scale_column = int(run.get("scale_count") or 1) > 1
    best_config = summary.get("best_config") or {}
    ranking_rows = [
        [
            str(item.get("feature")),
            _fmt_float(item.get("mean_best_nll")),
            _fmt_percent(item.get("mean_best_accuracy")),
            _fmt_float(item.get("mean_best_nll_delta_vs_raw")),
            _fmt_float(item.get("mean_best_step"), 2),
            _fmt_float(item.get("mean_validation_nll_mean")),
            _fmt_float(item.get("mean_validation_nll_mean_delta_vs_raw")),
            _fmt_float(item.get("mean_validation_nll_final_minus_best")),
            str(item.get("runs", "-")),
        ]
        for item in summary.get("ranking", [])
    ]
    stability_rows = [
        [
            str(item.get("feature")),
            _fmt_count_rate(item.get("win_count"), seed_total),
            _fmt_count_rate(item.get("near_win_count"), seed_total),
            _fmt_float(item.get("mean_rank"), 2),
            _fmt_float(item.get("mean_gap_to_winner")),
        ]
        for item in summary.get("feature_stability", [])
    ]
    family_rows = [
        [
            str(item.get("family")),
            _fmt_float(item.get("mean_best_nll")),
            _fmt_percent(item.get("mean_best_accuracy")),
            _fmt_float(item.get("mean_best_nll_delta_vs_raw")),
            _fmt_count_rate(item.get("win_count"), seed_total),
            _fmt_count_rate(item.get("near_win_count"), seed_total),
            _fmt_float(item.get("mean_rank"), 2),
            _fmt_float(item.get("mean_gap_to_winner")),
            _fmt_count_map(item.get("member_best_counts")),
        ]
        for item in summary.get("feature_family_stability", [])
    ]
    config_rows = []
    for item in config_summaries:
        top = item.get("ranking", [{}])[0] if item.get("ranking") else {}
        stability_by_feature = {
            stability.get("feature"): stability
            for stability in item.get("feature_stability", [])
        }
        best_stability = stability_by_feature.get(item.get("best_feature"), {})
        config_seed_total = int(item.get("seed_count") or seed_total)
        config_rows.append(
            [
                str(item.get("feature_normalize", run.get("feature_normalize"))),
                _fmt_float(item.get("hybrid_latent_scale"), 3),
                str(item.get("status")),
                str(item.get("best_feature")),
                _fmt_float(top.get("mean_best_nll")),
                _fmt_float(top.get("mean_best_nll_delta_vs_raw")),
                _fmt_count_rate(best_stability.get("win_count"), config_seed_total),
                f"`{item.get('run_dir')}`",
            ]
        )
    diagnostics_rows = []
    for item in summary.get("feature_diagnostics_summary", []):
        diagnostics_rows.append(
            [
                str(item.get("feature")),
                _fmt_stat_mean(item.get("dims"), 2),
                _fmt_stat_mean(item.get("variance_mean")),
                _fmt_percent(item.get("active_dim_fraction", {}).get("mean")),
                _fmt_stat_mean(item.get("norm_mean")),
                _fmt_stat_mean(item.get("raw_mse_mean")),
                _fmt_stat_mean(item.get("raw_cosine_mean")),
            ]
        )
    winner_by_seed = {
        (item.get("feature_normalize"), item.get("hybrid_latent_scale"), item.get("seed")): item
        for item in summary.get("seed_winners", [])
    }
    seed_rows = []
    for item in summary.get("seed_summaries", []):
        top = item.get("ranking", [{}])[0] if item.get("ranking") else {}
        winner = winner_by_seed.get(
            (
                item.get("feature_normalize"),
                item.get("hybrid_latent_scale"),
                item.get("seed"),
            ),
            {},
        )
        near_winners = winner.get("near_winners", [])
        row = []
        if show_normalize_column:
            row.append(str(item.get("feature_normalize", "-")))
        if show_scale_column:
            row.append(_fmt_float(item.get("hybrid_latent_scale"), 3))
        row.extend(
            [
                str(item.get("seed")),
                str(item.get("best_feature")),
                _fmt_float(top.get("best_mean_nll")),
                _fmt_percent(top.get("best_accuracy")),
                _fmt_float(winner.get("margin_to_runner_up")),
                ", ".join(str(feature) for feature in near_winners) if near_winners else "-",
                f"`{item.get('run_dir')}`",
            ]
        )
        seed_rows.append(row)

    lines = [
        "# Char VAE Context Sweep Report",
        "",
        f"- status: {summary.get('status')}",
        f"- best_feature: {summary.get('best_feature')}",
        (
            "- best_config: {feature} @ normalize={normalize} scale={scale}".format(
                feature=best_config.get("best_feature"),
                normalize=best_config.get("feature_normalize"),
                scale=best_config.get("hybrid_latent_scale"),
            )
            if best_config
            else "- best_config: -"
        ),
        f"- seeds: {', '.join(str(seed) for seed in run.get('seeds', []))}",
        f"- features: {', '.join(str(item) for item in run.get('features', []))}",
        f"- feature_normalize: {run.get('feature_normalize')}",
        f"- feature_normalize_modes: {', '.join(str(mode) for mode in run.get('feature_normalize_modes', [])) or '-'}",
        f"- hybrid_latent_scale: {run.get('hybrid_latent_scale')}",
        f"- hybrid_latent_scales: {', '.join(str(scale) for scale in run.get('hybrid_latent_scales', [])) or '-'}",
        f"- head_init: {run.get('head_init')}",
        f"- min_nll_delta: {run.get('min_nll_delta')}",
        f"- follow_up_confirm_tolerance: {run.get('follow_up_confirm_tolerance')}",
        f"- win_tolerance: {run.get('win_tolerance')}",
        f"- summary_json: `summary.json`",
        "",
    ]
    follow_up = summary.get("follow_up")
    follow_up_result = summary.get("follow_up_result")
    follow_up_chain = summary.get("follow_up_chain")
    follow_up_ancestors = summary.get("follow_up_ancestors")
    follow_up_trajectory = summary.get("follow_up_trajectory")
    follow_up_gate = summary.get("follow_up_gate")
    follow_up_guidance = summary.get("follow_up_guidance")
    best_generation_follow_up = summary.get("best_generation_follow_up_command")
    broadened_follow_up = summary.get("broadened_follow_up_command")
    if isinstance(follow_up_result, dict):
        source_config = follow_up_result.get("source_best_config")
        evaluated_config = follow_up_result.get("evaluated_config")
        source_feature_eval = follow_up_result.get("source_feature_evaluated")
        current_best = follow_up_result.get("current_best_config")
        lines.extend(
            [
                "## Follow-Up Result",
                "",
                f"- source: `{follow_up_result.get('source_summary_path')}`",
                f"- verdict: {follow_up_result.get('verdict')}",
                f"- config_verdict: {follow_up_result.get('config_verdict')}",
                f"- source_feature_verdict: {follow_up_result.get('source_feature_verdict')}",
                f"- source_feature_raw_verdict: {follow_up_result.get('source_feature_raw_verdict')}",
                f"- current_best_raw_verdict: {follow_up_result.get('current_best_raw_verdict')}",
                f"- source_best_feature_retained: {follow_up_result.get('source_best_feature_retained')}",
                f"- match_found: {follow_up_result.get('match_found')}",
                "- effective_source_feature_min_nll_delta: "
                f"{_fmt_float(follow_up_result.get('effective_source_feature_min_nll_delta'))}",
                "- source_feature_mean_best_nll_stderr: "
                f"{_fmt_float(follow_up_result.get('source_feature_mean_best_nll_stderr'))}",
                f"- run_budget_shifted: {follow_up_result.get('run_budget_shifted')}",
                "- run_budget_shift: "
                f"{_run_budget_shift_label(follow_up_result.get('run_budget_shift'))}",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                [
                    "source_config",
                    "evaluated_config",
                    "source_feature_eval",
                    "current_best_config",
                    "source_nll",
                    "evaluated_nll",
                    "delta_vs_source",
                    "source_feature_delta",
                    "source_feature_delta_vs_raw",
                    "current_best_delta_vs_raw",
                ],
                [
                    [
                        _config_label(source_config if isinstance(source_config, dict) else None),
                        _config_label(
                            evaluated_config if isinstance(evaluated_config, dict) else None
                        ),
                        _config_label(
                            source_feature_eval if isinstance(source_feature_eval, dict) else None
                        ),
                        _config_label(current_best if isinstance(current_best, dict) else None),
                        _fmt_float(
                            source_config.get("mean_best_nll")
                            if isinstance(source_config, dict)
                            else None
                        ),
                        _fmt_float(
                            evaluated_config.get("mean_best_nll")
                            if isinstance(evaluated_config, dict)
                            else None
                        ),
                        _fmt_float(follow_up_result.get("mean_best_nll_delta_vs_source")),
                        _fmt_float(
                            follow_up_result.get(
                                "source_feature_mean_best_nll_delta_vs_source"
                            )
                        ),
                        _fmt_float(
                            follow_up_result.get(
                                "source_feature_mean_best_nll_delta_vs_raw"
                            )
                        ),
                        _fmt_float(
                            follow_up_result.get(
                                "current_best_mean_best_nll_delta_vs_raw"
                            )
                        ),
                    ]
                ],
            )
        )
        lines.append("")
    elif isinstance(follow_up, dict):
        lines.extend(
            [
                "## Follow-Up Source",
                "",
                f"- source: `{follow_up.get('source_summary_path')}`",
                f"- source_best_config: {_config_label(follow_up.get('source_best_config'))}",
                "",
            ]
        )
    if isinstance(follow_up_chain, dict):
        ancestors = follow_up_chain.get("ancestors", [])
        verdict_history = follow_up_chain.get("verdict_history", [])
        verdict_history_text = (
            ", ".join(str(item) for item in verdict_history)
            if isinstance(verdict_history, list)
            else "-"
        )
        ancestors_text = (
            ", ".join(f"`{item}`" for item in ancestors)
            if isinstance(ancestors, list)
            else "-"
        )
        lines.extend(
            [
                "## Follow-Up Chain",
                "",
                f"- generation: {follow_up_chain.get('generation')}",
                f"- parent_summary: `{follow_up_chain.get('parent_summary_path')}`",
                f"- latest_verdict: {follow_up_chain.get('latest_verdict')}",
                f"- improved_streak: {follow_up_chain.get('improved_streak')}",
                f"- regressed_streak: {follow_up_chain.get('regressed_streak')}",
                f"- verdict_history: {verdict_history_text}",
                f"- ancestors: {ancestors_text}",
                "",
            ]
        )
    if isinstance(follow_up_ancestors, dict):
        ancestor_records = follow_up_ancestors.get("ancestors", [])
        if isinstance(ancestor_records, list) and ancestor_records:
            lines.extend(
                [
                    "## Follow-Up Ancestors",
                    "",
                ]
            )
            lines.extend(
                _markdown_table(
                    [
                        "generation",
                        "summary_path",
                        "status",
                        "best_config",
                        "mean_best_nll",
                        "verdict",
                        "guidance_action",
                        "guided_enabled",
                        "missing",
                    ],
                    [
                        _follow_up_ancestor_row(record)
                        for record in ancestor_records
                        if isinstance(record, dict)
                    ],
                )
            )
            lines.append("")
    if isinstance(follow_up_trajectory, dict):
        verdict_counts = follow_up_trajectory.get("verdict_counts", {})
        verdict_counts_text = (
            ", ".join(f"{key}={value}" for key, value in verdict_counts.items())
            if isinstance(verdict_counts, dict)
            else "-"
        )
        raw_positive_count = follow_up_trajectory.get("raw_positive_count")
        raw_evidence_count = follow_up_trajectory.get("raw_evidence_count")
        raw_positive_rate = _finite_float(follow_up_trajectory.get("raw_positive_rate"))
        raw_positive_text = (
            f"{raw_positive_count}/{raw_evidence_count} ({raw_positive_rate * 100.0:.1f}%)"
            if raw_positive_count is not None
            and raw_evidence_count
            and raw_positive_rate is not None
            else "-"
        )
        trajectory_reasons = follow_up_trajectory.get("trajectory_reasons", [])
        trajectory_reasons_text = (
            ", ".join(str(reason) for reason in trajectory_reasons)
            if isinstance(trajectory_reasons, list)
            else "-"
        )
        points = follow_up_trajectory.get("points", [])
        lines.extend(
            [
                "## Follow-Up Trajectory",
                "",
                f"- trajectory_verdict: {follow_up_trajectory.get('trajectory_verdict')}",
                f"- trajectory_action: {follow_up_trajectory.get('trajectory_action')}",
                f"- trajectory_reasons: {trajectory_reasons_text}",
                f"- latest_verdict: {follow_up_trajectory.get('latest_verdict')}",
                "- cumulative_mean_best_nll_delta: "
                f"{_fmt_float(follow_up_trajectory.get('cumulative_mean_best_nll_delta'))}",
                f"- raw_positive_points: {raw_positive_text}",
                "- raw_positive_streak: "
                f"{follow_up_trajectory.get('raw_positive_streak')}",
                "- raw_negative_streak: "
                f"{follow_up_trajectory.get('raw_negative_streak')}",
                "- mean_raw_delta_vs_raw: "
                f"{_fmt_float(follow_up_trajectory.get('mean_raw_delta_vs_raw'))}",
                "- current_raw_delta_vs_raw: "
                f"{_fmt_float(follow_up_trajectory.get('current_raw_delta_vs_raw'))}",
                "- best_raw_delta_vs_raw: "
                f"{_fmt_float(follow_up_trajectory.get('best_raw_delta_vs_raw'))}",
                "- start_mean_best_nll: "
                f"{_fmt_float(follow_up_trajectory.get('start_mean_best_nll'))}",
                "- current_mean_best_nll: "
                f"{_fmt_float(follow_up_trajectory.get('current_mean_best_nll'))}",
                "- best_mean_best_nll: "
                f"{_fmt_float(follow_up_trajectory.get('best_mean_best_nll'))}",
                f"- best_generation: {follow_up_trajectory.get('best_generation')}",
                f"- best_summary: `{follow_up_trajectory.get('best_summary_path') or '-'}`",
                "- source_feature_tradeoff: "
                f"{follow_up_trajectory.get('source_feature_tradeoff')}",
                f"- best_feature_changed: {follow_up_trajectory.get('best_feature_changed')}",
                f"- unsafe_promotion: {follow_up_trajectory.get('unsafe_promotion')}",
                f"- verdict_counts: {verdict_counts_text}",
                "",
            ]
        )
        if isinstance(points, list) and points:
            lines.extend(
                _markdown_table(
                    [
                        "generation",
                        "role",
                        "best_config",
                        "mean_best_nll",
                        "delta_from_previous",
                        "raw_delta_vs_raw",
                        "raw_positive",
                        "verdict",
                        "guidance_action",
                        "gate_failed",
                    ],
                    [
                        _follow_up_trajectory_point_row(point)
                        for point in points
                        if isinstance(point, dict)
                    ],
                )
            )
            lines.append("")
    if isinstance(follow_up_gate, dict):
        fail_on = follow_up_gate.get("fail_on_verdicts", [])
        fail_on_text = (
            ", ".join(str(item) for item in fail_on)
            if isinstance(fail_on, list)
            else "-"
        )
        effective_verdict = follow_up_gate.get("effective_verdict")
        verdict_basis = follow_up_gate.get("verdict_basis")
        lines.extend(
            [
                "## Follow-Up Gate",
                "",
                f"- verdict: {follow_up_gate.get('verdict')}",
                f"- effective_verdict: {effective_verdict or follow_up_gate.get('verdict')}",
                f"- verdict_basis: {verdict_basis or 'verdict'}",
                f"- fail_on_verdicts: {fail_on_text}",
                f"- failed: {follow_up_gate.get('failed')}",
                f"- exit_code: {follow_up_gate.get('exit_code')}",
                "",
            ]
        )
    if isinstance(follow_up_guidance, dict):
        reasons = follow_up_guidance.get("reasons", [])
        reasons_text = (
            ", ".join(str(item) for item in reasons)
            if isinstance(reasons, list)
            else "-"
        )
        lines.extend(
            [
                "## Follow-Up Guidance",
                "",
                f"- action: {follow_up_guidance.get('action')}",
                f"- local_action: {follow_up_guidance.get('local_action') or '-'}",
                f"- trajectory_action: {follow_up_guidance.get('trajectory_action') or '-'}",
                f"- trajectory_verdict: {follow_up_guidance.get('trajectory_verdict') or '-'}",
                f"- unsafe_promotion: {follow_up_guidance.get('unsafe_promotion')}",
                f"- promote_current_best: {follow_up_guidance.get('promote_current_best')}",
                "- use_next_follow_up_command: "
                f"{follow_up_guidance.get('use_next_follow_up_command')}",
                "- use_best_generation_follow_up_command: "
                f"{follow_up_guidance.get('use_best_generation_follow_up_command')}",
                "- use_broadened_follow_up_command: "
                f"{follow_up_guidance.get('use_broadened_follow_up_command')}",
                f"- gate_failed: {follow_up_guidance.get('gate_failed')}",
                f"- reasons: {reasons_text}",
                f"- command_usage: `{follow_up_guidance.get('command_usage') or '-'}`",
                "",
            ]
        )
    if config_rows:
        lines.extend(["## Context Config Grid", ""])
        lines.extend(
            _markdown_table(
                [
                    "normalize",
                    "scale",
                    "status",
                    "best_feature",
                    "mean_best_nll",
                    "mean_delta_vs_raw",
                    "wins",
                    "run_dir",
                ],
                config_rows,
            )
        )
        lines.extend(["", "## Aggregate Ranking", ""])
    else:
        lines.extend(["## Aggregate Ranking", ""])
    lines.extend(
        _markdown_table(
            [
                "feature",
                "mean_best_nll",
                "mean_best_acc",
                "mean_delta_vs_raw",
                "mean_best_step",
                "curve_nll",
                "curve_delta_vs_raw",
                "final_gap",
                "runs",
            ],
            ranking_rows,
        )
    )
    lines.extend(["", "## Feature Stability", ""])
    lines.extend(
        _markdown_table(
            ["feature", "wins_or_ties", "near_wins", "mean_rank", "mean_gap_to_winner"],
            stability_rows,
        )
    )
    lines.extend(["", "## Feature Family Stability", ""])
    lines.extend(
        _markdown_table(
            [
                "family",
                "mean_best_nll",
                "mean_best_acc",
                "mean_delta_vs_raw",
                "wins_or_ties",
                "near_wins",
                "mean_rank",
                "mean_gap_to_winner",
                "best_members",
            ],
            family_rows,
        )
    )
    lines.extend(["", "## Aggregate Feature Diagnostics", ""])
    lines.extend(
        _markdown_table(
            ["feature", "dims", "var_mean", "active_dims", "norm_mean", "raw_mse", "raw_cosine"],
            diagnostics_rows,
        )
    )
    lines.extend(["", "## Seed Runs", ""])
    seed_headers = [
        "seed",
        "best_feature",
        "best_nll",
        "best_acc",
        "runner_up_margin",
        "near_winners",
        "run_dir",
    ]
    if show_scale_column:
        seed_headers = ["scale", *seed_headers]
    if show_normalize_column:
        seed_headers = ["normalize", *seed_headers]
    lines.extend(
        _markdown_table(
            seed_headers,
            seed_rows,
        )
    )
    next_follow_up = summary.get("next_follow_up_command")
    if isinstance(next_follow_up, dict):
        lines.extend(
            [
                "",
                "## Next Follow-Up Command",
                "",
                f"- action: {next_follow_up.get('action')}",
                f"- default_follow_up_from: `{next_follow_up.get('default_follow_up_from')}`",
                "- default_follow_up_fail_on_verdict: "
                f"{next_follow_up.get('default_follow_up_fail_on_verdict') or '-'}",
                f"- default_new_seeds: {next_follow_up.get('default_new_seeds')}",
                "- used_seed_history: "
                f"{', '.join(str(seed) for seed in next_follow_up.get('used_seed_history', [])) or '-'}",
                f"- script: `{next_follow_up.get('script_path')}`",
                f"- usage: `{next_follow_up.get('script_usage')}`",
                "",
                "```bash",
                str(next_follow_up.get("shell_command")),
                "```",
            ]
        )
    if isinstance(broadened_follow_up, dict):
        family_focus = broadened_follow_up.get("feature_family_focus")
        focus_family = (
            family_focus.get("family")
            if isinstance(family_focus, dict)
            else None
        )
        focus_added = (
            family_focus.get("added_features", [])
            if isinstance(family_focus, dict)
            else []
        )
        lines.extend(
            [
                "",
                "## Broadened Follow-Up Command",
                "",
                f"- action: {broadened_follow_up.get('action')}",
                f"- default_follow_up_from: `{broadened_follow_up.get('default_follow_up_from')}`",
                "- default_follow_up_fail_on_verdict: "
                f"{broadened_follow_up.get('default_follow_up_fail_on_verdict') or '-'}",
                f"- default_new_seeds: {broadened_follow_up.get('default_new_seeds')}",
                "- used_seed_history: "
                f"{', '.join(str(seed) for seed in broadened_follow_up.get('used_seed_history', [])) or '-'}",
                f"- feature_family_focus: {focus_family or '-'}",
                "- focused_features: "
                f"{', '.join(str(feature) for feature in broadened_follow_up.get('focused_features', [])) or '-'}",
                "- features_added_for_family: "
                f"{', '.join(str(feature) for feature in focus_added) or '-'}",
                "- broadened_epochs/batches/eval_samples: "
                f"{broadened_follow_up.get('broadened_epochs')}/"
                f"{broadened_follow_up.get('broadened_batches')}/"
                f"{broadened_follow_up.get('broadened_eval_samples')}",
                "- broadened_vae_epochs/batches: "
                f"{broadened_follow_up.get('broadened_vae_epochs')}/"
                f"{broadened_follow_up.get('broadened_vae_batches')}",
                f"- script: `{broadened_follow_up.get('script_path')}`",
                f"- usage: `{broadened_follow_up.get('script_usage')}`",
                "",
                "```bash",
                str(broadened_follow_up.get("shell_command")),
                "```",
            ]
        )
    if isinstance(best_generation_follow_up, dict):
        lines.extend(
            [
                "",
                "## Best Generation Follow-Up Command",
                "",
                f"- action: {best_generation_follow_up.get('action')}",
                f"- best_generation: {best_generation_follow_up.get('best_generation')}",
                f"- best_summary: `{best_generation_follow_up.get('best_summary_path')}`",
                "- source_budget_matched: "
                f"{best_generation_follow_up.get('source_budget_matched')}",
                "- command_run_budget: "
                f"{_run_budget_label(best_generation_follow_up.get('command_run_budget'))}",
                "- default_follow_up_from: "
                f"`{best_generation_follow_up.get('default_follow_up_from')}`",
                "- default_follow_up_fail_on_verdict: "
                f"{best_generation_follow_up.get('default_follow_up_fail_on_verdict') or '-'}",
                "- default_new_seeds: "
                f"{best_generation_follow_up.get('default_new_seeds')}",
                "- used_seed_history: "
                f"{', '.join(str(seed) for seed in best_generation_follow_up.get('used_seed_history', [])) or '-'}",
                f"- script: `{best_generation_follow_up.get('script_path')}`",
                f"- usage: `{best_generation_follow_up.get('script_usage')}`",
                "",
                "```bash",
                str(best_generation_follow_up.get("shell_command")),
                "```",
            ]
        )
    guided_next_follow_up = summary.get("guided_next_follow_up_command")
    if isinstance(guided_next_follow_up, dict):
        reasons = guided_next_follow_up.get("reasons", [])
        reasons_text = (
            ", ".join(str(item) for item in reasons)
            if isinstance(reasons, list)
            else "-"
        )
        lines.extend(
            [
                "",
                "## Guided Next Follow-Up Command",
                "",
                f"- enabled: {guided_next_follow_up.get('enabled')}",
                f"- guidance_action: {guided_next_follow_up.get('guidance_action')}",
                f"- trajectory_action: {guided_next_follow_up.get('trajectory_action') or '-'}",
                f"- unsafe_promotion: {guided_next_follow_up.get('unsafe_promotion')}",
                f"- verdict: {guided_next_follow_up.get('verdict')}",
                f"- gate_failed: {guided_next_follow_up.get('gate_failed')}",
                f"- reasons: {reasons_text}",
                "- source_script: "
                f"`{guided_next_follow_up.get('source_next_follow_up_command') or '-'}`",
                "- used_seed_history: "
                f"{', '.join(str(seed) for seed in guided_next_follow_up.get('used_seed_history', [])) or '-'}",
                f"- script: `{guided_next_follow_up.get('script_path') or '-'}`",
                f"- usage: `{guided_next_follow_up.get('script_usage') or '-'}`",
            ]
        )
        shell_command = guided_next_follow_up.get("shell_command")
        if shell_command:
            lines.extend(["", "```bash", str(shell_command), "```"])
    lines.append("")
    return "\n".join(lines)


def _aggregate_summaries(
    summaries: list[dict[str, Any]],
    *,
    min_nll_delta: float,
    win_tolerance: float,
) -> dict[str, Any]:
    feature_names = sorted(
        {
            str(feature_result["feature"])
            for summary in summaries
            for feature_result in summary.get("features", [])
            if feature_result.get("feature") is not None
        }
        | {
            str(item["feature"])
            for summary in summaries
            for item in summary.get("ranking", [])
            if item.get("feature") is not None
        }
    )
    feature_rows = []
    for feature in feature_names:
        feature_results = [
            feature_result
            for summary in summaries
            for feature_result in summary.get("features", [])
            if feature_result.get("feature") == feature
        ]
        if not feature_results:
            feature_results = [
                {
                    "feature": item.get("feature"),
                    "best_epoch": item.get("best_epoch"),
                    "best_step": item.get("best_step"),
                    "best_validation": {
                        "mean_nll": item.get("best_mean_nll"),
                        "accuracy": item.get("best_accuracy"),
                    },
                }
                for summary in summaries
                for item in summary.get("ranking", [])
                if item.get("feature") == feature
            ]
        best_nlls = [
            float(feature_result["best_validation"]["mean_nll"])
            for feature_result in feature_results
            if feature_result.get("best_validation", {}).get("mean_nll") is not None
        ]
        best_accs = [
            float(feature_result["best_validation"]["accuracy"])
            for feature_result in feature_results
            if feature_result.get("best_validation", {}).get("accuracy") is not None
        ]
        best_steps = []
        for feature_result in feature_results:
            best_step = _finite_float(feature_result.get("best_step"))
            best_epoch = _finite_float(feature_result.get("best_epoch"))
            if best_step is None and best_epoch is not None:
                best_step = best_epoch + 1.0
            if best_step is not None:
                best_steps.append(best_step)
        validation_nll_means = [
            value
            for value in (
                _feature_result_validation_nll_mean(feature_result)
                for feature_result in feature_results
            )
            if value is not None
        ]
        validation_nll_initial_gains = [
            value
            for value in (
                _feature_result_initial_minus_best(feature_result)
                for feature_result in feature_results
            )
            if value is not None
        ]
        validation_nll_final_gaps = [
            value
            for value in (
                _feature_result_final_minus_best(feature_result)
                for feature_result in feature_results
            )
            if value is not None
        ]
        deltas = [
            float(summary.get("deltas", {}).get(f"{feature}_best_nll_vs_raw"))
            for summary in summaries
            if summary.get("deltas", {}).get(f"{feature}_best_nll_vs_raw") is not None
        ]
        curve_deltas = []
        for summary in summaries:
            existing_delta = _finite_float(
                summary.get("deltas", {}).get(
                    f"{feature}_validation_nll_mean_vs_raw"
                )
            )
            if existing_delta is not None:
                curve_deltas.append(existing_delta)
                continue
            feature_by_name = {
                str(item.get("feature")): item
                for item in summary.get("features", [])
                if isinstance(item, dict) and item.get("feature") is not None
            }
            raw_result = feature_by_name.get(FEATURE_RAW)
            feature_result = feature_by_name.get(feature)
            if raw_result is None or feature_result is None:
                continue
            raw_curve = _feature_result_validation_nll_mean(raw_result)
            feature_curve = _feature_result_validation_nll_mean(feature_result)
            if raw_curve is not None and feature_curve is not None:
                curve_deltas.append(feature_curve - raw_curve)
        feature_rows.append(
            {
                "feature": feature,
                "runs": len(feature_results),
                "best_nll": _metric_stats(best_nlls),
                "best_accuracy": _metric_stats(best_accs),
                "best_nll_delta_vs_raw": _metric_stats(deltas),
                "best_step": _metric_stats(best_steps),
                "validation_nll_mean": _metric_stats(validation_nll_means),
                "validation_nll_initial_minus_best": _metric_stats(
                    validation_nll_initial_gains
                ),
                "validation_nll_final_minus_best": _metric_stats(
                    validation_nll_final_gaps
                ),
                "validation_nll_mean_delta_vs_raw": _metric_stats(curve_deltas),
            }
        )
    diagnostic_rows = []
    diagnostic_keys = (
        "dims",
        "variance_mean",
        "variance_max",
        "active_dim_fraction",
        "norm_mean",
        "raw_l2_mean",
        "raw_mse_mean",
        "raw_cosine_mean",
    )
    for feature in feature_names:
        feature_diagnostics = [
            summary.get("feature_diagnostics", {}).get("features", {}).get(feature, {})
            for summary in summaries
        ]
        diagnostic_rows.append(
            {
                "feature": feature,
                **{
                    key: _metric_stats(
                        diag.get(key)
                        for diag in feature_diagnostics
                        if diag.get(key) is not None
                    )
                    for key in diagnostic_keys
                },
            }
        )

    seed_winners = []
    feature_stability_acc: dict[str, dict[str, Any]] = {
        feature: {
            "win_count": 0,
            "near_win_count": 0,
            "ranks": [],
            "gaps": [],
        }
        for feature in feature_names
    }

    def new_feature_family_stats() -> dict[str, Any]:
        return {
            "win_count": 0,
            "near_win_count": 0,
            "ranks": [],
            "gaps": [],
            "best_nlls": [],
            "best_accs": [],
            "deltas_vs_raw": [],
            "member_best_counts": {},
        }

    family_names = sorted({_feature_family(feature) for feature in feature_names})
    feature_family_stability_acc: dict[str, dict[str, Any]] = {
        family: new_feature_family_stats() for family in family_names
    }
    for summary in summaries:
        entries = []
        for item in summary.get("ranking", []):
            feature = str(item.get("feature"))
            best_nll = _finite_float(item.get("best_mean_nll"))
            if best_nll is None:
                continue
            entries.append(
                {
                    "feature": feature,
                    "best_nll": best_nll,
                    "best_accuracy": _finite_float(item.get("best_accuracy")),
                    "rank": None,
                }
            )
        entries.sort(key=lambda item: (item["best_nll"], item["feature"]))
        if not entries:
            continue

        current_rank = 0
        previous_nll: float | None = None
        for position, item in enumerate(entries, start=1):
            best_nll = float(item["best_nll"])
            if previous_nll is None or best_nll > previous_nll:
                current_rank = position
                previous_nll = best_nll
            item["rank"] = current_rank

        best = entries[0]
        strict_winners = [
            item["feature"]
            for item in entries
            if float(item["best_nll"]) - float(best["best_nll"]) <= 0.0
        ]
        runner_up = next(
            (
                item
                for item in entries
                if float(item["best_nll"]) - float(best["best_nll"]) > 0.0
            ),
            None,
        )
        near_winners = [
            item["feature"]
            for item in entries
            if float(item["best_nll"]) - float(best["best_nll"]) <= win_tolerance
        ]
        for feature in strict_winners:
            feature_stability_acc.setdefault(
                feature,
                {"win_count": 0, "near_win_count": 0, "ranks": [], "gaps": []},
            )["win_count"] += 1
        for item in entries:
            stats = feature_stability_acc.setdefault(
                item["feature"],
                {"win_count": 0, "near_win_count": 0, "ranks": [], "gaps": []},
            )
            stats["ranks"].append(float(item["rank"]))
            stats["gaps"].append(float(item["best_nll"]) - float(best["best_nll"]))
        for feature in near_winners:
            feature_stability_acc.setdefault(
                feature,
                {"win_count": 0, "near_win_count": 0, "ranks": [], "gaps": []},
            )["near_win_count"] += 1

        raw_entry = next((item for item in entries if item["feature"] == FEATURE_RAW), None)
        raw_best_nll = float(raw_entry["best_nll"]) if raw_entry is not None else None
        family_best_by_name: dict[str, dict[str, Any]] = {}
        for item in entries:
            family = _feature_family(str(item["feature"]))
            current = family_best_by_name.get(family)
            if current is None or (
                float(item["best_nll"]),
                str(item["feature"]),
            ) < (
                float(current["best_nll"]),
                str(current["feature"]),
            ):
                family_best_by_name[family] = item
        family_entries = [
            {
                "family": family,
                "feature": item["feature"],
                "best_nll": item["best_nll"],
                "best_accuracy": item["best_accuracy"],
                "rank": None,
            }
            for family, item in family_best_by_name.items()
        ]
        family_entries.sort(
            key=lambda item: (
                float(item["best_nll"]),
                str(item["family"]),
                str(item["feature"]),
            )
        )
        family_current_rank = 0
        family_previous_nll: float | None = None
        for position, item in enumerate(family_entries, start=1):
            best_nll = float(item["best_nll"])
            if family_previous_nll is None or best_nll > family_previous_nll:
                family_current_rank = position
                family_previous_nll = best_nll
            item["rank"] = family_current_rank
        if family_entries:
            family_best = family_entries[0]
            family_strict_winners = [
                str(item["family"])
                for item in family_entries
                if float(item["best_nll"]) - float(family_best["best_nll"]) <= 0.0
            ]
            family_near_winners = [
                str(item["family"])
                for item in family_entries
                if float(item["best_nll"]) - float(family_best["best_nll"]) <= win_tolerance
            ]
            for item in family_entries:
                family = str(item["family"])
                stats = feature_family_stability_acc.setdefault(
                    family,
                    new_feature_family_stats(),
                )
                stats["ranks"].append(float(item["rank"]))
                stats["gaps"].append(
                    float(item["best_nll"]) - float(family_best["best_nll"])
                )
                stats["best_nlls"].append(float(item["best_nll"]))
                if item["best_accuracy"] is not None:
                    stats["best_accs"].append(float(item["best_accuracy"]))
                if raw_best_nll is not None:
                    stats["deltas_vs_raw"].append(float(item["best_nll"]) - raw_best_nll)
                member_counts = stats["member_best_counts"]
                member = str(item["feature"])
                member_counts[member] = int(member_counts.get(member, 0)) + 1
            for family in family_strict_winners:
                feature_family_stability_acc.setdefault(
                    family,
                    new_feature_family_stats(),
                )["win_count"] += 1
            for family in family_near_winners:
                feature_family_stability_acc.setdefault(
                    family,
                    new_feature_family_stats(),
                )["near_win_count"] += 1

        seed_winners.append(
            {
                "seed": summary.get("run", {}).get("seed"),
                "feature_normalize": summary.get("run", {}).get("feature_normalize"),
                "hybrid_latent_scale": summary.get("run", {}).get("hybrid_latent_scale"),
                "winner": best["feature"],
                "winners": strict_winners,
                "near_winners": near_winners,
                "best_nll": best["best_nll"],
                "best_accuracy": best["best_accuracy"],
                "runner_up": runner_up["feature"] if runner_up is not None else None,
                "runner_up_nll": runner_up["best_nll"] if runner_up is not None else None,
                "margin_to_runner_up": (
                    float(runner_up["best_nll"]) - float(best["best_nll"])
                    if runner_up is not None
                    else None
                ),
            }
        )

    seed_count = len(summaries)
    feature_stability = []
    for feature in feature_names:
        stats = feature_stability_acc.get(
            feature,
            {"win_count": 0, "near_win_count": 0, "ranks": [], "gaps": []},
        )
        rank_stats = _metric_stats(stats.get("ranks", []))
        gap_stats = _metric_stats(stats.get("gaps", []))
        win_count = int(stats.get("win_count", 0))
        near_win_count = int(stats.get("near_win_count", 0))
        feature_stability.append(
            {
                "feature": feature,
                "win_count": win_count,
                "win_rate": win_count / float(seed_count) if seed_count else None,
                "near_win_count": near_win_count,
                "near_win_rate": near_win_count / float(seed_count) if seed_count else None,
                "rank": rank_stats,
                "mean_rank": rank_stats["mean"],
                "gap_to_winner": gap_stats,
                "mean_gap_to_winner": gap_stats["mean"],
            }
        )
    feature_stability.sort(
        key=lambda item: (
            -int(item.get("win_count", 0)),
            -int(item.get("near_win_count", 0)),
            float("inf") if item.get("mean_rank") is None else float(item["mean_rank"]),
            float("inf")
            if item.get("mean_gap_to_winner") is None
            else float(item["mean_gap_to_winner"]),
            str(item.get("feature")),
        )
    )

    feature_family_stability = []
    for family in family_names:
        stats = feature_family_stability_acc.get(
            family,
            new_feature_family_stats(),
        )
        rank_stats = _metric_stats(stats.get("ranks", []))
        gap_stats = _metric_stats(stats.get("gaps", []))
        best_nll_stats = _metric_stats(stats.get("best_nlls", []))
        best_acc_stats = _metric_stats(stats.get("best_accs", []))
        delta_stats = _metric_stats(stats.get("deltas_vs_raw", []))
        win_count = int(stats.get("win_count", 0))
        near_win_count = int(stats.get("near_win_count", 0))
        feature_family_stability.append(
            {
                "family": family,
                "win_count": win_count,
                "win_rate": win_count / float(seed_count) if seed_count else None,
                "near_win_count": near_win_count,
                "near_win_rate": near_win_count / float(seed_count) if seed_count else None,
                "rank": rank_stats,
                "mean_rank": rank_stats["mean"],
                "gap_to_winner": gap_stats,
                "mean_gap_to_winner": gap_stats["mean"],
                "best_nll": best_nll_stats,
                "mean_best_nll": best_nll_stats["mean"],
                "best_accuracy": best_acc_stats,
                "mean_best_accuracy": best_acc_stats["mean"],
                "best_nll_delta_vs_raw": delta_stats,
                "mean_best_nll_delta_vs_raw": delta_stats["mean"],
                "member_best_counts": stats.get("member_best_counts", {}),
            }
        )
    feature_family_stability.sort(
        key=lambda item: (
            -int(item.get("win_count", 0)),
            -int(item.get("near_win_count", 0)),
            float("inf") if item.get("mean_rank") is None else float(item["mean_rank"]),
            float("inf")
            if item.get("mean_gap_to_winner") is None
            else float(item["mean_gap_to_winner"]),
            str(item.get("family")),
        )
    )

    ranking = sorted(
        feature_rows,
        key=lambda item: (
            float("inf")
            if item["best_nll"]["mean"] is None
            else float(item["best_nll"]["mean"]),
            item["feature"],
        ),
    )
    best_feature = ranking[0]["feature"] if ranking else None
    non_raw_deltas = [
        row
        for row in feature_rows
        if row["feature"] != FEATURE_RAW and row["best_nll_delta_vs_raw"]["mean"] is not None
    ]
    if not non_raw_deltas:
        status = "no_raw_baseline"
    elif any(float(row["best_nll_delta_vs_raw"]["mean"]) < -min_nll_delta for row in non_raw_deltas):
        status = "improved"
    elif all(float(row["best_nll_delta_vs_raw"]["mean"]) > min_nll_delta for row in non_raw_deltas):
        status = "regression"
    else:
        status = "neutral"

    return {
        "feature_summary": feature_rows,
        "feature_diagnostics_summary": diagnostic_rows,
        "ranking": [
            {
                "feature": item["feature"],
                "mean_best_nll": item["best_nll"]["mean"],
                "mean_best_accuracy": item["best_accuracy"]["mean"],
                "mean_best_nll_delta_vs_raw": item["best_nll_delta_vs_raw"]["mean"],
                "mean_best_step": item["best_step"]["mean"],
                "mean_validation_nll_mean": item["validation_nll_mean"]["mean"],
                "mean_validation_nll_mean_delta_vs_raw": item[
                    "validation_nll_mean_delta_vs_raw"
                ]["mean"],
                "mean_validation_nll_initial_minus_best": item[
                    "validation_nll_initial_minus_best"
                ]["mean"],
                "mean_validation_nll_final_minus_best": item[
                    "validation_nll_final_minus_best"
                ]["mean"],
                "runs": item["runs"],
            }
            for item in ranking
        ],
        "seed_winners": seed_winners,
        "feature_stability": feature_stability,
        "feature_family_stability": feature_family_stability,
        "win_tolerance": win_tolerance,
        "best_feature": best_feature,
        "status": status,
    }


def _seed_run_dir(root: pathlib.Path, seed: int) -> pathlib.Path:
    if seed < 0:
        return root / f"seed_neg_{abs(seed):06d}"
    return root / f"seed_{seed:06d}"


def _scale_slug(scale: float) -> str:
    text = f"{float(scale):.8g}".replace("-", "neg_").replace(".", "p")
    return text.replace("+", "")


def _scale_run_dir(root: pathlib.Path, scale: float) -> pathlib.Path:
    return root / f"scale_{_scale_slug(scale)}"


def _normalize_run_dir(root: pathlib.Path, mode: str) -> pathlib.Path:
    return root / f"normalize_{mode}"


def _config_run_dir(
    root: pathlib.Path,
    mode: str,
    scale: float,
    normalize_modes: list[str],
    scales: list[float],
) -> pathlib.Path:
    run_dir = root
    if len(normalize_modes) > 1:
        run_dir = _normalize_run_dir(run_dir, mode)
    if len(scales) > 1:
        run_dir = _scale_run_dir(run_dir, scale)
    return run_dir


def _best_config_summary(config_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not config_summaries:
        return None
    ranked = sorted(
        config_summaries,
        key=lambda item: (
            float("inf")
            if not item.get("ranking")
            or item["ranking"][0].get("mean_best_nll") is None
            else float(item["ranking"][0]["mean_best_nll"]),
            str(item.get("feature_normalize", "")),
            float(item.get("hybrid_latent_scale", 0.0)),
        ),
    )
    best = ranked[0]
    top = best.get("ranking", [{}])[0] if best.get("ranking") else {}
    return {
        "feature_normalize": best.get("feature_normalize"),
        "hybrid_latent_scale": best.get("hybrid_latent_scale"),
        "best_feature": best.get("best_feature"),
        "status": best.get("status"),
        "mean_best_nll": top.get("mean_best_nll"),
        "mean_best_accuracy": top.get("mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": top.get("mean_best_nll_delta_vs_raw"),
        "run_dir": best.get("run_dir"),
    }


def _fmt_arg_float(value: Any) -> str:
    return f"{float(value):.8g}"


def _fresh_seed_csv(seeds: list[int], *, count: int | None = None) -> str:
    used = set(int(seed) for seed in seeds)
    candidates = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
    target_count = max(3, len(seeds)) if count is None else max(1, int(count))
    fresh = [seed for seed in candidates if seed not in used][:target_count]
    if len(fresh) < target_count:
        cursor = 1_001
        while len(fresh) < target_count:
            if cursor not in used:
                fresh.append(cursor)
            cursor += 2
    return ",".join(str(seed) for seed in fresh)


def _append_unique_ints(target: list[int], values: Any) -> None:
    if not isinstance(values, list):
        return
    seen = set(target)
    for value in values:
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if seed not in seen:
            target.append(seed)
            seen.add(seed)


def _seed_csv_values(raw: Any) -> list[int]:
    if raw is None:
        return []
    seeds: list[int] = []
    seen: set[int] = set()
    for item in str(raw).split(","):
        value = item.strip()
        if not value:
            continue
        try:
            seed = int(value)
        except ValueError:
            continue
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return seeds


def _follow_up_used_seeds(
    follow_up: dict[str, Any] | None,
    current_seeds: list[int],
) -> list[int]:
    used: list[int] = []
    if isinstance(follow_up, dict):
        source_chain = follow_up.get("source_chain")
        source_chain = source_chain if isinstance(source_chain, dict) else {}
        ancestors = source_chain.get("ancestors", [])
        if isinstance(ancestors, list):
            for raw_path in ancestors:
                if raw_path is None or not str(raw_path):
                    continue
                try:
                    _loaded_path, ancestor_summary = _load_follow_up_summary(
                        pathlib.Path(str(raw_path)).expanduser()
                    )
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                _append_unique_ints(used, _summary_seeds(ancestor_summary))
        _append_unique_ints(used, follow_up.get("external_seed_history"))
        _append_unique_ints(used, follow_up.get("source_seeds"))
        resolved = follow_up.get("resolved")
        if isinstance(resolved, dict):
            _append_unique_ints(used, resolved.get("seeds"))
    _append_unique_ints(used, current_seeds)
    return used


def _summary_seed_history(summary: dict[str, Any]) -> list[int]:
    used: list[int] = []
    next_follow_up = summary.get("next_follow_up_command")
    if isinstance(next_follow_up, dict):
        _append_unique_ints(used, next_follow_up.get("used_seed_history"))
    _append_unique_ints(
        used,
        _follow_up_used_seeds(
            {"source_chain": _follow_up_chain_source(summary)},
            _summary_seeds(summary),
        ),
    )
    return used


def _feature_family_focus_record(
    best_config: dict[str, Any] | None,
    feature_family_stability: list[dict[str, Any]] | None,
    features: list[str],
) -> dict[str, Any] | None:
    if not isinstance(best_config, dict) or not feature_family_stability:
        return None
    best_feature = str(best_config.get("best_feature") or "")
    if best_feature not in FEATURE_CHOICES:
        return None
    best_family = _feature_family(best_feature)
    top_family = next(
        (
            item
            for item in feature_family_stability
            if isinstance(item, dict) and item.get("family") is not None
        ),
        None,
    )
    if not isinstance(top_family, dict) or str(top_family.get("family")) != best_family:
        return None
    win_count = int(top_family.get("win_count") or 0)
    near_win_count = int(top_family.get("near_win_count") or 0)
    if win_count <= 0 and near_win_count <= 0:
        return None
    mean_delta_vs_raw = _finite_float(top_family.get("mean_best_nll_delta_vs_raw"))
    if (
        best_family != FEATURE_RAW
        and mean_delta_vs_raw is not None
        and mean_delta_vs_raw >= 0.0
    ):
        return None
    required_features = _feature_family_members(best_family)
    if not required_features:
        return None
    focused_features = _feature_family_focused_features(features, best_family)
    original = set(str(feature) for feature in features)
    added_features = [feature for feature in focused_features if feature not in original]
    return {
        "schema": "st.llm_char_vae_context.feature_family_focus.v1",
        "family": best_family,
        "best_feature": best_feature,
        "required_features": required_features,
        "focused_features": focused_features,
        "added_features": added_features,
        "win_count": win_count,
        "near_win_count": near_win_count,
        "win_rate": top_family.get("win_rate"),
        "near_win_rate": top_family.get("near_win_rate"),
        "mean_best_nll": top_family.get("mean_best_nll"),
        "mean_best_accuracy": top_family.get("mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": top_family.get("mean_best_nll_delta_vs_raw"),
        "mean_rank": top_family.get("mean_rank"),
        "mean_gap_to_winner": top_family.get("mean_gap_to_winner"),
        "member_best_counts": top_family.get("member_best_counts", {}),
    }


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    command.extend([flag, str(value)])


def _follow_up_command_parts(
    args: argparse.Namespace,
    features: list[str],
    best_config: dict[str, Any],
    *,
    seeds_value: str,
    run_dir_value: str,
    follow_up_from_value: str | None,
    fail_on_verdict_value: str | None,
    used_seed_history_value: str | None = None,
) -> list[str]:
    command = [
        "python3",
        "-S",
        "-s",
        "models/python/llm_char_vae_context.py",
        *[str(value) for value in args.text_or_dir],
    ]
    for flag, value in (
        ("--features", ",".join(features)),
        ("--feature-normalize", best_config.get("feature_normalize")),
        ("--hybrid-latent-scale", _fmt_arg_float(best_config.get("hybrid_latent_scale", 1.0))),
        ("--seeds", seeds_value),
        ("--run-dir", run_dir_value),
        ("--follow-up-from", follow_up_from_value),
        ("--follow-up-fail-on-verdict", fail_on_verdict_value),
        ("--follow-up-used-seeds", used_seed_history_value),
        ("--follow-up-confirm-tolerance", _fmt_arg_float(args.follow_up_confirm_tolerance)),
        ("--window-chars", int(args.window_chars)),
        ("--latent-dim", int(args.latent_dim)),
        ("--hidden", int(args.hidden)),
        ("--head-init", str(args.head_init)),
        ("--epochs", int(args.epochs)),
        ("--batches", int(args.batches)),
        ("--batch-size", int(args.batch_size)),
        ("--lr", _fmt_arg_float(args.lr)),
        ("--eval-samples", int(args.eval_samples)),
        ("--val-ratio", _fmt_arg_float(args.val_ratio)),
        ("--curvature", _fmt_arg_float(args.curvature)),
        ("--temperature", _fmt_arg_float(args.temperature)),
        ("--backend", str(args.backend)),
        ("--min-nll-delta", _fmt_arg_float(args.min_nll_delta)),
        ("--win-tolerance", _fmt_arg_float(args.win_tolerance)),
        ("--prompt", str(args.prompt)),
        ("--gen", int(args.gen)),
        ("--top-k", int(args.top_k)),
        ("--vae-epochs", int(args.vae_epochs)),
        ("--vae-batches", int(args.vae_batches)),
        ("--vae-batch-size", int(args.vae_batch_size)),
        ("--vae-lr", _fmt_arg_float(args.vae_lr)),
        ("--vae-kl-weight", _fmt_arg_float(args.vae_kl_weight)),
        ("--vae-optimizer", str(args.vae_optimizer)),
        ("--vae-grad-clip", str(args.vae_grad_clip)),
        ("--mellin", str(args.mellin)),
        ("--mellin-exponent", _fmt_arg_float(args.mellin_exponent)),
        ("--mellin-start", _fmt_arg_float(args.mellin_start)),
        ("--mellin-end", _fmt_arg_float(args.mellin_end)),
    ):
        if value is not None:
            _append_flag(command, flag, value)
    if args.vae_load is not None:
        _append_flag(command, "--vae-load", args.vae_load)
    return command


def _next_follow_up_command_record(
    args: argparse.Namespace,
    features: list[str],
    best_config: dict[str, Any],
    root_run_dir: pathlib.Path,
    seeds: list[int],
    follow_up: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not best_config:
        return None
    used_seeds = _follow_up_used_seeds(follow_up, seeds)
    default_new_seeds = _fresh_seed_csv(used_seeds or seeds, count=max(3, len(seeds)))
    used_seed_history_value = (
        ",".join(str(seed) for seed in used_seeds) if used_seeds else None
    )
    default_run_dir = root_run_dir / "follow_up_best_config"
    default_follow_up_from = root_run_dir / "summary.json"
    default_fail_on_verdict = (
        str(args.follow_up_fail_on_verdict).strip()
        if args.follow_up_fail_on_verdict is not None
        and str(args.follow_up_fail_on_verdict).strip()
        else None
    )
    script_path = root_run_dir / "next_follow_up_command.sh"
    literal_command = _follow_up_command_parts(
        args,
        features,
        best_config,
        seeds_value=default_new_seeds,
        run_dir_value=str(default_run_dir),
        follow_up_from_value=str(default_follow_up_from),
        fail_on_verdict_value=default_fail_on_verdict,
        used_seed_history_value=used_seed_history_value,
    )
    shell_command = "PYTHONNOUSERSITE=1 " + shlex.join(literal_command)
    script_command = _follow_up_command_parts(
        args,
        features,
        best_config,
        seeds_value="${NEW_SEEDS}",
        run_dir_value="${NEXT_RUN_DIR}",
        follow_up_from_value="${FOLLOW_UP_FROM}",
        fail_on_verdict_value=(
            "${FOLLOW_UP_FAIL_ON_VERDICT}"
            if default_fail_on_verdict is not None
            else None
        ),
        used_seed_history_value=used_seed_history_value,
    )
    script_usage = (
        f"FOLLOW_UP_FROM={default_follow_up_from} NEW_SEEDS={default_new_seeds} "
        f"NEXT_RUN_DIR={default_run_dir}"
    )
    if default_fail_on_verdict is not None:
        script_usage += f" FOLLOW_UP_FAIL_ON_VERDICT={default_fail_on_verdict}"
    script_usage += f" bash {script_path}"
    return {
        "schema": "st.llm_char_vae_context.next_follow_up_command.v1",
        "action": "confirm_best_config_fresh_seeds",
        "best_config": best_config,
        "default_new_seeds": default_new_seeds,
        "used_seed_history": used_seeds,
        "default_run_dir": str(default_run_dir),
        "default_follow_up_from": str(default_follow_up_from),
        "default_follow_up_fail_on_verdict": default_fail_on_verdict,
        "script_path": str(script_path),
        "shell_command": shell_command,
        "script_command": script_command,
        "script_usage": script_usage,
    }


def _guided_next_follow_up_command_record(
    root_run_dir: pathlib.Path,
    follow_up_guidance: dict[str, Any] | None,
    next_follow_up: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(follow_up_guidance, dict):
        return None

    enabled = (
        bool(
            follow_up_guidance.get("use_next_follow_up_command")
            or follow_up_guidance.get("use_best_generation_follow_up_command")
            or follow_up_guidance.get("use_broadened_follow_up_command")
        )
        and isinstance(
            next_follow_up,
            dict,
        )
    )
    reasons_raw = follow_up_guidance.get("reasons", [])
    reasons = (
        [str(reason) for reason in reasons_raw]
        if isinstance(reasons_raw, list)
        else []
    )
    record: dict[str, Any] = {
        "schema": "st.llm_char_vae_context.guided_next_follow_up_command.v1",
        "enabled": enabled,
        "guidance_action": follow_up_guidance.get("action"),
        "verdict": follow_up_guidance.get("verdict"),
        "config_verdict": follow_up_guidance.get("config_verdict"),
        "source_feature_verdict": follow_up_guidance.get("source_feature_verdict"),
        "source_feature_raw_verdict": follow_up_guidance.get(
            "source_feature_raw_verdict"
        ),
        "gate_failed": follow_up_guidance.get("gate_failed"),
        "trajectory_action": follow_up_guidance.get("trajectory_action"),
        "trajectory_verdict": follow_up_guidance.get("trajectory_verdict"),
        "unsafe_promotion": follow_up_guidance.get("unsafe_promotion"),
        "reasons": reasons,
        "source_next_follow_up_command": (
            next_follow_up.get("script_path") if isinstance(next_follow_up, dict) else None
        ),
    }
    if not enabled or not isinstance(next_follow_up, dict):
        record.update(
            {
                "script_path": None,
                "script_usage": None,
                "shell_command": None,
                "script_command": None,
            }
        )
        return record

    script_path = root_run_dir / "guided_next_follow_up_command.sh"
    script_usage = (
        f"FOLLOW_UP_FROM={next_follow_up.get('default_follow_up_from')} "
        f"NEW_SEEDS={next_follow_up.get('default_new_seeds')} "
        f"NEXT_RUN_DIR={next_follow_up.get('default_run_dir')}"
    )
    fail_on_verdict = next_follow_up.get("default_follow_up_fail_on_verdict")
    if fail_on_verdict is not None:
        script_usage += f" FOLLOW_UP_FAIL_ON_VERDICT={fail_on_verdict}"
    script_usage += f" bash {script_path}"

    record.update(
        {
            "default_follow_up_from": next_follow_up.get("default_follow_up_from"),
            "default_new_seeds": next_follow_up.get("default_new_seeds"),
            "used_seed_history": next_follow_up.get("used_seed_history", []),
            "default_run_dir": next_follow_up.get("default_run_dir"),
            "default_follow_up_fail_on_verdict": fail_on_verdict,
            "script_path": str(script_path),
            "script_usage": script_usage,
            "shell_command": next_follow_up.get("shell_command"),
            "script_command": next_follow_up.get("script_command"),
        }
    )
    return record


def _best_generation_follow_up_command_record(
    args: argparse.Namespace,
    features: list[str],
    root_run_dir: pathlib.Path,
    seeds: list[int],
    follow_up_trajectory: dict[str, Any] | None,
    next_follow_up: dict[str, Any] | None,
    follow_up_result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(follow_up_trajectory, dict):
        return None
    if (
        follow_up_trajectory.get("trajectory_action")
        != "reconfirm_best_raw_positive_generation"
    ):
        return None
    best_config = follow_up_trajectory.get("best_config")
    best_summary_path = follow_up_trajectory.get("best_summary_path")
    if not isinstance(best_config, dict) or not best_summary_path:
        return None

    used_seeds: list[int] = []
    if isinstance(next_follow_up, dict):
        _append_unique_ints(used_seeds, next_follow_up.get("used_seed_history"))
    _append_unique_ints(used_seeds, seeds)
    default_new_seeds = _fresh_seed_csv(used_seeds or seeds, count=max(3, len(seeds)))
    used_seed_history_value = (
        ",".join(str(seed) for seed in used_seeds) if used_seeds else None
    )
    default_run_dir = root_run_dir / "follow_up_best_generation"
    default_follow_up_from = pathlib.Path(str(best_summary_path)).expanduser()
    default_fail_on_verdict = (
        str(args.follow_up_fail_on_verdict).strip()
        if args.follow_up_fail_on_verdict is not None
        and str(args.follow_up_fail_on_verdict).strip()
        else None
    )
    source_run_budget = (
        follow_up_result.get("source_run_budget")
        if isinstance(follow_up_result, dict)
        and isinstance(follow_up_result.get("source_run_budget"), dict)
        else {}
    )
    source_budget_matched = bool(
        isinstance(follow_up_result, dict)
        and follow_up_result.get("run_budget_shifted")
        and source_run_budget
    )
    command_args = (
        _args_with_run_budget(args, source_run_budget)
        if source_budget_matched
        else args
    )
    script_path = root_run_dir / "best_generation_follow_up_command.sh"
    literal_command = _follow_up_command_parts(
        command_args,
        features,
        best_config,
        seeds_value=default_new_seeds,
        run_dir_value=str(default_run_dir),
        follow_up_from_value=str(default_follow_up_from),
        fail_on_verdict_value=default_fail_on_verdict,
        used_seed_history_value=used_seed_history_value,
    )
    shell_command = "PYTHONNOUSERSITE=1 " + shlex.join(literal_command)
    script_command = _follow_up_command_parts(
        command_args,
        features,
        best_config,
        seeds_value="${NEW_SEEDS}",
        run_dir_value="${NEXT_RUN_DIR}",
        follow_up_from_value="${FOLLOW_UP_FROM}",
        fail_on_verdict_value=(
            "${FOLLOW_UP_FAIL_ON_VERDICT}"
            if default_fail_on_verdict is not None
            else None
        ),
        used_seed_history_value=used_seed_history_value,
    )
    script_usage = (
        f"FOLLOW_UP_FROM={default_follow_up_from} NEW_SEEDS={default_new_seeds} "
        f"NEXT_RUN_DIR={default_run_dir}"
    )
    if default_fail_on_verdict is not None:
        script_usage += f" FOLLOW_UP_FAIL_ON_VERDICT={default_fail_on_verdict}"
    script_usage += f" bash {script_path}"
    return {
        "schema": "st.llm_char_vae_context.best_generation_follow_up_command.v1",
        "action": "reconfirm_best_raw_positive_generation",
        "best_config": best_config,
        "best_generation": follow_up_trajectory.get("best_generation"),
        "best_summary_path": str(default_follow_up_from),
        "source_budget_matched": source_budget_matched,
        "source_run_budget": source_run_budget,
        "command_run_budget": _args_run_budget(command_args),
        "default_new_seeds": default_new_seeds,
        "used_seed_history": used_seeds,
        "default_run_dir": str(default_run_dir),
        "default_follow_up_from": str(default_follow_up_from),
        "default_follow_up_fail_on_verdict": default_fail_on_verdict,
        "script_path": str(script_path),
        "shell_command": shell_command,
        "script_command": script_command,
        "script_usage": script_usage,
    }


def _broadened_follow_up_command_record(
    args: argparse.Namespace,
    features: list[str],
    best_config: dict[str, Any] | None,
    root_run_dir: pathlib.Path,
    seeds: list[int],
    follow_up_chain: dict[str, Any] | None,
    follow_up_trajectory: dict[str, Any] | None,
    next_follow_up: dict[str, Any] | None,
    feature_family_stability: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(best_config, dict):
        return None
    if not isinstance(follow_up_chain, dict) or not isinstance(follow_up_trajectory, dict):
        return None
    if int(follow_up_chain.get("improved_streak") or 0) < 2:
        return None
    if follow_up_trajectory.get("trajectory_action") != "confirm_trajectory_with_fresh_seeds":
        return None
    if follow_up_trajectory.get("unsafe_promotion"):
        return None
    if follow_up_trajectory.get("best_summary_path") != str(root_run_dir / "summary.json"):
        return None

    used_seeds: list[int] = []
    if isinstance(next_follow_up, dict):
        _append_unique_ints(used_seeds, next_follow_up.get("used_seed_history"))
    _append_unique_ints(used_seeds, seeds)
    default_new_seeds = _fresh_seed_csv(used_seeds or seeds, count=max(5, len(seeds) + 2))
    used_seed_history_value = (
        ",".join(str(seed) for seed in used_seeds) if used_seeds else None
    )
    default_run_dir = root_run_dir / "follow_up_broadened"
    default_follow_up_from = root_run_dir / "summary.json"
    default_fail_on_verdict = (
        str(args.follow_up_fail_on_verdict).strip()
        if args.follow_up_fail_on_verdict is not None
        and str(args.follow_up_fail_on_verdict).strip()
        else None
    )
    broadened_args = _clone_args(
        args,
        epochs=max(int(args.epochs) + 1, int(args.epochs) * 2),
        batches=max(int(args.batches) + 1, int(args.batches) * 2),
        eval_samples=max(int(args.eval_samples) + 16, int(args.eval_samples) * 2),
        vae_epochs=max(int(args.vae_epochs) + 1, int(args.vae_epochs) * 2),
        vae_batches=max(int(args.vae_batches) + 1, int(args.vae_batches) * 2),
    )
    family_focus = _feature_family_focus_record(
        best_config,
        feature_family_stability,
        features,
    )
    focused_features = (
        list(family_focus["focused_features"])
        if isinstance(family_focus, dict)
        else features
    )
    script_path = root_run_dir / "broadened_follow_up_command.sh"
    literal_command = _follow_up_command_parts(
        broadened_args,
        focused_features,
        best_config,
        seeds_value=default_new_seeds,
        run_dir_value=str(default_run_dir),
        follow_up_from_value=str(default_follow_up_from),
        fail_on_verdict_value=default_fail_on_verdict,
        used_seed_history_value=used_seed_history_value,
    )
    shell_command = "PYTHONNOUSERSITE=1 " + shlex.join(literal_command)
    script_command = _follow_up_command_parts(
        broadened_args,
        focused_features,
        best_config,
        seeds_value="${NEW_SEEDS}",
        run_dir_value="${NEXT_RUN_DIR}",
        follow_up_from_value="${FOLLOW_UP_FROM}",
        fail_on_verdict_value=(
            "${FOLLOW_UP_FAIL_ON_VERDICT}"
            if default_fail_on_verdict is not None
            else None
        ),
        used_seed_history_value=used_seed_history_value,
    )
    script_usage = (
        f"FOLLOW_UP_FROM={default_follow_up_from} NEW_SEEDS={default_new_seeds} "
        f"NEXT_RUN_DIR={default_run_dir}"
    )
    if default_fail_on_verdict is not None:
        script_usage += f" FOLLOW_UP_FAIL_ON_VERDICT={default_fail_on_verdict}"
    script_usage += f" bash {script_path}"
    return {
        "schema": "st.llm_char_vae_context.broadened_follow_up_command.v1",
        "action": "promote_and_broaden_after_streak",
        "best_config": best_config,
        "default_new_seeds": default_new_seeds,
        "used_seed_history": used_seeds,
        "default_run_dir": str(default_run_dir),
        "default_follow_up_from": str(default_follow_up_from),
        "default_follow_up_fail_on_verdict": default_fail_on_verdict,
        "feature_family_focus": family_focus,
        "focused_features": focused_features,
        "broadened_epochs": int(broadened_args.epochs),
        "broadened_batches": int(broadened_args.batches),
        "broadened_eval_samples": int(broadened_args.eval_samples),
        "broadened_vae_epochs": int(broadened_args.vae_epochs),
        "broadened_vae_batches": int(broadened_args.vae_batches),
        "script_path": str(script_path),
        "shell_command": shell_command,
        "script_command": script_command,
        "script_usage": script_usage,
    }


def _script_command_line(command: list[str]) -> str:
    quoted = [shlex.quote(str(part)) for part in command]
    return (
        "PYTHONNOUSERSITE=\"${PYTHONNOUSERSITE:-1}\" "
        + " ".join(
            part.replace("'${NEW_SEEDS}'", '"${NEW_SEEDS}"').replace(
                "'${NEXT_RUN_DIR}'", '"${NEXT_RUN_DIR}"'
            ).replace(
                "'${FOLLOW_UP_FROM}'", '"${FOLLOW_UP_FROM}"'
            ).replace(
                "'${FOLLOW_UP_FAIL_ON_VERDICT}'", '"${FOLLOW_UP_FAIL_ON_VERDICT}"'
            )
            for part in quoted
        )
    )


def _write_next_follow_up_script(record: dict[str, Any]) -> None:
    script_path = pathlib.Path(str(record["script_path"]))
    default_fail_on_verdict = record.get("default_follow_up_fail_on_verdict")
    env_lines = [
        f"FOLLOW_UP_FROM=\"${{FOLLOW_UP_FROM:-{record['default_follow_up_from']}}}\"",
        f"NEW_SEEDS=\"${{NEW_SEEDS:-{record['default_new_seeds']}}}\"",
        f"NEXT_RUN_DIR=\"${{NEXT_RUN_DIR:-{record['default_run_dir']}}}\"",
    ]
    if default_fail_on_verdict is not None:
        env_lines.append(
            "FOLLOW_UP_FAIL_ON_VERDICT="
            f"\"${{FOLLOW_UP_FAIL_ON_VERDICT:-{default_fail_on_verdict}}}\""
        )
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            *env_lines,
            _script_command_line([str(part) for part in record["script_command"]]),
            "",
        ]
    )
    _write_text(script_path, text)
    script_path.chmod(script_path.stat().st_mode | 0o755)


def _write_guided_next_follow_up_script(record: dict[str, Any]) -> None:
    if not record.get("enabled"):
        return
    script_path_raw = record.get("script_path")
    script_command = record.get("script_command")
    if script_path_raw is None or not isinstance(script_command, list):
        return

    script_path = pathlib.Path(str(script_path_raw))
    default_fail_on_verdict = record.get("default_follow_up_fail_on_verdict")
    env_lines = [
        f"FOLLOW_UP_FROM=\"${{FOLLOW_UP_FROM:-{record['default_follow_up_from']}}}\"",
        f"NEW_SEEDS=\"${{NEW_SEEDS:-{record['default_new_seeds']}}}\"",
        f"NEXT_RUN_DIR=\"${{NEXT_RUN_DIR:-{record['default_run_dir']}}}\"",
    ]
    if default_fail_on_verdict is not None:
        env_lines.append(
            "FOLLOW_UP_FAIL_ON_VERDICT="
            f"\"${{FOLLOW_UP_FAIL_ON_VERDICT:-{default_fail_on_verdict}}}\""
        )
    reasons = record.get("reasons", [])
    reason_lines = (
        [f"# Reason: {reason}" for reason in reasons]
        if isinstance(reasons, list) and reasons
        else ["# Reason: -"]
    )
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Generated by models/python/llm_char_vae_context.py.",
            f"# Guidance action: {record.get('guidance_action') or '-'}",
            f"# Trajectory action: {record.get('trajectory_action') or '-'}",
            f"# Unsafe promotion: {record.get('unsafe_promotion')}",
            f"# Verdict: {record.get('verdict') or '-'}",
            *reason_lines,
            "# Example:",
            f"#   {record.get('script_usage') or f'bash {script_path}'}",
            "",
            *env_lines,
            _script_command_line([str(part) for part in script_command]),
            "",
        ]
    )
    _write_text(script_path, text)
    script_path.chmod(script_path.stat().st_mode | 0o755)


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(part == flag or part.startswith(f"{flag}=") for part in argv)


def _load_follow_up_summary(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"--follow-up-from must point to a summary object: {candidate}")
    return candidate, payload


def _summary_seeds(summary: dict[str, Any]) -> list[int]:
    run = summary.get("run", {})
    raw_seeds = run.get("seeds")
    if isinstance(raw_seeds, list):
        seeds = []
        for value in raw_seeds:
            try:
                seeds.append(int(value))
            except (TypeError, ValueError):
                continue
        if seeds:
            return list(dict.fromkeys(seeds))
    raw_seed = run.get("seed")
    if raw_seed is not None:
        try:
            return [int(raw_seed)]
        except (TypeError, ValueError):
            return []
    return []


def _summary_features(summary: dict[str, Any]) -> list[str]:
    run = summary.get("run", {})
    raw_features = run.get("features")
    if not isinstance(raw_features, list):
        return []
    features = [str(feature) for feature in raw_features if str(feature) in FEATURE_CHOICES]
    return list(dict.fromkeys(features))


def _args_run_budget(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "window_chars": int(args.window_chars),
        "latent_dim": int(args.latent_dim),
        "hidden": int(args.hidden),
        "epochs": int(args.epochs),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "eval_samples": int(args.eval_samples),
        "vae_epochs": int(args.vae_epochs),
        "vae_batches": int(args.vae_batches),
        "vae_batch_size": int(args.vae_batch_size),
    }


def _run_budget_from_run(run: Any) -> dict[str, Any]:
    if not isinstance(run, dict):
        return {}
    budget = {
        key: run.get(key)
        for key in RUN_BUDGET_KEYS
        if run.get(key) is not None
    }
    vae = run.get("vae")
    if isinstance(vae, dict):
        for source_key, target_key in (
            ("epochs", "vae_epochs"),
            ("batches", "vae_batches"),
            ("batch_size", "vae_batch_size"),
        ):
            if budget.get(target_key) is None and vae.get(source_key) is not None:
                budget[target_key] = vae.get(source_key)
    return budget


def _summary_run_budget(summary: dict[str, Any]) -> dict[str, Any]:
    budget = _run_budget_from_run(summary.get("run"))
    missing = [key for key in RUN_BUDGET_KEYS if key not in budget]
    if not missing:
        return budget

    seed_summaries = summary.get("seed_summaries", [])
    if not isinstance(seed_summaries, list):
        return budget
    for seed_summary in seed_summaries:
        if not isinstance(seed_summary, dict):
            continue
        run_dir = seed_summary.get("run_dir")
        if run_dir is None:
            continue
        seed_summary_path = pathlib.Path(str(run_dir)) / "summary.json"
        try:
            payload = json.loads(seed_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            nested = _summary_run_budget(payload)
            if nested:
                for key in missing:
                    if key in nested and key not in budget:
                        budget[key] = nested[key]
                missing = [key for key in RUN_BUDGET_KEYS if key not in budget]
                if not missing:
                    break
    return budget


def _run_budget_shift(
    source_budget: dict[str, Any] | None,
    current_budget: dict[str, Any] | None,
) -> dict[str, Any]:
    source_budget = source_budget if isinstance(source_budget, dict) else {}
    current_budget = current_budget if isinstance(current_budget, dict) else {}
    changes = []
    for key in RUN_BUDGET_KEYS:
        source_value = source_budget.get(key)
        current_value = current_budget.get(key)
        if source_value is None or current_value is None or source_value == current_value:
            continue
        changes.append(
            {
                "key": key,
                "source": source_value,
                "current": current_value,
            }
        )
    return {
        "schema": "st.llm_char_vae_context.run_budget_shift.v1",
        "changed": bool(changes),
        "changed_keys": [str(item["key"]) for item in changes],
        "changes": changes,
    }


def _run_budget_shift_label(shift: Any) -> str:
    if not isinstance(shift, dict):
        return "-"
    changes = shift.get("changes", [])
    if not isinstance(changes, list) or not changes:
        return "-"
    labels = []
    for item in changes:
        if not isinstance(item, dict):
            continue
        labels.append(f"{item.get('key')}:{item.get('source')}->{item.get('current')}")
    return ", ".join(labels) if labels else "-"


def _run_budget_label(budget: Any) -> str:
    if not isinstance(budget, dict):
        return "-"
    labels = [
        f"{key}={budget.get(key)}"
        for key in RUN_BUDGET_KEYS
        if budget.get(key) is not None
    ]
    return ", ".join(labels) if labels else "-"


def _args_with_run_budget(
    args: argparse.Namespace,
    run_budget: dict[str, Any] | None,
) -> argparse.Namespace:
    if not isinstance(run_budget, dict):
        return args
    overrides: dict[str, int] = {}
    for key in RUN_BUDGET_KEYS:
        value = run_budget.get(key)
        if value is None or not hasattr(args, key):
            continue
        try:
            overrides[key] = int(value)
        except (TypeError, ValueError):
            continue
    return _clone_args(args, **overrides) if overrides else args


def _source_best_config(summary: dict[str, Any]) -> dict[str, Any]:
    best_config = summary.get("best_config")
    if not isinstance(best_config, dict):
        next_follow_up = summary.get("next_follow_up_command")
        if isinstance(next_follow_up, dict):
            best_config = next_follow_up.get("best_config")
    if not isinstance(best_config, dict):
        raise ValueError("--follow-up-from summary does not contain best_config")

    normalize = str(best_config.get("feature_normalize", ""))
    if normalize not in NORMALIZE_CHOICES:
        joined = ", ".join(NORMALIZE_CHOICES)
        raise ValueError(
            f"--follow-up-from best_config has invalid feature_normalize={normalize!r}; "
            f"expected one of {joined}"
        )
    scale = _finite_float(best_config.get("hybrid_latent_scale"))
    if scale is None or scale < 0.0:
        raise ValueError("--follow-up-from best_config has invalid hybrid_latent_scale")

    sanitized = dict(best_config)
    sanitized["feature_normalize"] = normalize
    sanitized["hybrid_latent_scale"] = scale
    return sanitized


def _default_follow_up_seeds(summary: dict[str, Any]) -> str:
    source_seeds = _summary_seeds(summary)
    used_seeds = _summary_seed_history(summary)
    next_follow_up = summary.get("next_follow_up_command")
    if isinstance(next_follow_up, dict):
        raw = next_follow_up.get("default_new_seeds")
        if raw is not None and str(raw).strip():
            raw_seeds = _seed_csv_values(raw)
            used_set = set(used_seeds)
            if raw_seeds and all(seed not in used_set for seed in raw_seeds):
                return ",".join(str(seed) for seed in raw_seeds)
            return _fresh_seed_csv(
                used_seeds or source_seeds or [42],
                count=max(3, len(raw_seeds)),
            )
    return _fresh_seed_csv(
        used_seeds or source_seeds or [42],
        count=max(3, len(source_seeds)),
    )


def _follow_up_chain_source(summary: dict[str, Any]) -> dict[str, Any]:
    chain = summary.get("follow_up_chain")
    if isinstance(chain, dict):
        generation_raw = chain.get("generation", 0)
        try:
            generation = max(0, int(generation_raw))
        except (TypeError, ValueError):
            generation = 0
        ancestors_raw = chain.get("ancestors", [])
        verdicts_raw = chain.get("verdict_history", [])
        return {
            "generation": generation,
            "ancestors": [str(item) for item in ancestors_raw if str(item)]
            if isinstance(ancestors_raw, list)
            else [],
            "verdict_history": [str(item) for item in verdicts_raw if str(item)]
            if isinstance(verdicts_raw, list)
            else [],
        }

    follow_up = summary.get("follow_up")
    if isinstance(follow_up, dict):
        parent = follow_up.get("source_summary_path")
        result = summary.get("follow_up_result")
        verdict = result.get("verdict") if isinstance(result, dict) else None
        return {
            "generation": 1,
            "ancestors": [str(parent)] if parent is not None and str(parent) else [],
            "verdict_history": [str(verdict)] if verdict is not None and str(verdict) else [],
        }

    return {
        "generation": 0,
        "ancestors": [],
        "verdict_history": [],
    }


def _apply_follow_up_defaults(
    args: argparse.Namespace,
    argv: list[str],
) -> dict[str, Any] | None:
    if args.follow_up_from is None:
        return None

    source_path, source_summary = _load_follow_up_summary(args.follow_up_from)
    best_config = _source_best_config(source_summary)
    source_features = _summary_features(source_summary)
    source_seeds = _summary_seeds(source_summary)
    source_run_budget = _summary_run_budget(source_summary)
    source_chain = _follow_up_chain_source(source_summary)
    external_seed_history = _seed_csv_values(args.follow_up_used_seeds)

    explicit_features = _flag_present(argv, "--features")
    explicit_normalize = _flag_present(argv, "--feature-normalize") or _flag_present(
        argv,
        "--feature-normalize-modes",
    )
    explicit_scale = _flag_present(argv, "--hybrid-latent-scale") or _flag_present(
        argv,
        "--hybrid-latent-scales",
    )
    explicit_head_init = _flag_present(argv, "--head-init")
    explicit_seeds = _flag_present(argv, "--seeds")

    applied_defaults: dict[str, Any] = {}
    if source_features and not explicit_features:
        args.features = ",".join(source_features)
        applied_defaults["features"] = args.features
    if not explicit_normalize:
        args.feature_normalize = best_config["feature_normalize"]
        args.feature_normalize_modes = None
        applied_defaults["feature_normalize"] = args.feature_normalize
    if not explicit_scale:
        args.hybrid_latent_scale = float(best_config["hybrid_latent_scale"])
        args.hybrid_latent_scales = None
        applied_defaults["hybrid_latent_scale"] = args.hybrid_latent_scale
    if not explicit_head_init:
        source_run = source_summary.get("run", {})
        source_head_init = (
            source_run.get("head_init") if isinstance(source_run, dict) else None
        )
        args.head_init = (
            str(source_head_init)
            if source_head_init in HEAD_INIT_CHOICES
            else "legacy"
        )
        applied_defaults["head_init"] = args.head_init
    if not explicit_seeds:
        args.seeds = _default_follow_up_seeds(source_summary)
        applied_defaults["seeds"] = args.seeds

    return {
        "schema": "st.llm_char_vae_context.follow_up.v1",
        "source_summary_path": str(source_path),
        "source_status": source_summary.get("status"),
        "source_best_feature": source_summary.get("best_feature"),
        "source_best_config": best_config,
        "source_features": source_features,
        "source_seeds": source_seeds,
        "source_run_budget": source_run_budget,
        "source_chain": source_chain,
        "external_seed_history": external_seed_history,
        "applied_defaults": applied_defaults,
        "user_overrides": {
            "features": explicit_features,
            "feature_normalize": explicit_normalize,
            "hybrid_latent_scale": explicit_scale,
            "head_init": explicit_head_init,
            "seeds": explicit_seeds,
        },
    }


def _config_label(config: dict[str, Any] | None) -> str:
    if not config:
        return "-"
    feature = config.get("best_feature") or config.get("feature") or "-"
    normalize = config.get("feature_normalize", "-")
    scale = config.get("hybrid_latent_scale", "-")
    return f"{feature} @ normalize={normalize} scale={scale}"


def _matching_config_summary(
    config_summaries: list[dict[str, Any]],
    source_best_config: dict[str, Any],
) -> dict[str, Any] | None:
    source_normalize = source_best_config.get("feature_normalize")
    source_scale = _finite_float(source_best_config.get("hybrid_latent_scale"))
    for summary in config_summaries:
        if summary.get("feature_normalize") != source_normalize:
            continue
        scale = _finite_float(summary.get("hybrid_latent_scale"))
        if source_scale is None or scale is None:
            continue
        if abs(scale - source_scale) <= 1e-12:
            return summary
    return None


def _config_feature_best_nll_stats(
    config: dict[str, Any] | None,
    feature: Any,
) -> dict[str, Any]:
    if not config or feature is None:
        return {}
    feature_name = str(feature)
    for row in config.get("feature_summary", []):
        if row.get("feature") != feature_name:
            continue
        stats = row.get("best_nll")
        if isinstance(stats, dict) and stats.get("stderr") is not None:
            return stats
        if isinstance(stats, dict) and stats.get("stddev") is not None:
            count = _finite_float(stats.get("count"))
            if count is not None and count > 0.0:
                enriched = dict(stats)
                enriched["stderr"] = float(stats["stddev"]) / math.sqrt(count)
                return enriched
            return stats

    seed_summaries = config.get("seed_summaries", [])
    if not isinstance(seed_summaries, list):
        seed_summaries = []
    if not seed_summaries and config.get("run_dir") is not None:
        summary_path = pathlib.Path(str(config.get("run_dir"))) / "summary.json"
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("seed_summaries"), list):
            seed_summaries = payload["seed_summaries"]

    values = []
    for seed_summary in seed_summaries:
        if not isinstance(seed_summary, dict):
            continue
        for row in seed_summary.get("ranking", []):
            if row.get("feature") != feature_name:
                continue
            value = _finite_float(row.get("best_mean_nll"))
            if value is not None:
                values.append(value)
            break
    if values:
        return _metric_stats(values)
    return {}


def _config_feature_uncertainty_fields(
    config: dict[str, Any] | None,
    feature: Any,
) -> dict[str, Any]:
    stats = _config_feature_best_nll_stats(config, feature)
    return {
        "mean_best_nll_count": stats.get("count"),
        "mean_best_nll_stddev": stats.get("stddev"),
        "mean_best_nll_stderr": stats.get("stderr"),
    }


def _compact_config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    top = config.get("ranking", [{}])[0] if config.get("ranking") else {}
    feature = top.get("feature")
    return {
        "feature_normalize": config.get("feature_normalize"),
        "hybrid_latent_scale": config.get("hybrid_latent_scale"),
        "best_feature": config.get("best_feature"),
        "status": config.get("status"),
        "mean_best_nll": top.get("mean_best_nll"),
        "mean_best_accuracy": top.get("mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": top.get("mean_best_nll_delta_vs_raw"),
        **_config_feature_uncertainty_fields(config, feature),
        "run_dir": config.get("run_dir"),
    }


def _config_feature_summary(
    config: dict[str, Any] | None,
    feature: Any,
) -> dict[str, Any] | None:
    if not config or feature is None:
        return None
    feature_name = str(feature)
    for row in config.get("ranking", []):
        if row.get("feature") != feature_name:
            continue
        return {
            "feature_normalize": config.get("feature_normalize"),
            "hybrid_latent_scale": config.get("hybrid_latent_scale"),
            "feature": feature_name,
            "best_feature": feature_name,
            "mean_best_nll": row.get("mean_best_nll"),
            "mean_best_accuracy": row.get("mean_best_accuracy"),
            "mean_best_nll_delta_vs_raw": row.get("mean_best_nll_delta_vs_raw"),
            **_config_feature_uncertainty_fields(config, feature_name),
            "runs": row.get("runs"),
            "run_dir": config.get("run_dir"),
        }
    return None


def _delta_verdict(delta: float | None, min_nll_delta: float) -> str:
    if delta is None:
        return "unknown"
    if delta < -float(min_nll_delta):
        return "improved"
    if delta > float(min_nll_delta):
        return "regressed"
    return "confirmed"


def _effective_follow_up_tolerance(
    *,
    min_nll_delta: float,
    follow_up_confirm_tolerance: float,
    stderr: float | None,
) -> float:
    values = [float(min_nll_delta), float(follow_up_confirm_tolerance)]
    if stderr is not None and math.isfinite(float(stderr)):
        values.append(float(stderr))
    return max(value for value in values if math.isfinite(value))


def _parse_follow_up_fail_verdicts(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    verdicts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [verdict for verdict in verdicts if verdict not in FOLLOW_UP_VERDICTS]
    if unknown:
        joined = ", ".join(FOLLOW_UP_VERDICTS)
        raise ValueError(
            f"unknown --follow-up-fail-on-verdict entries {unknown}; "
            f"expected comma-separated {joined}"
        )
    return list(dict.fromkeys(verdicts))


def _verdict_streak(verdict_history: list[str], target: str) -> int:
    streak = 0
    for verdict in reversed(verdict_history):
        if verdict != target:
            break
        streak += 1
    return streak


def _follow_up_result(
    follow_up: dict[str, Any] | None,
    config_summaries: list[dict[str, Any]],
    current_best_config: dict[str, Any] | None,
    *,
    min_nll_delta: float,
    follow_up_confirm_tolerance: float = 0.0,
    current_run_budget: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not follow_up:
        return None
    source_best_config = follow_up.get("source_best_config")
    if not isinstance(source_best_config, dict):
        return None
    matched_config = _matching_config_summary(config_summaries, source_best_config)
    evaluated = _compact_config_summary(matched_config)
    source_feature_evaluated = _config_feature_summary(
        matched_config,
        source_best_config.get("best_feature"),
    )
    source_nll = _finite_float(source_best_config.get("mean_best_nll"))
    evaluated_nll = _finite_float(evaluated.get("mean_best_nll") if evaluated else None)
    source_feature_nll = _finite_float(
        source_feature_evaluated.get("mean_best_nll") if source_feature_evaluated else None
    )
    evaluated_delta_vs_raw = _finite_float(
        evaluated.get("mean_best_nll_delta_vs_raw") if evaluated else None
    )
    source_feature_delta_vs_raw = _finite_float(
        source_feature_evaluated.get("mean_best_nll_delta_vs_raw")
        if source_feature_evaluated
        else None
    )
    current_best_delta_vs_raw = _finite_float(
        current_best_config.get("mean_best_nll_delta_vs_raw")
        if isinstance(current_best_config, dict)
        else None
    )
    config_delta = None
    source_feature_delta = None
    if source_nll is not None and evaluated_nll is not None:
        config_delta = evaluated_nll - source_nll
    if source_nll is not None and source_feature_nll is not None:
        source_feature_delta = source_feature_nll - source_nll

    evaluated_stderr = _finite_float(
        evaluated.get("mean_best_nll_stderr") if evaluated else None
    )
    source_feature_stderr = _finite_float(
        source_feature_evaluated.get("mean_best_nll_stderr")
        if source_feature_evaluated
        else None
    )
    config_tolerance = _effective_follow_up_tolerance(
        min_nll_delta=min_nll_delta,
        follow_up_confirm_tolerance=follow_up_confirm_tolerance,
        stderr=evaluated_stderr,
    )
    source_feature_tolerance = _effective_follow_up_tolerance(
        min_nll_delta=min_nll_delta,
        follow_up_confirm_tolerance=follow_up_confirm_tolerance,
        stderr=source_feature_stderr,
    )

    config_verdict = _delta_verdict(config_delta, config_tolerance)
    source_feature_verdict = _delta_verdict(
        source_feature_delta,
        source_feature_tolerance,
    )
    source_feature = source_best_config.get("best_feature")
    source_best_feature_retained = (
        evaluated is not None
        and source_feature is not None
        and evaluated.get("best_feature") == source_feature
    )
    source_run_budget = (
        follow_up.get("source_run_budget")
        if isinstance(follow_up.get("source_run_budget"), dict)
        else {}
    )
    current_run_budget = (
        current_run_budget if isinstance(current_run_budget, dict) else {}
    )
    run_budget_shift = _run_budget_shift(source_run_budget, current_run_budget)

    return {
        "schema": "st.llm_char_vae_context.follow_up_result.v1",
        "source_summary_path": follow_up.get("source_summary_path"),
        "source_best_config": source_best_config,
        "source_run_budget": source_run_budget,
        "current_run_budget": current_run_budget,
        "run_budget_shift": run_budget_shift,
        "run_budget_shifted": bool(run_budget_shift.get("changed")),
        "run_budget_shift_keys": run_budget_shift.get("changed_keys", []),
        "evaluated_config": evaluated,
        "source_feature_evaluated": source_feature_evaluated,
        "current_best_config": current_best_config,
        "mean_best_nll_delta_vs_source": config_delta,
        "source_feature_mean_best_nll_delta_vs_source": source_feature_delta,
        "evaluated_mean_best_nll_delta_vs_raw": evaluated_delta_vs_raw,
        "source_feature_mean_best_nll_delta_vs_raw": source_feature_delta_vs_raw,
        "current_best_mean_best_nll_delta_vs_raw": current_best_delta_vs_raw,
        "follow_up_confirm_tolerance": float(follow_up_confirm_tolerance),
        "effective_config_min_nll_delta": config_tolerance,
        "effective_source_feature_min_nll_delta": source_feature_tolerance,
        "evaluated_mean_best_nll_stderr": evaluated_stderr,
        "source_feature_mean_best_nll_stderr": source_feature_stderr,
        "config_verdict": config_verdict,
        "source_feature_verdict": source_feature_verdict,
        "evaluated_raw_verdict": _delta_verdict(evaluated_delta_vs_raw, min_nll_delta),
        "source_feature_raw_verdict": _delta_verdict(
            source_feature_delta_vs_raw,
            min_nll_delta,
        ),
        "current_best_raw_verdict": _delta_verdict(
            current_best_delta_vs_raw,
            min_nll_delta,
        ),
        "source_best_feature_retained": source_best_feature_retained,
        "verdict": source_feature_verdict
        if source_feature_verdict != "unknown"
        else config_verdict,
        "match_found": evaluated is not None,
    }


def _follow_up_chain_record(
    follow_up: dict[str, Any] | None,
    follow_up_result: dict[str, Any] | None,
    root_run_dir: pathlib.Path,
) -> dict[str, Any] | None:
    if not follow_up:
        return None
    source_chain = follow_up.get("source_chain")
    source_chain = source_chain if isinstance(source_chain, dict) else {}
    try:
        source_generation = max(0, int(source_chain.get("generation", 0)))
    except (TypeError, ValueError):
        source_generation = 0
    ancestors_raw = source_chain.get("ancestors", [])
    ancestors = (
        [str(item) for item in ancestors_raw if str(item)]
        if isinstance(ancestors_raw, list)
        else []
    )
    parent = follow_up.get("source_summary_path")
    if parent is not None and str(parent):
        ancestors.append(str(parent))
    verdicts_raw = source_chain.get("verdict_history", [])
    verdict_history = (
        [str(item) for item in verdicts_raw if str(item)]
        if isinstance(verdicts_raw, list)
        else []
    )
    latest_verdict = (
        str(follow_up_result.get("verdict"))
        if isinstance(follow_up_result, dict) and follow_up_result.get("verdict") is not None
        else "unknown"
    )
    verdict_history.append(latest_verdict)
    return {
        "schema": "st.llm_char_vae_context.follow_up_chain.v1",
        "generation": source_generation + 1,
        "parent_summary_path": str(parent) if parent is not None else None,
        "run_dir": str(root_run_dir),
        "ancestors": ancestors,
        "verdict_history": verdict_history,
        "latest_verdict": latest_verdict,
        "improved_streak": _verdict_streak(verdict_history, "improved"),
        "regressed_streak": _verdict_streak(verdict_history, "regressed"),
        "unknown_streak": _verdict_streak(verdict_history, "unknown"),
    }


def _follow_up_ancestor_record(
    summary_path: pathlib.Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    run = payload.get("run")
    run = run if isinstance(run, dict) else {}
    chain = payload.get("follow_up_chain")
    chain = chain if isinstance(chain, dict) else {}
    best_config = payload.get("best_config")
    best_config = best_config if isinstance(best_config, dict) else None
    follow_up_result = payload.get("follow_up_result")
    follow_up_result = follow_up_result if isinstance(follow_up_result, dict) else {}
    follow_up_guidance = payload.get("follow_up_guidance")
    follow_up_guidance = (
        follow_up_guidance if isinstance(follow_up_guidance, dict) else {}
    )
    guided_next = payload.get("guided_next_follow_up_command")
    guided_next = guided_next if isinstance(guided_next, dict) else {}
    gate = payload.get("follow_up_gate")
    gate = gate if isinstance(gate, dict) else {}

    return {
        "schema": "st.llm_char_vae_context.follow_up_ancestor.v1",
        "summary_path": str(summary_path),
        "run_dir": str(run.get("run_dir") or payload.get("run_dir") or "-"),
        "generation": chain.get("generation", 0),
        "status": payload.get("status"),
        "best_feature": payload.get("best_feature"),
        "best_config": best_config,
        "mean_best_nll": best_config.get("mean_best_nll") if best_config else None,
        "mean_best_accuracy": (
            best_config.get("mean_best_accuracy") if best_config else None
        ),
        "verdict": follow_up_result.get("verdict"),
        "config_verdict": follow_up_result.get("config_verdict"),
        "source_feature_verdict": follow_up_result.get("source_feature_verdict"),
        "source_feature_raw_verdict": follow_up_result.get(
            "source_feature_raw_verdict"
        ),
        "source_feature_mean_best_nll_delta_vs_raw": follow_up_result.get(
            "source_feature_mean_best_nll_delta_vs_raw"
        ),
        "current_best_raw_verdict": follow_up_result.get("current_best_raw_verdict"),
        "current_best_mean_best_nll_delta_vs_raw": follow_up_result.get(
            "current_best_mean_best_nll_delta_vs_raw"
        ),
        "source_best_feature_retained": follow_up_result.get(
            "source_best_feature_retained"
        ),
        "guidance_action": follow_up_guidance.get("action"),
        "guided_enabled": guided_next.get("enabled"),
        "gate_failed": gate.get("failed"),
    }


def _follow_up_missing_ancestor_record(
    summary_path: pathlib.Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema": "st.llm_char_vae_context.follow_up_ancestor.v1",
        "summary_path": str(summary_path),
        "missing": True,
        "error": str(error),
    }


def _follow_up_ancestor_records(
    follow_up_chain: dict[str, Any] | None,
    *,
    max_ancestors: int = FOLLOW_UP_CHAIN_MAX_ANCESTORS,
) -> dict[str, Any] | None:
    if not isinstance(follow_up_chain, dict):
        return None
    ancestors_raw = follow_up_chain.get("ancestors", [])
    if not isinstance(ancestors_raw, list) or not ancestors_raw:
        return None

    records = []
    seen: set[str] = set()
    for raw_path in ancestors_raw[:max_ancestors]:
        if raw_path is None or not str(raw_path):
            continue
        summary_path = pathlib.Path(str(raw_path)).expanduser()
        summary_key = str(summary_path)
        if summary_key in seen:
            records.append(
                {
                    "schema": "st.llm_char_vae_context.follow_up_ancestor.v1",
                    "summary_path": summary_key,
                    "cycle_detected": True,
                }
            )
            break
        seen.add(summary_key)
        try:
            loaded_path, payload = _load_follow_up_summary(summary_path)
            records.append(_follow_up_ancestor_record(loaded_path, payload))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            records.append(_follow_up_missing_ancestor_record(summary_path, exc))
            break

    if not records:
        return None
    return {
        "schema": "st.llm_char_vae_context.follow_up_ancestors.v1",
        "ancestor_count": len(records),
        "truncated": len(ancestors_raw) > max_ancestors,
        "ancestors": records,
    }


def _follow_up_ancestor_row(record: dict[str, Any]) -> list[str]:
    best_config = record.get("best_config")
    guided_enabled = record.get("guided_enabled")
    return [
        str(record.get("generation") or "-"),
        f"`{record.get('summary_path') or '-'}`",
        str(record.get("status") or "-"),
        _config_label(best_config if isinstance(best_config, dict) else None),
        _fmt_float(record.get("mean_best_nll")),
        str(record.get("verdict") or "-"),
        str(record.get("guidance_action") or "-"),
        str(guided_enabled if guided_enabled is not None else "-"),
        "yes" if record.get("missing") else "no",
    ]


def _follow_up_current_trajectory_point(
    root_run_dir: pathlib.Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    chain = summary.get("follow_up_chain")
    chain = chain if isinstance(chain, dict) else {}
    best_config = summary.get("best_config")
    best_config = best_config if isinstance(best_config, dict) else None
    result = summary.get("follow_up_result")
    result = result if isinstance(result, dict) else {}
    guidance = summary.get("follow_up_guidance")
    guidance = guidance if isinstance(guidance, dict) else {}
    guided_next = summary.get("guided_next_follow_up_command")
    guided_next = guided_next if isinstance(guided_next, dict) else {}
    gate = summary.get("follow_up_gate")
    gate = gate if isinstance(gate, dict) else {}
    return {
        "schema": "st.llm_char_vae_context.follow_up_trajectory_point.v1",
        "role": "current",
        "summary_path": str(root_run_dir / "summary.json"),
        "run_dir": str(root_run_dir),
        "generation": chain.get("generation", 0),
        "status": summary.get("status"),
        "best_feature": summary.get("best_feature"),
        "best_config": best_config,
        "mean_best_nll": best_config.get("mean_best_nll") if best_config else None,
        "mean_best_accuracy": (
            best_config.get("mean_best_accuracy") if best_config else None
        ),
        "verdict": result.get("verdict"),
        "config_verdict": result.get("config_verdict"),
        "source_feature_verdict": result.get("source_feature_verdict"),
        "source_feature_raw_verdict": result.get("source_feature_raw_verdict"),
        "source_feature_mean_best_nll_delta_vs_raw": result.get(
            "source_feature_mean_best_nll_delta_vs_raw"
        ),
        "current_best_raw_verdict": result.get("current_best_raw_verdict"),
        "current_best_mean_best_nll_delta_vs_raw": result.get(
            "current_best_mean_best_nll_delta_vs_raw"
        ),
        "source_best_feature_retained": result.get("source_best_feature_retained"),
        "guidance_action": guidance.get("action"),
        "guided_enabled": guided_next.get("enabled"),
        "gate_failed": gate.get("failed"),
    }


def _follow_up_trajectory_point_from_ancestor(
    record: dict[str, Any],
) -> dict[str, Any]:
    point = dict(record)
    point["schema"] = "st.llm_char_vae_context.follow_up_trajectory_point.v1"
    point["role"] = "ancestor"
    return point


def _trajectory_best_feature(point: dict[str, Any] | None) -> str | None:
    if not isinstance(point, dict):
        return None
    best_config = point.get("best_config")
    if isinstance(best_config, dict) and best_config.get("best_feature") is not None:
        return str(best_config.get("best_feature"))
    if point.get("best_feature") is not None:
        return str(point.get("best_feature"))
    return None


def _follow_up_trajectory_action(
    *,
    trajectory_verdict: str,
    latest_verdict: str,
    current_gate_failed: bool,
    current_raw_positive: bool,
    raw_positive_streak: int,
    source_feature_tradeoff: bool,
    best_feature_changed: bool,
    current_is_best_generation: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if current_gate_failed and source_feature_tradeoff:
        reasons.append("overall trajectory improved while source feature regressed")
        return "audit_feature_swap_before_promotion", reasons
    if (
        current_gate_failed
        and current_raw_positive
        and raw_positive_streak >= 3
        and not current_is_best_generation
    ):
        reasons.append(f"raw-positive streak={raw_positive_streak} survived")
        reasons.append("latest point regressed away from the best generation")
        reasons.append("reconfirm the best raw-positive generation with fresh seeds")
        return "reconfirm_best_raw_positive_generation", reasons
    if current_gate_failed and current_raw_positive:
        reasons.append(f"follow-up gate failed on latest verdict={latest_verdict}")
        reasons.append("source feature still beats the raw baseline")
        reasons.append("widen seed confirmation before rejecting raw-positive candidate")
        return "widen_seed_confirmation_on_raw_positive_regression", reasons
    if current_gate_failed:
        reasons.append(f"follow-up gate failed on latest verdict={latest_verdict}")
        return "stop_on_follow_up_gate", reasons
    if source_feature_tradeoff:
        reasons.append("config improved while source feature regressed")
        return "review_feature_swap_before_promotion", reasons
    if trajectory_verdict == "improved" and latest_verdict == "improved":
        reasons.append("trajectory and latest follow-up both improved")
        if current_is_best_generation:
            reasons.append("current generation is the best NLL point")
        if best_feature_changed:
            reasons.append("best feature changed across the chain")
        return "confirm_trajectory_with_fresh_seeds", reasons
    if trajectory_verdict == "improved":
        reasons.append("trajectory improved but latest verdict is not improved")
        return "continue_or_audit_mixed_trajectory", reasons
    if trajectory_verdict == "confirmed":
        reasons.append("trajectory stayed within the confirmation band")
        return "collect_more_seed_evidence", reasons
    if trajectory_verdict == "regressed":
        reasons.append("trajectory regressed relative to the first measured point")
        return "return_to_best_generation_or_rerun", reasons
    reasons.append("trajectory verdict is unknown")
    return "inspect_trajectory", reasons


def _trajectory_raw_delta(point: dict[str, Any]) -> float | None:
    for key in (
        "current_best_mean_best_nll_delta_vs_raw",
        "source_feature_mean_best_nll_delta_vs_raw",
    ):
        value = _finite_float(point.get(key))
        if value is not None:
            return value
    best_config = point.get("best_config")
    if isinstance(best_config, dict):
        return _finite_float(best_config.get("mean_best_nll_delta_vs_raw"))
    return None


def _trajectory_raw_verdict(point: dict[str, Any]) -> str:
    for key in ("current_best_raw_verdict", "source_feature_raw_verdict"):
        verdict = str(point.get(key) or "unknown")
        if verdict != "unknown":
            return verdict
    raw_delta = _trajectory_raw_delta(point)
    return _delta_verdict(raw_delta, 0.0) if raw_delta is not None else "unknown"


def _trajectory_raw_positive(
    point: dict[str, Any],
    *,
    min_nll_delta: float,
) -> bool | None:
    raw_delta = _trajectory_raw_delta(point)
    if raw_delta is not None:
        return _delta_verdict(raw_delta, min_nll_delta) == "improved"
    raw_verdict = _trajectory_raw_verdict(point)
    if raw_verdict == "improved":
        return True
    if raw_verdict in {"confirmed", "regressed"}:
        return False
    return None


def _bool_streak(points: list[dict[str, Any]], key: str, value: bool) -> int:
    streak = 0
    for point in reversed(points):
        if point.get(key) is value:
            streak += 1
            continue
        break
    return streak


def _follow_up_trajectory_record(
    root_run_dir: pathlib.Path,
    summary: dict[str, Any],
    follow_up_ancestors: dict[str, Any] | None,
    *,
    min_nll_delta: float,
) -> dict[str, Any] | None:
    chain = summary.get("follow_up_chain")
    if not isinstance(chain, dict):
        return None

    points = []
    ancestors_raw = (
        follow_up_ancestors.get("ancestors", [])
        if isinstance(follow_up_ancestors, dict)
        else []
    )
    if isinstance(ancestors_raw, list):
        points.extend(
            _follow_up_trajectory_point_from_ancestor(record)
            for record in ancestors_raw
            if isinstance(record, dict)
        )
    points.append(_follow_up_current_trajectory_point(root_run_dir, summary))

    previous_nll = None
    metric_points = []
    for point in points:
        nll = _finite_float(point.get("mean_best_nll"))
        point["raw_delta_vs_raw"] = _trajectory_raw_delta(point)
        point["raw_verdict"] = _trajectory_raw_verdict(point)
        point["raw_positive"] = _trajectory_raw_positive(
            point,
            min_nll_delta=min_nll_delta,
        )
        point["mean_best_nll_delta_from_previous"] = (
            nll - previous_nll if nll is not None and previous_nll is not None else None
        )
        if (
            nll is not None
            and not point.get("missing")
            and not point.get("cycle_detected")
        ):
            metric_points.append((nll, point))
            previous_nll = nll

    verdict_history = chain.get("verdict_history", [])
    verdicts = (
        [str(verdict) for verdict in verdict_history if str(verdict)]
        if isinstance(verdict_history, list)
        else []
    )
    verdict_counts = {verdict: verdicts.count(verdict) for verdict in FOLLOW_UP_VERDICTS}
    latest_verdict = str(
        chain.get("latest_verdict") or (verdicts[-1] if verdicts else "unknown")
    )

    start_nll = metric_points[0][0] if metric_points else None
    current_nll = _finite_float(points[-1].get("mean_best_nll")) if points else None
    cumulative_delta = (
        current_nll - start_nll
        if current_nll is not None and start_nll is not None
        else None
    )
    best_nll = None
    best_point: dict[str, Any] | None = None
    if metric_points:
        best_nll, best_point = min(metric_points, key=lambda item: item[0])

    metric_point_records = [point for _nll, point in metric_points]
    raw_evidence_points = [
        point
        for point in metric_point_records
        if point.get("raw_positive") is not None
    ]
    raw_positive_count = sum(
        1 for point in raw_evidence_points if point.get("raw_positive") is True
    )
    raw_positive_rate = (
        raw_positive_count / len(raw_evidence_points)
        if raw_evidence_points
        else None
    )
    raw_deltas = [
        value
        for value in (
            _finite_float(point.get("raw_delta_vs_raw"))
            for point in metric_point_records
        )
        if value is not None
    ]
    mean_raw_delta = sum(raw_deltas) / len(raw_deltas) if raw_deltas else None
    best_raw_delta = min(raw_deltas) if raw_deltas else None
    raw_positive_streak = _bool_streak(metric_point_records, "raw_positive", True)
    raw_negative_streak = _bool_streak(metric_point_records, "raw_positive", False)

    current_point = points[-1] if points else {}
    trajectory_verdict = _delta_verdict(cumulative_delta, min_nll_delta)
    current_gate_failed = bool(current_point.get("gate_failed"))
    current_config_verdict = str(current_point.get("config_verdict") or "unknown")
    current_source_feature_verdict = str(
        current_point.get("source_feature_verdict") or "unknown"
    )
    current_source_feature_raw_verdict = str(
        current_point.get("source_feature_raw_verdict") or "unknown"
    )
    current_source_retained = current_point.get("source_best_feature_retained")
    source_feature_tradeoff = (
        current_config_verdict == "improved"
        and (
            current_source_feature_verdict == "regressed"
            or current_source_retained is False
        )
    )
    current_raw_positive = (
        current_source_retained is True
        and current_source_feature_raw_verdict == "improved"
        and not source_feature_tradeoff
    )
    start_point = metric_points[0][1] if metric_points else None
    start_best_feature = _trajectory_best_feature(start_point)
    current_best_feature = _trajectory_best_feature(current_point)
    best_feature_changed = (
        start_best_feature is not None
        and current_best_feature is not None
        and start_best_feature != current_best_feature
    )
    current_is_best_generation = (
        best_point is current_point if best_point is not None else False
    )
    trajectory_action, trajectory_reasons = _follow_up_trajectory_action(
        trajectory_verdict=trajectory_verdict,
        latest_verdict=latest_verdict,
        current_gate_failed=current_gate_failed,
        current_raw_positive=current_raw_positive,
        raw_positive_streak=raw_positive_streak,
        source_feature_tradeoff=source_feature_tradeoff,
        best_feature_changed=best_feature_changed,
        current_is_best_generation=current_is_best_generation,
    )
    unsafe_promotion = bool(
        source_feature_tradeoff or (current_gate_failed and not current_raw_positive)
    )
    return {
        "schema": "st.llm_char_vae_context.follow_up_trajectory.v1",
        "point_count": len(points),
        "metric_point_count": len(metric_points),
        "generation": chain.get("generation"),
        "latest_verdict": latest_verdict,
        "trajectory_verdict": trajectory_verdict,
        "trajectory_action": trajectory_action,
        "trajectory_reasons": trajectory_reasons,
        "verdict_counts": verdict_counts,
        "start_mean_best_nll": start_nll,
        "current_mean_best_nll": current_nll,
        "cumulative_mean_best_nll_delta": cumulative_delta,
        "raw_evidence_count": len(raw_evidence_points),
        "raw_positive_count": raw_positive_count,
        "raw_positive_rate": raw_positive_rate,
        "raw_positive_streak": raw_positive_streak,
        "raw_negative_streak": raw_negative_streak,
        "mean_raw_delta_vs_raw": mean_raw_delta,
        "best_raw_delta_vs_raw": best_raw_delta,
        "current_raw_delta_vs_raw": _finite_float(
            current_point.get("raw_delta_vs_raw")
        ),
        "best_mean_best_nll": best_nll,
        "best_generation": best_point.get("generation") if best_point else None,
        "best_summary_path": best_point.get("summary_path") if best_point else None,
        "best_config": best_point.get("best_config") if best_point else None,
        "start_best_feature": start_best_feature,
        "current_best_feature": current_best_feature,
        "best_feature_changed": best_feature_changed,
        "source_feature_tradeoff": source_feature_tradeoff,
        "unsafe_promotion": unsafe_promotion,
        "current_guidance_action": current_point.get("guidance_action"),
        "current_guided_enabled": current_point.get("guided_enabled"),
        "current_gate_failed": current_point.get("gate_failed"),
        "current_config_verdict": current_config_verdict,
        "current_source_feature_verdict": current_source_feature_verdict,
        "current_source_feature_raw_verdict": current_source_feature_raw_verdict,
        "current_source_best_feature_retained": current_source_retained,
        "current_raw_positive": current_raw_positive,
        "points": points,
    }


def _follow_up_trajectory_point_row(point: dict[str, Any]) -> list[str]:
    best_config = point.get("best_config")
    return [
        str(point.get("generation") or "-"),
        str(point.get("role") or "-"),
        _config_label(best_config if isinstance(best_config, dict) else None),
        _fmt_float(point.get("mean_best_nll")),
        _fmt_float(point.get("mean_best_nll_delta_from_previous")),
        _fmt_float(point.get("raw_delta_vs_raw")),
        str(point.get("raw_positive") if point.get("raw_positive") is not None else "-"),
        str(point.get("verdict") or "-"),
        str(point.get("guidance_action") or "-"),
        str(point.get("gate_failed") if point.get("gate_failed") is not None else "-"),
    ]


def _follow_up_gate_record(
    follow_up_result: dict[str, Any] | None,
    fail_on_verdicts: list[str],
) -> dict[str, Any] | None:
    if not fail_on_verdicts or not isinstance(follow_up_result, dict):
        return None
    verdict = str(follow_up_result.get("verdict") or "unknown")
    effective_verdict = verdict
    verdict_basis = "verdict"
    if follow_up_result.get("run_budget_shifted"):
        raw_verdict = str(
            follow_up_result.get("source_feature_raw_verdict") or "unknown"
        )
        if (
            follow_up_result.get("source_best_feature_retained")
            and raw_verdict in FOLLOW_UP_VERDICTS
            and raw_verdict != "unknown"
        ):
            effective_verdict = raw_verdict
            verdict_basis = "source_feature_raw_verdict_after_run_budget_shift"
    failed = effective_verdict in set(fail_on_verdicts)
    return {
        "schema": "st.llm_char_vae_context.follow_up_gate.v1",
        "verdict": verdict,
        "effective_verdict": effective_verdict,
        "verdict_basis": verdict_basis,
        "fail_on_verdicts": fail_on_verdicts,
        "failed": failed,
        "exit_code": 1 if failed else 0,
    }


def _follow_up_guidance_record(
    follow_up_result: dict[str, Any] | None,
    follow_up_chain: dict[str, Any] | None,
    follow_up_gate: dict[str, Any] | None,
    next_follow_up: dict[str, Any] | None,
    follow_up_trajectory: dict[str, Any] | None = None,
    best_generation_follow_up: dict[str, Any] | None = None,
    broadened_follow_up: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(follow_up_result, dict):
        return None

    verdict = str(follow_up_result.get("verdict") or "unknown")
    config_verdict = str(follow_up_result.get("config_verdict") or "unknown")
    source_feature_verdict = str(
        follow_up_result.get("source_feature_verdict") or "unknown"
    )
    source_feature_raw_verdict = str(
        follow_up_result.get("source_feature_raw_verdict") or "unknown"
    )
    source_retained = bool(follow_up_result.get("source_best_feature_retained"))
    gate_failed = (
        bool(follow_up_gate.get("failed"))
        if isinstance(follow_up_gate, dict)
        else False
    )
    improved_streak = (
        int(follow_up_chain.get("improved_streak") or 0)
        if isinstance(follow_up_chain, dict)
        else 0
    )
    regressed_streak = (
        int(follow_up_chain.get("regressed_streak") or 0)
        if isinstance(follow_up_chain, dict)
        else 0
    )

    source_feature_needs_review = (
        source_feature_verdict == "regressed" or not source_retained
    )
    config_improved_while_source_regressed = (
        config_verdict == "improved" and source_feature_verdict == "regressed"
    )
    raw_positive_regression = (
        source_retained
        and source_feature_verdict == "regressed"
        and source_feature_raw_verdict == "improved"
    )
    reasons: list[str] = []
    promote_current_best = False
    use_next_follow_up_command = False
    use_best_generation_follow_up_command = False
    use_broadened_follow_up_command = False
    action = "review_follow_up"

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    run_budget_shift = follow_up_result.get("run_budget_shift")
    if follow_up_result.get("run_budget_shifted"):
        add_reason(f"run budget shifted: {_run_budget_shift_label(run_budget_shift)}")

    if gate_failed:
        action = (
            "widen_seed_confirmation_on_raw_positive_regression"
            if raw_positive_regression
            else "stop_on_follow_up_gate"
        )
        use_next_follow_up_command = (
            raw_positive_regression and isinstance(next_follow_up, dict)
        )
        add_reason(f"gate failed on verdict={verdict}")
        if raw_positive_regression:
            add_reason("source feature still beats the raw baseline")
            add_reason("widen seed confirmation before rejecting raw-positive candidate")
    elif source_feature_needs_review:
        action = "review_feature_swap_before_promotion"
        if not source_retained:
            add_reason("source best feature did not retain its role")
        if source_feature_verdict == "regressed":
            add_reason("source best feature regressed on fresh seeds")
        if config_improved_while_source_regressed:
            add_reason("config improved while source feature regressed")
    elif verdict == "improved":
        promote_current_best = True
        use_next_follow_up_command = isinstance(next_follow_up, dict)
        action = (
            "promote_and_broaden_after_streak"
            if improved_streak >= 2
            else "continue_fresh_seed_confirmation"
        )
        add_reason(f"source feature improved with streak={improved_streak}")
    elif verdict == "confirmed":
        promote_current_best = True
        use_next_follow_up_command = isinstance(next_follow_up, dict)
        action = "continue_confirmation_or_widen_seeds"
        add_reason("source feature stayed within the confirmation band")
    elif verdict == "regressed":
        action = "rerun_or_audit_source_feature"
        add_reason(f"source feature regressed with streak={regressed_streak}")
    else:
        action = "rerun_with_more_evidence"
        add_reason("follow-up verdict is unknown")

    if gate_failed and not source_retained:
        add_reason("source best feature did not retain its role")
    if gate_failed and source_feature_verdict == "regressed":
        add_reason("source best feature regressed on fresh seeds")
    if gate_failed and source_feature_raw_verdict == "improved":
        add_reason("source best feature remained raw-positive")
    if gate_failed and config_improved_while_source_regressed:
        add_reason("config improved while source feature regressed")

    command_usage = None
    if use_next_follow_up_command and isinstance(next_follow_up, dict):
        command_usage = next_follow_up.get("script_usage")

    local_action = action
    trajectory_action = None
    trajectory_verdict = None
    unsafe_promotion = None
    trajectory_reasons: list[str] = []
    if isinstance(follow_up_trajectory, dict):
        trajectory_action = follow_up_trajectory.get("trajectory_action")
        trajectory_verdict = follow_up_trajectory.get("trajectory_verdict")
        unsafe_promotion = bool(follow_up_trajectory.get("unsafe_promotion"))
        raw_reasons = follow_up_trajectory.get("trajectory_reasons", [])
        if isinstance(raw_reasons, list):
            trajectory_reasons = [str(reason) for reason in raw_reasons]

        for reason in trajectory_reasons:
            add_reason(f"trajectory: {reason}")

        if unsafe_promotion:
            action = str(trajectory_action or "audit_unsafe_trajectory")
            promote_current_best = False
            use_next_follow_up_command = False
            command_usage = None
            add_reason("trajectory marked promotion unsafe")
        elif trajectory_action == "confirm_trajectory_with_fresh_seeds":
            promote_current_best = True
            if (
                local_action == "promote_and_broaden_after_streak"
                and isinstance(broadened_follow_up, dict)
            ):
                action = "promote_and_broaden_after_streak"
                use_next_follow_up_command = False
                use_broadened_follow_up_command = True
                command_usage = broadened_follow_up.get("script_usage")
                family_focus = broadened_follow_up.get("feature_family_focus")
                if isinstance(family_focus, dict):
                    add_reason(
                        "family focus: {family} wins={wins} near_wins={near_wins}".format(
                            family=family_focus.get("family"),
                            wins=family_focus.get("win_count"),
                            near_wins=family_focus.get("near_win_count"),
                        )
                    )
            else:
                action = "confirm_trajectory_with_fresh_seeds"
                use_next_follow_up_command = isinstance(next_follow_up, dict)
                if use_next_follow_up_command and isinstance(next_follow_up, dict):
                    command_usage = next_follow_up.get("script_usage")
        elif trajectory_action == "collect_more_seed_evidence":
            action = "collect_more_seed_evidence"
            use_next_follow_up_command = isinstance(next_follow_up, dict)
            if use_next_follow_up_command and isinstance(next_follow_up, dict):
                command_usage = next_follow_up.get("script_usage")
        elif trajectory_action == "widen_seed_confirmation_on_raw_positive_regression":
            action = "widen_seed_confirmation_on_raw_positive_regression"
            promote_current_best = False
            use_next_follow_up_command = isinstance(next_follow_up, dict)
            if use_next_follow_up_command and isinstance(next_follow_up, dict):
                command_usage = next_follow_up.get("script_usage")
        elif trajectory_action == "reconfirm_best_raw_positive_generation":
            action = "reconfirm_best_raw_positive_generation"
            promote_current_best = False
            use_next_follow_up_command = False
            use_best_generation_follow_up_command = isinstance(
                best_generation_follow_up,
                dict,
            )
            command_usage = (
                best_generation_follow_up.get("script_usage")
                if use_best_generation_follow_up_command
                and isinstance(best_generation_follow_up, dict)
                else None
            )
        elif trajectory_action in {
            "return_to_best_generation_or_rerun",
            "inspect_trajectory",
        }:
            action = str(trajectory_action)
            promote_current_best = False
            use_next_follow_up_command = False
            command_usage = None

    return {
        "schema": "st.llm_char_vae_context.follow_up_guidance.v1",
        "action": action,
        "local_action": local_action,
        "verdict": verdict,
        "config_verdict": config_verdict,
        "source_feature_verdict": source_feature_verdict,
        "source_feature_raw_verdict": source_feature_raw_verdict,
        "source_best_feature_retained": source_retained,
        "trajectory_action": trajectory_action,
        "trajectory_verdict": trajectory_verdict,
        "unsafe_promotion": unsafe_promotion,
        "trajectory_reasons": trajectory_reasons,
        "promote_current_best": promote_current_best,
        "use_next_follow_up_command": use_next_follow_up_command,
        "use_best_generation_follow_up_command": use_best_generation_follow_up_command,
        "use_broadened_follow_up_command": use_broadened_follow_up_command,
        "gate_failed": gate_failed,
        "reasons": reasons,
        "command_usage": command_usage,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a text Z-space VAE and compare raw/reconstruction/latent "
            "context features on a real next-character prediction loss."
        )
    )
    parser.add_argument("text_or_dir", nargs="+", help="Input .txt file(s) or directories")
    parser.add_argument(
        "--features",
        default="raw,reconstruction,latent",
        help=(
            "comma-separated context features: raw,reconstruction,latent,"
            "raw_latent,reconstruction_latent"
        ),
    )
    parser.add_argument("--window-chars", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--head-init",
        choices=HEAD_INIT_CHOICES,
        default="legacy",
        help=(
            "prediction-head initialization policy; xavier rescales Linear "
            "weights to an Xavier-style max-abs limit while keeping deterministic "
            "ordering"
        ),
    )
    parser.add_argument(
        "--feature-normalize",
        choices=NORMALIZE_CHOICES,
        default="none",
        help=(
            "feature scaling before the prediction head; blocks normalizes hybrid "
            "raw/reconstruction and latent segments separately"
        ),
    )
    parser.add_argument(
        "--feature-normalize-modes",
        default=None,
        help=(
            "comma-separated feature normalization grid; overrides --feature-normalize "
            "when provided"
        ),
    )
    parser.add_argument(
        "--hybrid-latent-scale",
        type=float,
        default=1.0,
        help="multiply the latent segment of raw_latent/reconstruction_latent features",
    )
    parser.add_argument(
        "--hybrid-latent-scales",
        default=None,
        help=(
            "comma-separated scale grid for the latent segment of hybrid features; "
            "overrides --hybrid-latent-scale when provided"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--eval-samples", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--backend", default="cpu", help="cpu|wgpu|cuda|hip|auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="comma-separated seed list for sweep mode")
    parser.add_argument(
        "--min-nll-delta",
        type=float,
        default=0.0,
        help="minimum mean NLL improvement over raw required for sweep status=improved",
    )
    parser.add_argument(
        "--win-tolerance",
        type=float,
        default=1e-4,
        help="per-seed NLL tolerance for counting near-win feature ties in sweep reports",
    )
    parser.add_argument("--prompt", default="SpiralTorch ")
    parser.add_argument("--gen", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--vae-load", type=pathlib.Path, default=None)
    parser.add_argument("--vae-save", type=pathlib.Path, default=None)
    parser.add_argument("--vae-epochs", type=int, default=2)
    parser.add_argument("--vae-batches", type=int, default=16)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--vae-lr", type=float, default=1e-2)
    parser.add_argument("--vae-kl-weight", type=float, default=1e-3)
    parser.add_argument("--vae-optimizer", choices=("sgd", "adam", "rmsprop"), default="adam")
    parser.add_argument("--vae-grad-clip", default="5.0")
    parser.add_argument("--mellin", choices=("none", "constant", "ramp"), default="ramp")
    parser.add_argument("--mellin-exponent", type=float, default=1.0)
    parser.add_argument("--mellin-start", type=float, default=0.8)
    parser.add_argument("--mellin-end", type=float, default=1.2)
    parser.add_argument("--run-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--follow-up-from",
        type=pathlib.Path,
        default=None,
        help=(
            "previous aggregate summary.json or run directory; reuses its best context "
            "config as defaults for a fresh-seed confirmation run"
        ),
    )
    parser.add_argument(
        "--follow-up-fail-on-verdict",
        default=None,
        help=(
            "comma-separated follow-up verdicts that should exit non-zero after "
            "writing summary/report, e.g. regressed,unknown"
        ),
    )
    parser.add_argument(
        "--follow-up-confirm-tolerance",
        type=float,
        default=0.0,
        help=(
            "extra NLL tolerance for follow-up source comparisons; the effective "
            "confirmation band also includes current fresh-seed standard error"
        ),
    )
    parser.add_argument(
        "--follow-up-used-seeds",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_chars <= 0:
        raise ValueError("--window-chars must be > 0")
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be > 0")
    if args.hidden < 0:
        raise ValueError("--hidden must be >= 0")
    if str(args.head_init) not in HEAD_INIT_CHOICES:
        raise ValueError("--head-init must be one of: legacy|xavier")
    if args.epochs < 0 or args.vae_epochs < 0:
        raise ValueError("--epochs and --vae-epochs must be >= 0")
    if args.batches <= 0 or args.vae_batches <= 0:
        raise ValueError("--batches and --vae-batches must be > 0")
    if args.batch_size <= 0 or args.vae_batch_size <= 0:
        raise ValueError("--batch-size and --vae-batch-size must be > 0")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be positive and finite")
    if args.vae_lr <= 0.0 or not math.isfinite(args.vae_lr):
        raise ValueError("--vae-lr must be positive and finite")
    if args.vae_kl_weight < 0.0 or not math.isfinite(args.vae_kl_weight):
        raise ValueError("--vae-kl-weight must be non-negative and finite")
    if args.eval_samples <= 0:
        raise ValueError("--eval-samples must be > 0")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.gen < 0:
        raise ValueError("--gen must be >= 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.min_nll_delta < 0.0 or not math.isfinite(args.min_nll_delta):
        raise ValueError("--min-nll-delta must be non-negative and finite")
    if (
        args.follow_up_confirm_tolerance < 0.0
        or not math.isfinite(args.follow_up_confirm_tolerance)
    ):
        raise ValueError("--follow-up-confirm-tolerance must be non-negative and finite")
    if args.win_tolerance < 0.0 or not math.isfinite(args.win_tolerance):
        raise ValueError("--win-tolerance must be non-negative and finite")
    if args.hybrid_latent_scale < 0.0 or not math.isfinite(args.hybrid_latent_scale):
        raise ValueError("--hybrid-latent-scale must be non-negative and finite")


def _run_single(args: argparse.Namespace, features: list[str]) -> dict[str, Any]:
    vae_grad_clip = _normalise_grad_clip(str(args.vae_grad_clip))

    data_paths = [pathlib.Path(value) for value in args.text_or_dir]
    data_files = _collect_text_files(data_paths)
    if not data_files:
        raise ValueError("no .txt files found in inputs")
    text = "\n\n".join(_read_text(path) for path in data_files)
    if len(text) <= int(args.window_chars):
        raise ValueError(
            f"combined text must be longer than --window-chars ({len(text)} <= {args.window_chars})"
        )
    train_text, val_text = _split_text(text, float(args.val_ratio), int(args.window_chars))
    symbols, index = _build_vocab(text)
    vocab_size = len(symbols)

    run_dir = args.run_dir or _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    (run_dir / "data_files.txt").write_text(
        "\n".join(str(path) for path in data_files) + "\n",
        encoding="utf-8",
    )

    if args.vae_load is not None:
        vae = st.nn.ZSpaceTextVae.load(str(args.vae_load))
    else:
        vae = st.nn.ZSpaceTextVae(
            int(args.window_chars),
            int(args.latent_dim),
            curvature=float(args.curvature),
            temperature=float(args.temperature),
            seed=int(args.seed),
        )
    vae.configure_optimizer(
        optimizer=str(args.vae_optimizer),
        grad_clip=vae_grad_clip,
    )
    basis = _build_mellin_basis(vae, args)

    print(
        f"files={len(data_files)} chars={len(text)} vocab={vocab_size} "
        f"window_chars={vae.window_chars} latent_dim={vae.latent_dim} features={','.join(features)} "
        f"feature_normalize={args.feature_normalize} hybrid_latent_scale={args.hybrid_latent_scale} "
        f"head_init={args.head_init} epochs={args.epochs} "
        f"vae_epochs={args.vae_epochs} run_dir={run_dir}",
        flush=True,
    )

    started = time.time()
    vae_history = _train_vae(vae, basis, train_text, args)
    vae_save = args.vae_save or (run_dir / "text_vae_weights.bin")
    vae.save(str(vae_save))
    feature_diagnostics = _feature_diagnostics(
        vae,
        basis,
        features,
        str(args.feature_normalize),
        float(args.hybrid_latent_scale),
        val_text,
        index,
        int(args.window_chars),
        int(args.eval_samples),
        int(args.seed) + 700_000,
    )

    run_meta = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "arch": "llm_char_vae_context",
        "data_paths": [str(path) for path in data_paths],
        "data_file_count": len(data_files),
        "text_chars": len(text),
        "train_chars": len(train_text),
        "validation_chars": len(val_text),
        "window_chars": int(vae.window_chars),
        "input_dim": int(vae.input_dim),
        "latent_dim": int(vae.latent_dim),
        "features": features,
        "feature_normalize": str(args.feature_normalize),
        "hybrid_latent_scale": float(args.hybrid_latent_scale),
        "hidden": int(args.hidden),
        "head_init": str(args.head_init),
        "epochs": int(args.epochs),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "eval_samples": int(args.eval_samples),
        "curvature": float(args.curvature),
        "temperature": float(args.temperature),
        "backend": str(args.backend),
        "seed": int(args.seed),
        "run_dir": str(run_dir),
        "vocab_size": vocab_size,
        "vae": {
            "load_path": str(args.vae_load) if args.vae_load is not None else None,
            "save_path": str(vae_save),
            "epochs": int(args.vae_epochs),
            "batches": int(args.vae_batches),
            "batch_size": int(args.vae_batch_size),
            "lr": float(args.vae_lr),
            "kl_weight": float(args.vae_kl_weight),
            "optimizer": str(args.vae_optimizer),
            "grad_clip": vae_grad_clip,
            "history": vae_history,
        },
        "mellin": {
            "mode": str(args.mellin),
            "exponent": float(args.mellin_exponent) if args.mellin == "constant" else None,
            "start": float(args.mellin_start) if args.mellin == "ramp" else None,
            "end": float(args.mellin_end) if args.mellin == "ramp" else None,
        },
    }
    _write_json(run_dir / "run.json", run_meta)

    results = []
    for feature in features:
        results.append(
            _train_feature_head(
                feature,
                vae,
                basis,
                train_text,
                val_text,
                symbols,
                index,
                args,
                run_dir,
            )
        )

    ranked = sorted(
        results,
        key=lambda item: (
            float("inf")
            if item["best_validation"]["mean_nll"] is None
            else float(item["best_validation"]["mean_nll"]),
            item["feature"],
        ),
    )
    best = ranked[0] if ranked else None
    raw = next((item for item in results if item["feature"] == FEATURE_RAW), None)
    raw_best_nll = (
        None
        if raw is None or raw["best_validation"]["mean_nll"] is None
        else float(raw["best_validation"]["mean_nll"])
    )
    raw_curve_nll = (
        None
        if raw is None or raw.get("validation_nll_mean") is None
        else float(raw["validation_nll_mean"])
    )
    deltas = {}
    if raw_best_nll is not None:
        for item in results:
            best_nll = item["best_validation"]["mean_nll"]
            if best_nll is not None:
                deltas[f"{item['feature']}_best_nll_vs_raw"] = float(best_nll) - raw_best_nll
    if raw_curve_nll is not None:
        for item in results:
            curve_nll = item.get("validation_nll_mean")
            if curve_nll is not None:
                deltas[f"{item['feature']}_validation_nll_mean_vs_raw"] = (
                    float(curve_nll) - raw_curve_nll
                )

    summary = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "elapsed_seconds": time.time() - started,
        "run": run_meta,
        "features": results,
        "feature_diagnostics": feature_diagnostics,
        "ranking": [
            {
                "feature": item["feature"],
                "best_epoch": item["best_epoch"],
                "best_step": item["best_step"],
                "best_mean_nll": item["best_validation"]["mean_nll"],
                "best_accuracy": item["best_validation"]["accuracy"],
                "validation_nll_mean": item["validation_nll_mean"],
                "validation_nll_mean_delta_vs_raw": deltas.get(
                    f"{item['feature']}_validation_nll_mean_vs_raw"
                ),
                "validation_nll_initial_minus_best": item[
                    "validation_nll_initial_minus_best"
                ],
                "validation_nll_final_minus_best": item[
                    "validation_nll_final_minus_best"
                ],
            }
            for item in ranked
        ],
        "best_feature": best["feature"] if best is not None else None,
        "deltas": deltas,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_text(run_dir / "report.md", _single_report(summary))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if best is not None:
        print(
            "best_feature={feature} best_nll={nll:.6f} best_acc={acc:.2f}% summary_json={path}".format(
                feature=best["feature"],
                nll=float(best["best_validation"]["mean_nll"]),
                acc=float(best["best_validation"]["accuracy"]) * 100.0,
                path=run_dir / "summary.json",
            ),
            flush=True,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    follow_up = _apply_follow_up_defaults(args, raw_argv)
    _validate_args(args)
    features = _parse_features(str(args.features))
    seeds = _parse_seeds(int(args.seed), args.seeds)
    normalize_modes = _parse_feature_normalize_modes(
        str(args.feature_normalize),
        args.feature_normalize_modes,
    )
    scales = _parse_hybrid_latent_scales(
        float(args.hybrid_latent_scale),
        args.hybrid_latent_scales,
    )
    follow_up_fail_on_verdicts = _parse_follow_up_fail_verdicts(
        args.follow_up_fail_on_verdict,
    )
    if follow_up is not None:
        follow_up["resolved"] = {
            "features": features,
            "feature_normalize_modes": normalize_modes,
            "hybrid_latent_scales": scales,
            "head_init": str(args.head_init),
            "seeds": seeds,
        }

    if len(seeds) == 1 and len(scales) == 1 and len(normalize_modes) == 1 and follow_up is None:
        args.seed = seeds[0]
        args.feature_normalize = normalize_modes[0]
        args.hybrid_latent_scale = scales[0]
        _run_single(args, features)
        return 0

    root_run_dir = args.run_dir or _default_run_dir()
    root_run_dir.mkdir(parents=True, exist_ok=True)
    (root_run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    run_budget = _args_run_budget(args)
    aggregate_run = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "arch": "llm_char_vae_context_sweep",
        **run_budget,
        "budget": run_budget,
        "features": features,
        "feature_normalize": normalize_modes[0] if len(normalize_modes) == 1 else None,
        "feature_normalize_modes": normalize_modes,
        "normalize_count": len(normalize_modes),
        "hybrid_latent_scale": scales[0] if len(scales) == 1 else None,
        "hybrid_latent_scales": scales,
        "scale_count": len(scales),
        "config_count": len(normalize_modes) * len(scales),
        "run_count": len(normalize_modes) * len(scales) * len(seeds),
        "seeds": seeds,
        "seed_count": len(seeds),
        "head_init": str(args.head_init),
        "min_nll_delta": float(args.min_nll_delta),
        "follow_up_confirm_tolerance": float(args.follow_up_confirm_tolerance),
        "win_tolerance": float(args.win_tolerance),
        "follow_up_fail_on_verdicts": follow_up_fail_on_verdicts,
        "run_dir": str(root_run_dir),
    }
    if follow_up is not None:
        aggregate_run["follow_up"] = follow_up
    _write_json(root_run_dir / "run.json", aggregate_run)

    summaries = []
    runs_jsonl = root_run_dir / "runs.jsonl"
    if runs_jsonl.exists():
        runs_jsonl.unlink()
    started = time.time()
    config_summaries = []
    for mode in normalize_modes:
        for scale in scales:
            config_run_dir = _config_run_dir(root_run_dir, mode, scale, normalize_modes, scales)
            config_run_dir.mkdir(parents=True, exist_ok=True)
            config_seed_summaries = []
            for seed in seeds:
                seed_dir = _seed_run_dir(config_run_dir, seed)
                print(
                    "sweep_normalize={mode} sweep_scale={scale:.6g} "
                    "sweep_seed={seed} run_dir={run_dir}".format(
                        mode=mode,
                        scale=scale,
                        seed=seed,
                        run_dir=seed_dir,
                    ),
                    flush=True,
                )
                seed_args = _clone_args(
                    args,
                    seed=seed,
                    feature_normalize=mode,
                    hybrid_latent_scale=scale,
                    run_dir=seed_dir,
                    vae_save=None,
                    json=False,
                )
                summary = _run_single(seed_args, features)
                summaries.append(summary)
                config_seed_summaries.append(summary)
                _append_jsonl(
                    runs_jsonl,
                    {
                        "seed": seed,
                        "feature_normalize": mode,
                        "hybrid_latent_scale": scale,
                        "run_dir": str(seed_dir),
                        "summary_path": str(seed_dir / "summary.json"),
                        "best_feature": summary.get("best_feature"),
                        "ranking": summary.get("ranking", []),
                        "deltas": summary.get("deltas", {}),
                    },
                )

            config_aggregate = _aggregate_summaries(
                config_seed_summaries,
                min_nll_delta=float(args.min_nll_delta),
                win_tolerance=float(args.win_tolerance),
            )
            config_summaries.append(
                {
                    "feature_normalize": mode,
                    "hybrid_latent_scale": scale,
                    "run_dir": str(config_run_dir),
                    "seed_count": len(config_seed_summaries),
                    **config_aggregate,
                }
            )

    scale_summaries = config_summaries if len(normalize_modes) == 1 else []

    aggregate = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "aggregate": True,
        "elapsed_seconds": time.time() - started,
        "run": aggregate_run,
        "seed_summaries": [
            {
                "seed": summary.get("run", {}).get("seed"),
                "feature_normalize": summary.get("run", {}).get("feature_normalize"),
                "hybrid_latent_scale": summary.get("run", {}).get("hybrid_latent_scale"),
                "run_dir": str(summary.get("run", {}).get("run_dir")),
                "best_feature": summary.get("best_feature"),
                "ranking": summary.get("ranking", []),
                "deltas": summary.get("deltas", {}),
            }
            for summary in summaries
        ],
        "config_summaries": config_summaries,
        "scale_summaries": scale_summaries,
    }
    if follow_up is not None:
        aggregate["follow_up"] = follow_up
    aggregate.update(
        _aggregate_summaries(
            summaries,
            min_nll_delta=float(args.min_nll_delta),
            win_tolerance=float(args.win_tolerance),
        )
    )
    best_config = _best_config_summary(config_summaries)
    next_follow_up = None
    if best_config is not None:
        aggregate["best_config"] = best_config
        aggregate["best_feature_normalize"] = best_config.get("feature_normalize")
        aggregate["best_hybrid_latent_scale"] = best_config.get("hybrid_latent_scale")
        next_follow_up = _next_follow_up_command_record(
            args,
            features,
            best_config,
            root_run_dir,
            seeds,
            follow_up,
        )
        if next_follow_up is not None:
            aggregate["next_follow_up_command"] = next_follow_up
            _write_next_follow_up_script(next_follow_up)
    follow_up_result = _follow_up_result(
        follow_up,
        config_summaries,
        best_config,
        min_nll_delta=float(args.min_nll_delta),
        follow_up_confirm_tolerance=float(args.follow_up_confirm_tolerance),
        current_run_budget=run_budget,
    )
    if follow_up_result is not None:
        aggregate["follow_up_result"] = follow_up_result
    follow_up_chain = _follow_up_chain_record(
        follow_up,
        follow_up_result,
        root_run_dir,
    )
    if follow_up_chain is not None:
        aggregate["follow_up_chain"] = follow_up_chain
    follow_up_ancestors = _follow_up_ancestor_records(follow_up_chain)
    if follow_up_ancestors is not None:
        aggregate["follow_up_ancestors"] = follow_up_ancestors
    follow_up_gate = _follow_up_gate_record(
        follow_up_result,
        follow_up_fail_on_verdicts,
    )
    if follow_up_gate is not None:
        aggregate["follow_up_gate"] = follow_up_gate

    preliminary_guidance = _follow_up_guidance_record(
        follow_up_result,
        follow_up_chain,
        follow_up_gate,
        next_follow_up,
    )
    if preliminary_guidance is not None:
        aggregate["follow_up_guidance"] = preliminary_guidance
    follow_up_trajectory = _follow_up_trajectory_record(
        root_run_dir,
        aggregate,
        follow_up_ancestors,
        min_nll_delta=float(args.min_nll_delta),
    )
    if follow_up_trajectory is not None:
        aggregate["follow_up_trajectory"] = follow_up_trajectory

    best_generation_follow_up = _best_generation_follow_up_command_record(
        args,
        features,
        root_run_dir,
        seeds,
        follow_up_trajectory,
        next_follow_up,
        follow_up_result,
    )
    if best_generation_follow_up is not None:
        aggregate["best_generation_follow_up_command"] = best_generation_follow_up
        _write_next_follow_up_script(best_generation_follow_up)

    broadened_follow_up = _broadened_follow_up_command_record(
        args,
        features,
        best_config,
        root_run_dir,
        seeds,
        follow_up_chain,
        follow_up_trajectory,
        next_follow_up,
        aggregate.get("feature_family_stability"),
    )
    if broadened_follow_up is not None:
        aggregate["broadened_follow_up_command"] = broadened_follow_up
        _write_next_follow_up_script(broadened_follow_up)

    follow_up_guidance = _follow_up_guidance_record(
        follow_up_result,
        follow_up_chain,
        follow_up_gate,
        next_follow_up,
        follow_up_trajectory,
        best_generation_follow_up,
        broadened_follow_up,
    )
    if follow_up_guidance is not None:
        aggregate["follow_up_guidance"] = follow_up_guidance
    selected_guided_follow_up = next_follow_up
    if isinstance(follow_up_guidance, dict):
        if follow_up_guidance.get("use_best_generation_follow_up_command"):
            selected_guided_follow_up = best_generation_follow_up
        elif follow_up_guidance.get("use_broadened_follow_up_command"):
            selected_guided_follow_up = broadened_follow_up
    guided_next_follow_up = _guided_next_follow_up_command_record(
        root_run_dir,
        follow_up_guidance,
        selected_guided_follow_up,
    )
    if guided_next_follow_up is not None:
        aggregate["guided_next_follow_up_command"] = guided_next_follow_up
        _write_guided_next_follow_up_script(guided_next_follow_up)
    follow_up_trajectory = _follow_up_trajectory_record(
        root_run_dir,
        aggregate,
        follow_up_ancestors,
        min_nll_delta=float(args.min_nll_delta),
    )
    if follow_up_trajectory is not None:
        aggregate["follow_up_trajectory"] = follow_up_trajectory
    _write_json(root_run_dir / "summary.json", aggregate)
    _write_text(root_run_dir / "report.md", _aggregate_report(aggregate))
    if args.json:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))

    config_text = "-"
    if best_config is not None:
        config_text = "{feature}@normalize={normalize},scale={scale}".format(
            feature=best_config.get("best_feature"),
            normalize=best_config.get("feature_normalize"),
            scale=best_config.get("hybrid_latent_scale"),
        )
    print(
        (
            "sweep_status={status} best_feature={feature} best_config={config} "
            "normalizes={normalizes} scales={scales} seeds={seeds} summary_json={path}"
        ).format(
            status=aggregate["status"],
            feature=aggregate["best_feature"],
            config=config_text,
            normalizes=",".join(normalize_modes),
            scales=",".join(f"{scale:.6g}" for scale in scales),
            seeds=",".join(str(seed) for seed in seeds),
            path=root_run_dir / "summary.json",
        ),
        flush=True,
    )
    if isinstance(follow_up_gate, dict) and follow_up_gate.get("failed"):
        print(
            "follow_up_gate_failed verdict={verdict} fail_on={fail_on} summary_json={path}".format(
                verdict=follow_up_gate.get("verdict"),
                fail_on=",".join(str(item) for item in follow_up_fail_on_verdicts),
                path=root_run_dir / "summary.json",
            ),
            flush=True,
        )
        return int(follow_up_gate.get("exit_code") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
