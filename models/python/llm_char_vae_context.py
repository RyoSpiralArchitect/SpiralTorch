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
FEATURE_CHOICES = (
    FEATURE_RAW,
    FEATURE_RECONSTRUCTION,
    FEATURE_LATENT,
    FEATURE_RAW_LATENT,
    FEATURE_RECONSTRUCTION_LATENT,
)
NORMALIZE_CHOICES = ("none", "vector", "blocks")

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
    return {
        "feature": feature,
        "feature_dim": feature_dim,
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "final_validation": history[-1]["validation"] if history else initial_validation,
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
            "min": None,
            "max": None,
        }
    mean = sum(vals) / float(len(vals))
    variance = sum((value - mean) ** 2 for value in vals) / float(len(vals))
    return {
        "count": len(vals),
        "mean": mean,
        "stddev": math.sqrt(variance),
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
        f"- window_chars: {run.get('window_chars')}",
        f"- latent_dim: {run.get('latent_dim')}",
        f"- summary_json: `summary.json`",
        "",
        "## Ranking",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["feature", "best_epoch", "best_nll", "best_acc", "init_nll", "final_nll"],
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
        f"- min_nll_delta: {run.get('min_nll_delta')}",
        f"- win_tolerance: {run.get('win_tolerance')}",
        f"- summary_json: `summary.json`",
        "",
    ]
    follow_up = summary.get("follow_up")
    follow_up_result = summary.get("follow_up_result")
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
                f"- source_best_feature_retained: {follow_up_result.get('source_best_feature_retained')}",
                f"- match_found: {follow_up_result.get('match_found')}",
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
            ["feature", "mean_best_nll", "mean_best_acc", "mean_delta_vs_raw", "runs"],
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
                f"- default_new_seeds: {next_follow_up.get('default_new_seeds')}",
                f"- script: `{next_follow_up.get('script_path')}`",
                f"- usage: `{next_follow_up.get('script_usage')}`",
                "",
                "```bash",
                str(next_follow_up.get("shell_command")),
                "```",
            ]
        )
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
        deltas = [
            float(summary.get("deltas", {}).get(f"{feature}_best_nll_vs_raw"))
            for summary in summaries
            if summary.get("deltas", {}).get(f"{feature}_best_nll_vs_raw") is not None
        ]
        feature_rows.append(
            {
                "feature": feature,
                "runs": len(feature_results),
                "best_nll": _metric_stats(best_nlls),
                "best_accuracy": _metric_stats(best_accs),
                "best_nll_delta_vs_raw": _metric_stats(deltas),
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
                "runs": item["runs"],
            }
            for item in ranking
        ],
        "seed_winners": seed_winners,
        "feature_stability": feature_stability,
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


def _fresh_seed_csv(seeds: list[int]) -> str:
    used = set(int(seed) for seed in seeds)
    candidates = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
    count = max(3, len(seeds))
    fresh = [seed for seed in candidates if seed not in used][:count]
    if len(fresh) < count:
        cursor = 1_001
        while len(fresh) < count:
            if cursor not in used:
                fresh.append(cursor)
            cursor += 2
    return ",".join(str(seed) for seed in fresh)


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
        ("--window-chars", int(args.window_chars)),
        ("--latent-dim", int(args.latent_dim)),
        ("--hidden", int(args.hidden)),
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
) -> dict[str, Any] | None:
    if not best_config:
        return None
    default_new_seeds = _fresh_seed_csv(seeds)
    default_run_dir = root_run_dir / "follow_up_best_config"
    default_follow_up_from = root_run_dir / "summary.json"
    script_path = root_run_dir / "next_follow_up_command.sh"
    literal_command = _follow_up_command_parts(
        args,
        features,
        best_config,
        seeds_value=default_new_seeds,
        run_dir_value=str(default_run_dir),
        follow_up_from_value=str(default_follow_up_from),
    )
    shell_command = "PYTHONNOUSERSITE=1 " + shlex.join(literal_command)
    script_command = _follow_up_command_parts(
        args,
        features,
        best_config,
        seeds_value="${NEW_SEEDS}",
        run_dir_value="${NEXT_RUN_DIR}",
        follow_up_from_value="${FOLLOW_UP_FROM}",
    )
    return {
        "schema": "st.llm_char_vae_context.next_follow_up_command.v1",
        "action": "confirm_best_config_fresh_seeds",
        "best_config": best_config,
        "default_new_seeds": default_new_seeds,
        "default_run_dir": str(default_run_dir),
        "default_follow_up_from": str(default_follow_up_from),
        "script_path": str(script_path),
        "shell_command": shell_command,
        "script_command": script_command,
        "script_usage": (
            f"FOLLOW_UP_FROM={default_follow_up_from} NEW_SEEDS={default_new_seeds} "
            f"NEXT_RUN_DIR={default_run_dir} "
            f"bash {script_path}"
        ),
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
            )
            for part in quoted
        )
    )


def _write_next_follow_up_script(record: dict[str, Any]) -> None:
    script_path = pathlib.Path(str(record["script_path"]))
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"FOLLOW_UP_FROM=\"${{FOLLOW_UP_FROM:-{record['default_follow_up_from']}}}\"",
            f"NEW_SEEDS=\"${{NEW_SEEDS:-{record['default_new_seeds']}}}\"",
            f"NEXT_RUN_DIR=\"${{NEXT_RUN_DIR:-{record['default_run_dir']}}}\"",
            _script_command_line([str(part) for part in record["script_command"]]),
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
    next_follow_up = summary.get("next_follow_up_command")
    if isinstance(next_follow_up, dict):
        raw = next_follow_up.get("default_new_seeds")
        if raw is not None and str(raw).strip():
            return str(raw)
    source_seeds = _summary_seeds(summary)
    return _fresh_seed_csv(source_seeds or [42])


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

    explicit_features = _flag_present(argv, "--features")
    explicit_normalize = _flag_present(argv, "--feature-normalize") or _flag_present(
        argv,
        "--feature-normalize-modes",
    )
    explicit_scale = _flag_present(argv, "--hybrid-latent-scale") or _flag_present(
        argv,
        "--hybrid-latent-scales",
    )
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
        "applied_defaults": applied_defaults,
        "user_overrides": {
            "features": explicit_features,
            "feature_normalize": explicit_normalize,
            "hybrid_latent_scale": explicit_scale,
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


def _compact_config_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    top = config.get("ranking", [{}])[0] if config.get("ranking") else {}
    return {
        "feature_normalize": config.get("feature_normalize"),
        "hybrid_latent_scale": config.get("hybrid_latent_scale"),
        "best_feature": config.get("best_feature"),
        "status": config.get("status"),
        "mean_best_nll": top.get("mean_best_nll"),
        "mean_best_accuracy": top.get("mean_best_accuracy"),
        "mean_best_nll_delta_vs_raw": top.get("mean_best_nll_delta_vs_raw"),
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


def _follow_up_result(
    follow_up: dict[str, Any] | None,
    config_summaries: list[dict[str, Any]],
    current_best_config: dict[str, Any] | None,
    *,
    min_nll_delta: float,
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
    config_delta = None
    source_feature_delta = None
    if source_nll is not None and evaluated_nll is not None:
        config_delta = evaluated_nll - source_nll
    if source_nll is not None and source_feature_nll is not None:
        source_feature_delta = source_feature_nll - source_nll

    config_verdict = _delta_verdict(config_delta, min_nll_delta)
    source_feature_verdict = _delta_verdict(source_feature_delta, min_nll_delta)
    source_feature = source_best_config.get("best_feature")
    source_best_feature_retained = (
        evaluated is not None
        and source_feature is not None
        and evaluated.get("best_feature") == source_feature
    )

    return {
        "schema": "st.llm_char_vae_context.follow_up_result.v1",
        "source_summary_path": follow_up.get("source_summary_path"),
        "source_best_config": source_best_config,
        "evaluated_config": evaluated,
        "source_feature_evaluated": source_feature_evaluated,
        "current_best_config": current_best_config,
        "mean_best_nll_delta_vs_source": config_delta,
        "source_feature_mean_best_nll_delta_vs_source": source_feature_delta,
        "config_verdict": config_verdict,
        "source_feature_verdict": source_feature_verdict,
        "source_best_feature_retained": source_best_feature_retained,
        "verdict": source_feature_verdict
        if source_feature_verdict != "unknown"
        else config_verdict,
        "match_found": evaluated is not None,
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
    parser.add_argument("--json", action="store_true", help="Print summary JSON")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_chars <= 0:
        raise ValueError("--window-chars must be > 0")
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be > 0")
    if args.hidden < 0:
        raise ValueError("--hidden must be >= 0")
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
        f"epochs={args.epochs} vae_epochs={args.vae_epochs} run_dir={run_dir}",
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
        None if raw is None or raw["best_validation"]["mean_nll"] is None else float(raw["best_validation"]["mean_nll"])
    )
    deltas = {}
    if raw_best_nll is not None:
        for item in results:
            best_nll = item["best_validation"]["mean_nll"]
            if best_nll is not None:
                deltas[f"{item['feature']}_best_nll_vs_raw"] = float(best_nll) - raw_best_nll

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
                "best_mean_nll": item["best_validation"]["mean_nll"],
                "best_accuracy": item["best_validation"]["accuracy"],
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
    if follow_up is not None:
        follow_up["resolved"] = {
            "features": features,
            "feature_normalize_modes": normalize_modes,
            "hybrid_latent_scales": scales,
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
    aggregate_run = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "arch": "llm_char_vae_context_sweep",
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
        "min_nll_delta": float(args.min_nll_delta),
        "win_tolerance": float(args.win_tolerance),
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
        )
        if next_follow_up is not None:
            aggregate["next_follow_up_command"] = next_follow_up
            _write_next_follow_up_script(next_follow_up)
    follow_up_result = _follow_up_result(
        follow_up,
        config_summaries,
        best_config,
        min_nll_delta=float(args.min_nll_delta),
    )
    if follow_up_result is not None:
        aggregate["follow_up_result"] = follow_up_result
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
