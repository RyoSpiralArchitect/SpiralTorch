from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
import random
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
FEATURE_CHOICES = (FEATURE_RAW, FEATURE_RECONSTRUCTION, FEATURE_LATENT)

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


def _feature_vector(model: Any, basis: Any | None, feature: str, text: str) -> list[float]:
    if feature == FEATURE_RAW:
        if basis is None:
            values = model.encode_text(text)
        else:
            values = model.encode_text_with_mellin(text, basis)
        return [float(value) for value in values]

    if basis is None:
        state = model.forward_mean_text(text)
    else:
        state = model.forward_mean_text_with_mellin(text, basis)
    values = state.latent if feature == FEATURE_LATENT else state.reconstruction
    return [float(value) for value in values]


def _feature_tensor(
    model: Any,
    basis: Any | None,
    feature: str,
    samples: list[_WindowSample],
) -> st.Tensor:
    dim = _feature_dim(model, feature)
    data: list[float] = []
    for sample in samples:
        values = _feature_vector(model, basis, feature, sample.window)
        if len(values) != dim:
            raise ValueError(f"{feature} feature length mismatch: expected {dim}, got {len(values)}")
        data.extend(values)
    return st.Tensor(len(samples), dim, data)


def _build_batches(
    vae: Any,
    basis: Any | None,
    feature: str,
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
        out.append((_feature_tensor(vae, basis, feature, samples), _one_hot_targets(samples, vocab_size)))
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


def _evaluate(
    head: Any,
    vae: Any,
    basis: Any | None,
    feature: str,
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
    x = _feature_tensor(vae, basis, feature, samples)
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
        x = _feature_tensor(vae, basis, feature, [sample])
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


def _aggregate_summaries(
    summaries: list[dict[str, Any]],
    *,
    min_nll_delta: float,
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
        "best_feature": best_feature,
        "status": status,
    }


def _seed_run_dir(root: pathlib.Path, seed: int) -> pathlib.Path:
    if seed < 0:
        return root / f"seed_neg_{abs(seed):06d}"
    return root / f"seed_{seed:06d}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a text Z-space VAE and compare raw/reconstruction/latent "
            "context features on a real next-character prediction loss."
        )
    )
    parser.add_argument("text_or_dir", nargs="+", help="Input .txt file(s) or directories")
    parser.add_argument("--features", default="raw,reconstruction,latent")
    parser.add_argument("--window-chars", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
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
        f"epochs={args.epochs} vae_epochs={args.vae_epochs} run_dir={run_dir}",
        flush=True,
    )

    started = time.time()
    vae_history = _train_vae(vae, basis, train_text, args)
    vae_save = args.vae_save or (run_dir / "text_vae_weights.bin")
    vae.save(str(vae_save))

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
    args = parser.parse_args(argv)
    _validate_args(args)
    features = _parse_features(str(args.features))
    seeds = _parse_seeds(int(args.seed), args.seeds)

    if len(seeds) == 1:
        args.seed = seeds[0]
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
        "seeds": seeds,
        "seed_count": len(seeds),
        "min_nll_delta": float(args.min_nll_delta),
        "run_dir": str(root_run_dir),
    }
    _write_json(root_run_dir / "run.json", aggregate_run)

    summaries = []
    runs_jsonl = root_run_dir / "runs.jsonl"
    if runs_jsonl.exists():
        runs_jsonl.unlink()
    started = time.time()
    for seed in seeds:
        seed_dir = _seed_run_dir(root_run_dir, seed)
        print(f"sweep_seed={seed} run_dir={seed_dir}", flush=True)
        seed_args = _clone_args(
            args,
            seed=seed,
            run_dir=seed_dir,
            vae_save=None,
            json=False,
        )
        summary = _run_single(seed_args, features)
        summaries.append(summary)
        _append_jsonl(
            runs_jsonl,
            {
                "seed": seed,
                "run_dir": str(seed_dir),
                "summary_path": str(seed_dir / "summary.json"),
                "best_feature": summary.get("best_feature"),
                "ranking": summary.get("ranking", []),
                "deltas": summary.get("deltas", {}),
            },
        )

    aggregate = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "aggregate": True,
        "elapsed_seconds": time.time() - started,
        "run": aggregate_run,
        "seed_summaries": [
            {
                "seed": summary.get("run", {}).get("seed"),
                "run_dir": str(_seed_run_dir(root_run_dir, int(summary.get("run", {}).get("seed", 0)))),
                "best_feature": summary.get("best_feature"),
                "ranking": summary.get("ranking", []),
                "deltas": summary.get("deltas", {}),
            }
            for summary in summaries
        ],
    }
    aggregate.update(
        _aggregate_summaries(
            summaries,
            min_nll_delta=float(args.min_nll_delta),
        )
    )
    _write_json(root_run_dir / "summary.json", aggregate)
    if args.json:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))

    print(
        "sweep_status={status} best_feature={feature} seeds={seeds} summary_json={path}".format(
            status=aggregate["status"],
            feature=aggregate["best_feature"],
            seeds=",".join(str(seed) for seed in seeds),
            path=root_run_dir / "summary.json",
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
