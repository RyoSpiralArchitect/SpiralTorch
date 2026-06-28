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

FORMAT = "st-zspace-text-vae-compare-v1"
RUN_SCHEMA = "st.modelzoo.run.v1"

_TEXT_EXTS = {".txt"}


@dataclass(frozen=True)
class _TextWindow:
    window: str
    target: str


@dataclass(frozen=True)
class _FeatureWindow:
    window: str
    target: str
    raw: list[float]
    reconstruction: list[float]
    latent: list[float]


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


def _write_jsonl(path: pathlib.Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pick_window_with_target(text: str, window_chars: int, rng: random.Random) -> _TextWindow:
    if len(text) <= window_chars:
        raise ValueError(
            f"text too short for window_chars={window_chars}: len(text)={len(text)}"
        )
    start = rng.randrange(0, len(text) - window_chars)
    end = start + window_chars
    return _TextWindow(window=text[start:end], target=text[end])


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


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def _squared_l2(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _cosine(left: list[float], right: list[float]) -> float:
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
    return dot / denom if denom > 0.0 else 0.0


def _feature_variance(samples: list[_FeatureWindow], field: str, threshold: float) -> dict[str, Any]:
    vectors = [list(getattr(sample, field)) for sample in samples]
    if not vectors:
        return {
            "dims": 0,
            "variance_mean": 0.0,
            "active_dim_fraction": 0.0,
            "rms": 0.0,
            "norm_mean": 0.0,
        }

    dims = min(len(vec) for vec in vectors)
    if dims == 0:
        return {
            "dims": 0,
            "variance_mean": 0.0,
            "active_dim_fraction": 0.0,
            "rms": 0.0,
            "norm_mean": 0.0,
        }

    means = [_mean(vec[idx] for vec in vectors) for idx in range(dims)]
    variances = []
    total_sq = 0.0
    norms = []
    for vec in vectors:
        norm_sq = 0.0
        for idx in range(dims):
            value = float(vec[idx])
            total_sq += value * value
            norm_sq += value * value
        norms.append(math.sqrt(norm_sq))
    for idx in range(dims):
        variances.append(_mean((float(vec[idx]) - means[idx]) ** 2 for vec in vectors))

    active = sum(1 for value in variances if value > threshold)
    return {
        "dims": dims,
        "variance_mean": _mean(variances),
        "variance_max": max(variances) if variances else 0.0,
        "active_dim_fraction": active / float(dims),
        "rms": math.sqrt(total_sq / float(len(vectors) * dims)),
        "norm_mean": _mean(norms),
    }


def _reconstruction_summary(samples: list[_FeatureWindow]) -> dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "mse": 0.0,
            "l2": 0.0,
            "cosine": 0.0,
        }

    mses = []
    l2s = []
    cosines = []
    for sample in samples:
        dims = min(len(sample.raw), len(sample.reconstruction))
        if dims == 0:
            continue
        sq = _squared_l2(sample.raw[:dims], sample.reconstruction[:dims])
        mses.append(sq / float(dims))
        l2s.append(math.sqrt(sq))
        cosines.append(_cosine(sample.raw[:dims], sample.reconstruction[:dims]))

    return {
        "samples": len(mses),
        "mse": _mean(mses),
        "l2": _mean(l2s),
        "cosine": _mean(cosines),
    }


def _nearest_metrics(
    train: list[_FeatureWindow],
    eval_samples: list[_FeatureWindow],
    field: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    if not train or not eval_samples:
        return {
            "samples": 0,
            "top1_accuracy": 0.0,
            "topk_accuracy": 0.0,
            "top_k": top_k,
            "mean_nearest_distance": 0.0,
            "mean_target_rank": None,
            "target_coverage": 0.0,
        }

    total = 0
    top1 = 0
    topk = 0
    nearest_distances = []
    target_ranks = []
    k = max(1, int(top_k))

    for sample in eval_samples:
        vector = list(getattr(sample, field))
        distances = []
        for idx, candidate in enumerate(train):
            candidate_vector = list(getattr(candidate, field))
            dims = min(len(vector), len(candidate_vector))
            if dims == 0:
                continue
            distances.append(
                (
                    _squared_l2(vector[:dims], candidate_vector[:dims]),
                    idx,
                    candidate.target,
                )
            )
        if not distances:
            continue

        total += 1
        distances.sort(key=lambda item: (item[0], item[1]))
        nearest = distances[0]
        nearest_distances.append(math.sqrt(max(0.0, nearest[0])))
        if nearest[2] == sample.target:
            top1 += 1
        if any(target == sample.target for _distance, _idx, target in distances[:k]):
            topk += 1
        for rank, (_distance, _idx, target) in enumerate(distances, start=1):
            if target == sample.target:
                target_ranks.append(float(rank))
                break

    if total == 0:
        target_coverage = 0.0
    else:
        target_coverage = len(target_ranks) / float(total)
    return {
        "samples": total,
        "top1_accuracy": top1 / float(total) if total else 0.0,
        "topk_accuracy": topk / float(total) if total else 0.0,
        "top_k": k,
        "mean_nearest_distance": _mean(nearest_distances),
        "mean_target_rank": _mean(target_ranks) if target_ranks else None,
        "target_coverage": target_coverage,
    }


def _window_batch(text: str, window_chars: int, count: int, rng: random.Random) -> list[str]:
    return [_pick_window_with_target(text, window_chars, rng).window for _ in range(count)]


def _collect_windows(
    text: str,
    window_chars: int,
    count: int,
    rng: random.Random,
) -> list[_TextWindow]:
    return [_pick_window_with_target(text, window_chars, rng) for _ in range(max(0, count))]


def _encode_feature_window(model: Any, basis: Any | None, sample: _TextWindow) -> _FeatureWindow:
    if basis is None:
        raw = model.encode_text(sample.window)
        state = model.forward_mean_text(sample.window)
    else:
        raw = model.encode_text_with_mellin(sample.window, basis)
        state = model.forward_mean_text_with_mellin(sample.window, basis)
    return _FeatureWindow(
        window=sample.window,
        target=sample.target,
        raw=[float(value) for value in raw],
        reconstruction=[float(value) for value in state.reconstruction],
        latent=[float(value) for value in state.latent],
    )


def _encode_feature_windows(
    model: Any,
    basis: Any | None,
    samples: list[_TextWindow],
) -> list[_FeatureWindow]:
    return [_encode_feature_window(model, basis, sample) for sample in samples]


def _evaluate_vae_batch(
    model: Any,
    basis: Any | None,
    windows: list[str],
    kl_weight: float,
) -> dict[str, Any] | None:
    if not windows:
        return None
    if basis is None:
        stats = model.evaluate_text_batch(windows, kl_weight)
    else:
        stats = model.evaluate_text_batch_with_mellin(windows, basis, kl_weight)
    return {
        "batch_size": int(stats.batch_size),
        "recon_loss": float(stats.recon_loss),
        "kl_loss": float(stats.kl_loss),
        "weighted_loss": float(stats.weighted_loss),
        "evidence_lower_bound": float(stats.evidence_lower_bound),
    }


def _train_vae(model: Any, basis: Any | None, train_text: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for epoch in range(max(0, int(args.epochs))):
        rng = random.Random(int(args.seed) + epoch * 10_000)
        recon_sum = 0.0
        kl_sum = 0.0
        weighted_sum = 0.0
        grad_sum = 0.0
        clipped_grad_sum = 0.0
        update_sum = 0.0
        last_step = 0

        for _step in range(int(args.batches)):
            windows = _window_batch(train_text, int(args.window_chars), int(args.batch_size), rng)
            if basis is None:
                stats = model.train_text_batch(windows, float(args.lr), float(args.kl_weight))
            else:
                stats = model.train_text_batch_with_mellin(
                    windows,
                    basis,
                    float(args.lr),
                    float(args.kl_weight),
                )
            recon_sum += float(stats.recon_loss)
            kl_sum += float(stats.kl_loss)
            weighted_sum += float(stats.weighted_loss)
            grad_sum += float(stats.gradient_l2)
            clipped_grad_sum += float(stats.clipped_gradient_l2)
            update_sum += float(stats.update_l2)
            last_step = int(stats.optimizer_step)

        denom = float(args.batches)
        item = {
            "epoch": epoch,
            "avg_recon_loss": recon_sum / denom,
            "avg_kl_loss": kl_sum / denom,
            "avg_weighted_loss": weighted_sum / denom,
            "avg_gradient_l2": grad_sum / denom,
            "avg_clipped_gradient_l2": clipped_grad_sum / denom,
            "avg_update_l2": update_sum / denom,
            "optimizer_step": last_step,
        }
        history.append(item)
        print(
            "epoch[{epoch}] avg_recon_loss={recon:.6f} avg_weighted_loss={weighted:.6f} "
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


def _comparison_status(
    raw: dict[str, Any],
    reconstruction: dict[str, Any],
    latent: dict[str, Any],
    min_delta: float,
) -> str:
    raw_top1 = float(raw.get("top1_accuracy", 0.0))
    recon_delta = float(reconstruction.get("top1_accuracy", 0.0)) - raw_top1
    latent_delta = float(latent.get("top1_accuracy", 0.0)) - raw_top1
    best_delta = max(recon_delta, latent_delta)
    if best_delta >= min_delta:
        return "improved" if best_delta > 0.0 else "neutral"
    if recon_delta < -min_delta and latent_delta < -min_delta:
        return "regression"
    return "neutral"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a text Z-space VAE on a real corpus and compare raw/off, "
            "reconstruction/on, and latent/on nearest-neighbor next-char proxies."
        )
    )
    parser.add_argument("text_or_dir", nargs="+", help="Input .txt file(s) or directories")
    parser.add_argument("--window-chars", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "rmsprop"), default="adam")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--rms-decay", type=float, default=0.99)
    parser.add_argument("--grad-clip", default="5.0")
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--neighbor-train-samples", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--latent-active-threshold", type=float, default=1e-8)
    parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--mellin", choices=("none", "constant", "ramp"), default="none")
    parser.add_argument("--mellin-exponent", type=float, default=1.0)
    parser.add_argument("--mellin-start", type=float, default=0.8)
    parser.add_argument("--mellin-end", type=float, default=1.2)
    parser.add_argument("--run-dir", type=pathlib.Path, default=None)
    parser.add_argument("--save", type=pathlib.Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print the final comparison JSON")
    return parser


def _normalise_grad_clip(raw: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"none", "off", "0"}:
        return None
    parsed = float(value)
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise ValueError("--grad-clip must be positive, or one of none|off|0")
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_chars <= 0:
        raise ValueError("--window-chars must be > 0")
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be > 0")
    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0")
    if args.batches <= 0:
        raise ValueError("--batches must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.lr <= 0.0 or not math.isfinite(args.lr):
        raise ValueError("--lr must be positive and finite")
    if args.kl_weight < 0.0 or not math.isfinite(args.kl_weight):
        raise ValueError("--kl-weight must be non-negative and finite")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.neighbor_train_samples <= 0:
        raise ValueError("--neighbor-train-samples must be > 0")
    if args.eval_windows <= 0:
        raise ValueError("--eval-windows must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.latent_active_threshold < 0.0 or not math.isfinite(args.latent_active_threshold):
        raise ValueError("--latent-active-threshold must be non-negative and finite")
    if args.min_accuracy_delta < 0.0 or not math.isfinite(args.min_accuracy_delta):
        raise ValueError("--min-accuracy-delta must be non-negative and finite")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    grad_clip = _normalise_grad_clip(str(args.grad_clip))

    data_paths = [pathlib.Path(value) for value in args.text_or_dir]
    data_files = _collect_text_files(data_paths)
    if not data_files:
        raise ValueError("no .txt files found in inputs")

    text = "\n\n".join(_read_text(path) for path in data_files)
    if len(text) <= args.window_chars:
        raise ValueError(
            f"combined text must be longer than --window-chars ({len(text)} <= {args.window_chars})"
        )

    split_at = int(len(text) * (1.0 - args.val_ratio)) if args.val_ratio > 0.0 else len(text)
    split_at = min(max(split_at, args.window_chars + 1), len(text))
    train_text = text[:split_at]
    eval_text = text[split_at:] if split_at < len(text) else ""
    if len(eval_text) <= args.window_chars:
        eval_text = text

    run_dir = args.run_dir or _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save or (run_dir / "text_vae_weights.bin")
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    (run_dir / "data_files.txt").write_text(
        "\n".join(str(path) for path in data_files) + "\n",
        encoding="utf-8",
    )

    model = st.nn.ZSpaceTextVae(
        int(args.window_chars),
        int(args.latent_dim),
        curvature=float(args.curvature),
        temperature=float(args.temperature),
        seed=int(args.seed),
    )
    model.configure_optimizer(
        optimizer=str(args.optimizer),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        epsilon=float(args.epsilon),
        rms_decay=float(args.rms_decay),
        grad_clip=grad_clip,
    )
    basis = _build_mellin_basis(model, args)

    run_meta = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "arch": "zspace_text_vae_compare",
        "data_paths": [str(path) for path in data_paths],
        "data_file_count": len(data_files),
        "data_files_manifest": str(run_dir / "data_files.txt"),
        "text_chars": len(text),
        "train_chars": len(train_text),
        "eval_chars": len(eval_text),
        "window_chars": int(args.window_chars),
        "input_dim": int(model.input_dim),
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "kl_weight": float(args.kl_weight),
        "optimizer": {
            "name": str(args.optimizer),
            "beta1": float(args.beta1),
            "beta2": float(args.beta2),
            "epsilon": float(args.epsilon),
            "rms_decay": float(args.rms_decay),
            "grad_clip": grad_clip,
        },
        "mellin": {
            "mode": str(args.mellin),
            "exponent": float(args.mellin_exponent) if args.mellin == "constant" else None,
            "start": float(args.mellin_start) if args.mellin == "ramp" else None,
            "end": float(args.mellin_end) if args.mellin == "ramp" else None,
        },
        "neighbor_train_samples": int(args.neighbor_train_samples),
        "eval_windows": int(args.eval_windows),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "save_path": str(save_path),
    }
    _write_json(run_dir / "run.json", run_meta)

    print(
        f"files={len(data_files)} chars={len(text)} window_chars={args.window_chars} "
        f"latent_dim={args.latent_dim} epochs={args.epochs} batches={args.batches} "
        f"batch_size={args.batch_size} optimizer={args.optimizer} mellin={args.mellin} "
        f"run_dir={run_dir}",
        flush=True,
    )

    started = time.time()
    history = _train_vae(model, basis, train_text, args)
    model.save(str(save_path))

    train_rng = random.Random(int(args.seed) + 2_000_000)
    eval_rng = random.Random(int(args.seed) + 3_000_000)
    train_windows = _collect_windows(
        train_text,
        int(args.window_chars),
        int(args.neighbor_train_samples),
        train_rng,
    )
    eval_windows = _collect_windows(
        eval_text,
        int(args.window_chars),
        int(args.eval_windows),
        eval_rng,
    )

    train_features = _encode_feature_windows(model, basis, train_windows)
    eval_features = _encode_feature_windows(model, basis, eval_windows)

    raw_metrics = _nearest_metrics(train_features, eval_features, "raw", top_k=int(args.top_k))
    reconstruction_metrics = _nearest_metrics(
        train_features,
        eval_features,
        "reconstruction",
        top_k=int(args.top_k),
    )
    latent_metrics = _nearest_metrics(
        train_features,
        eval_features,
        "latent",
        top_k=int(args.top_k),
    )
    eval_batch = _evaluate_vae_batch(
        model,
        basis,
        [sample.window for sample in eval_windows],
        float(args.kl_weight),
    )
    reconstruction_summary = _reconstruction_summary(eval_features)
    latent_summary = _feature_variance(
        eval_features,
        "latent",
        float(args.latent_active_threshold),
    )
    reconstruction_feature_summary = _feature_variance(
        eval_features,
        "reconstruction",
        float(args.latent_active_threshold),
    )

    deltas = {
        "reconstruction_top1_vs_raw": float(reconstruction_metrics["top1_accuracy"])
        - float(raw_metrics["top1_accuracy"]),
        "latent_top1_vs_raw": float(latent_metrics["top1_accuracy"])
        - float(raw_metrics["top1_accuracy"]),
        "reconstruction_topk_vs_raw": float(reconstruction_metrics["topk_accuracy"])
        - float(raw_metrics["topk_accuracy"]),
        "latent_topk_vs_raw": float(latent_metrics["topk_accuracy"])
        - float(raw_metrics["topk_accuracy"]),
    }
    status = _comparison_status(
        raw_metrics,
        reconstruction_metrics,
        latent_metrics,
        float(args.min_accuracy_delta),
    )

    result = {
        "schema": RUN_SCHEMA,
        "format": FORMAT,
        "status": status,
        "elapsed_seconds": time.time() - started,
        "run": run_meta,
        "training": history,
        "evaluation": {
            "vae_batch": eval_batch,
            "raw_off": raw_metrics,
            "vae_reconstruction_on": reconstruction_metrics,
            "vae_latent_on": latent_metrics,
            "deltas": deltas,
            "reconstruction": reconstruction_summary,
            "latent": latent_summary,
            "reconstruction_features": reconstruction_feature_summary,
        },
    }

    _write_json(run_dir / "comparison.json", result)
    _write_jsonl(
        run_dir / "eval_windows.jsonl",
        (
            {
                "window": sample.window,
                "target": sample.target,
                "raw_dim": len(sample.raw),
                "reconstruction_dim": len(sample.reconstruction),
                "latent_dim": len(sample.latent),
            }
            for sample in eval_features
        ),
    )

    print(
        "compare raw_top1={raw:.3f} recon_top1={recon:.3f} latent_top1={latent:.3f} "
        "latent_delta={delta:+.3f} recon_mse={mse:.6f} latent_active={active:.3f} status={status}".format(
            raw=float(raw_metrics["top1_accuracy"]),
            recon=float(reconstruction_metrics["top1_accuracy"]),
            latent=float(latent_metrics["top1_accuracy"]),
            delta=float(deltas["latent_top1_vs_raw"]),
            mse=float(reconstruction_summary["mse"]),
            active=float(latent_summary["active_dim_fraction"]),
            status=status,
        ),
        flush=True,
    )
    print(f"comparison_json={run_dir / 'comparison.json'}", flush=True)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.fail_on_regression and status == "regression":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
