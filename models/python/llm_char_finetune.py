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

import _softlogic_cli

FORMAT_V1 = "st-char-lm-v1"
FORMAT_V2 = "st-char-lm-v2"
DEFAULT_UNK = "\uFFFD"

RUN_SCHEMA = "st.modelzoo.run.v1"
TRAINING_CONTRACT_SCHEMA = "st.llm_char_finetune.training_contract.v1"
SUMMARY_SCHEMA = "st.llm_char_finetune.summary.v1"


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


def _backend_contract(backend: str) -> dict[str, Any]:
    normalized = str(backend).strip().lower()
    flags = _native_feature_flags()
    required_any: list[str] = []
    if normalized in {"wgpu", "webgpu", "auto"}:
        required_any = ["wgpu", "wgpu-rt"]
    elif normalized == "cuda":
        required_any = ["cuda"]
    elif normalized in {"hip", "rocm"}:
        required_any = ["hip", "hip-real"]

    if normalized == "cpu":
        available = True
    elif required_any:
        available = any(flags.get(feature) for feature in required_any)
    else:
        available = False

    return {
        "requested": normalized,
        "status": "available" if available else "unavailable",
        "required_any_features": required_any,
        "native_features": flags,
        "availability_checked": True,
    }


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


def _build_training_contract(
    run_meta: dict[str, Any],
    *,
    weights_path: pathlib.Path,
    metrics_path: pathlib.Path,
    samples_dir: pathlib.Path,
    save_weights: pathlib.Path | None,
) -> dict[str, Any]:
    weights_loaded_from = run_meta.get("weights_loaded_from")
    trainable = ["char_rnn", "linear_head", "zspace_softmax"]
    if run_meta.get("embed_dim") is not None:
        trainable.insert(0, "embedding")
    return {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "scope": "llm_char_finetune",
        "learning_mode": "finetune" if weights_loaded_from else "scratch",
        "input": {
            "representation": "tokenizerless_char",
            "format": run_meta.get("format"),
            "unk": DEFAULT_UNK,
            "steps": run_meta.get("steps"),
            "vocab_size": run_meta.get("vocab_size"),
            "symbols_count": run_meta.get("symbols_count"),
            "embed_dim": run_meta.get("embed_dim"),
            "mode": run_meta.get("mode"),
        },
        "parameter_policy": {
            "trainable": trainable,
            "frozen": [],
            "reload_source": weights_loaded_from,
        },
        "geometry": {
            "curvature": run_meta.get("curvature"),
            "temperature": run_meta.get("temperature"),
            "zspace_softmax": True,
            "hypergrad_attached": True,
        },
        "backend": _backend_contract(str(run_meta.get("backend") or "cpu")),
        "optimization": {
            "epochs": run_meta.get("epochs"),
            "batches_per_epoch": run_meta.get("batches_per_epoch"),
            "batch": run_meta.get("batch"),
            "lr": run_meta.get("lr"),
            "roundtable": {
                "top_k": 1,
                "mid_k": 1,
                "bottom_k": 1,
                "here_tolerance": 1e-5,
            },
        },
        "validation": {
            "eval_samples": run_meta.get("eval_samples"),
            "validation_start_fraction_requested": run_meta.get(
                "validation_start_fraction_requested"
            ),
            "validation_start_fraction_actual": run_meta.get(
                "validation_start_fraction_actual"
            ),
        },
        "reload": {
            "weights_loaded_from": weights_loaded_from,
            "output_weights_path": str(weights_path),
            "output_meta_path": str(_meta_path_for_weights(weights_path)),
            "save_weights_path": str(save_weights) if save_weights is not None else None,
            "metadata_required": True,
            "reload_safe": True,
        },
        "artifacts": {
            "metrics_path": str(metrics_path),
            "samples_dir": str(samples_dir),
            "events_path": run_meta.get("events_path"),
        },
        "controls": {
            "desire": bool(run_meta.get("desire")),
            "softlogic": run_meta.get("softlogic"),
        },
    }


def _metrics_summary(metrics_path: pathlib.Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)

    losses = []
    for item in rows:
        try:
            loss = float(item.get("average_loss"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(loss):
            losses.append(loss)

    first_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    best_loss = min(losses) if losses else None
    return {
        "epoch_count": len(rows),
        "loss_count": len(losses),
        "first_average_loss": first_loss,
        "final_average_loss": final_loss,
        "best_average_loss": best_loss,
        "first_minus_final_average_loss": (
            first_loss - final_loss
            if first_loss is not None and final_loss is not None
            else None
        ),
    }


def _safe_prob(value: float, eps: float = 1e-12) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return eps
    return max(eps, min(1.0, value))


def _nll_from_prob(probability: float) -> float:
    return -math.log(_safe_prob(probability))


def _normalise_probs(values: list[float], vocab_size: int) -> list[float]:
    out = []
    for value in values[:vocab_size]:
        parsed = float(value)
        out.append(parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0)
    if len(out) < vocab_size:
        out.extend([0.0] * (vocab_size - len(out)))
    total = sum(out)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / float(vocab_size)] * vocab_size
    inv = 1.0 / total
    return [value * inv for value in out]


def _target_rank(values: list[float], target: int) -> int | None:
    if target < 0 or target >= len(values):
        return None
    target_value = values[target]
    return 1 + sum(1 for value in values if value > target_value)


def _entropy(values: list[float]) -> float:
    return -sum(prob * math.log(_safe_prob(prob)) for prob in values if prob > 0.0)


def _kl_divergence(lhs: list[float], rhs: list[float]) -> float:
    total = 0.0
    for left, right in zip(lhs, rhs):
        if left <= 0.0:
            continue
        total += left * (math.log(_safe_prob(left)) - math.log(_safe_prob(right)))
    return total


def _topk_indices(values: list[float], k: int) -> set[int]:
    if k <= 0:
        return set()
    return {
        idx
        for idx, _value in sorted(
            enumerate(values),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    }


def _split_token_stream(
    tokens: list[int],
    *,
    steps: int,
    val_split: float,
) -> tuple[list[int], list[int], float | None]:
    if val_split <= 0.0 or len(tokens) <= (2 * steps) + 2:
        return tokens, tokens, None
    split_at = int(len(tokens) * (1.0 - val_split))
    split_at = min(max(split_at, steps + 1), len(tokens) - steps - 1)
    train_tokens = tokens[:split_at]
    validation_tokens = tokens[split_at:]
    if len(train_tokens) <= steps or len(validation_tokens) <= steps:
        return tokens, tokens, None
    return train_tokens, validation_tokens, split_at / float(len(tokens))


def _sample_eval_windows(
    tokens: list[int],
    *,
    steps: int,
    count: int,
    seed: int,
) -> list[tuple[list[int], int]]:
    if count <= 0:
        return []
    max_start = len(tokens) - steps
    if max_start <= 0:
        return []
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        start = rng.randrange(0, max_start)
        samples.append((tokens[start : start + steps], tokens[start + steps]))
    return samples


def _tensor_from_eval_windows(
    samples: list[tuple[list[int], int]],
    *,
    vocab_size: int,
    steps: int,
    embed_dim: int | None,
) -> st.Tensor:
    if embed_dim is not None:
        data = [
            float(token)
            for context, _target in samples
            for token in context[:steps]
        ]
        return st.Tensor(len(samples), steps, data)

    cols = vocab_size * steps
    data = [0.0] * (len(samples) * cols)
    for row, (context, _target) in enumerate(samples):
        for step, token in enumerate(context[:steps]):
            if 0 <= token < vocab_size:
                data[row * cols + step * vocab_size + token] = 1.0
    return st.Tensor(len(samples), cols, data)


def _unigram_probs(tokens: list[int], vocab_size: int) -> list[float]:
    counts = [1.0] * vocab_size
    for token in tokens:
        if 0 <= token < vocab_size:
            counts[token] += 1.0
    total = sum(counts)
    return [count / total for count in counts]


def _bigram_probs(tokens: list[int], vocab_size: int) -> list[list[float]]:
    counts = [[1.0] * vocab_size for _ in range(vocab_size)]
    for prev, token in zip(tokens, tokens[1:]):
        if 0 <= prev < vocab_size and 0 <= token < vocab_size:
            counts[prev][token] += 1.0
    return [[value / sum(row) for value in row] for row in counts]


def _baseline_rows(
    samples: list[tuple[list[int], int]],
    *,
    unigram: list[float],
    bigram: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    unigram_rows = [list(unigram) for _ in samples]
    bigram_rows = []
    for context, _target in samples:
        previous = context[-1] if context else 0
        if 0 <= previous < len(bigram):
            bigram_rows.append(list(bigram[previous]))
        else:
            bigram_rows.append(list(unigram))
    return unigram_rows, bigram_rows


def _validation_from_probability_rows(
    rows: list[list[float]],
    samples: list[tuple[list[int], int]],
    *,
    vocab_size: int,
    unigram_rows: list[list[float]] | None = None,
    bigram_rows: list[list[float]] | None = None,
) -> dict[str, Any]:
    if not rows or not samples:
        return {
            "windows": 0,
            "mean_nll": None,
            "perplexity": None,
            "accuracy": None,
            "mean_target_probability": None,
        }

    nll = 0.0
    correct = 0
    target_probability = 0.0
    entropy = 0.0
    ranks: list[int] = []
    unigram_logprob_lifts: list[float] = []
    unigram_rank_lifts: list[float] = []
    unigram_ranks: list[int] = []
    unigram_rank_debts: list[float] = []
    unigram_kls: list[float] = []
    unigram_top5: list[float] = []
    bigram_logprob_lifts: list[float] = []
    bigram_rank_lifts: list[float] = []
    bigram_ranks: list[int] = []
    bigram_rank_debts: list[float] = []
    bigram_kls: list[float] = []
    bigram_top5: list[float] = []
    paired_count = 0

    for idx, ((context, target), raw_row) in enumerate(zip(samples, rows)):
        paired_count += 1
        del context
        values = _normalise_probs([float(value) for value in raw_row], vocab_size)
        prob = values[target] if 0 <= target < vocab_size else 0.0
        target_probability += prob
        nll += _nll_from_prob(prob)
        if _argmax(values) == target:
            correct += 1
        entropy += _entropy(values)
        rank = _target_rank(values, target)
        if rank is not None:
            ranks.append(rank)

        for baseline_rows, logprob_lifts, rank_lifts, baseline_ranks, rank_debts, kls, top5s in (
            (
                unigram_rows,
                unigram_logprob_lifts,
                unigram_rank_lifts,
                unigram_ranks,
                unigram_rank_debts,
                unigram_kls,
                unigram_top5,
            ),
            (
                bigram_rows,
                bigram_logprob_lifts,
                bigram_rank_lifts,
                bigram_ranks,
                bigram_rank_debts,
                bigram_kls,
                bigram_top5,
            ),
        ):
            if baseline_rows is None or idx >= len(baseline_rows):
                continue
            baseline = _normalise_probs(baseline_rows[idx], vocab_size)
            baseline_prob = baseline[target] if 0 <= target < vocab_size else 0.0
            logprob_lifts.append(
                math.log(_safe_prob(prob)) - math.log(_safe_prob(baseline_prob))
            )
            baseline_rank = _target_rank(baseline, target)
            if rank is not None and baseline_rank is not None:
                baseline_ranks.append(baseline_rank)
                rank_lifts.append(float(baseline_rank - rank))
                rank_debts.append(float(rank - baseline_rank))
            kls.append(_kl_divergence(values, baseline))
            denom = float(min(5, vocab_size))
            overlap = len(_topk_indices(values, 5) & _topk_indices(baseline, 5))
            top5s.append(overlap / denom if denom > 0.0 else 0.0)

    if paired_count <= 0:
        return _validation_from_probability_rows([], [], vocab_size=vocab_size)

    count = float(paired_count)
    mean_nll = nll / count

    def mean(values: list[float] | list[int]) -> float | None:
        return sum(float(value) for value in values) / float(len(values)) if values else None

    result: dict[str, Any] = {
        "windows": paired_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 80.0 else None,
        "accuracy": correct / count,
        "mean_target_probability": target_probability / count,
        "mean_entropy": entropy / count,
        "mean_target_rank": mean(ranks),
    }
    if unigram_rows is not None:
        result.update(
            {
                "mean_target_logprob_lift": mean(unigram_logprob_lifts),
                "mean_target_rank_lift": mean(unigram_rank_lifts),
                "mean_unigram_target_rank": mean(unigram_ranks),
                "mean_target_rank_debt_vs_unigram": mean(unigram_rank_debts),
                "mean_kl_to_unigram": mean(unigram_kls),
                "mean_top5_overlap_with_unigram": mean(unigram_top5),
            }
        )
    if bigram_rows is not None:
        result.update(
            {
                "mean_target_logprob_lift_vs_bigram": mean(bigram_logprob_lifts),
                "mean_target_rank_lift_vs_bigram": mean(bigram_rank_lifts),
                "mean_bigram_target_rank": mean(bigram_ranks),
                "mean_target_rank_debt_vs_bigram": mean(bigram_rank_debts),
                "mean_kl_to_bigram": mean(bigram_kls),
                "mean_top5_overlap_with_bigram": mean(bigram_top5),
            }
        )
    return result


def _evaluate_model(
    model: st.nn.Sequential,
    samples: list[tuple[list[int], int]],
    *,
    vocab_size: int,
    steps: int,
    embed_dim: int | None,
    unigram_rows: list[list[float]],
    bigram_rows: list[list[float]],
) -> dict[str, Any]:
    if not samples:
        return _validation_from_probability_rows([], [], vocab_size=vocab_size)
    x = _tensor_from_eval_windows(
        samples,
        vocab_size=vocab_size,
        steps=steps,
        embed_dim=embed_dim,
    )
    rows = model.forward(x).tolist()
    return _validation_from_probability_rows(
        rows,
        samples,
        vocab_size=vocab_size,
        unigram_rows=unigram_rows,
        bigram_rows=bigram_rows,
    )


def _validation_summary_payload(
    *,
    initial_validation: dict[str, Any],
    final_validation: dict[str, Any],
    unigram_validation: dict[str, Any],
    bigram_validation: dict[str, Any],
    best_validation: dict[str, Any],
    best_epoch: int | None,
) -> dict[str, Any]:
    initial_nll = initial_validation.get("mean_nll")
    final_nll = final_validation.get("mean_nll")
    unigram_nll = unigram_validation.get("mean_nll")
    bigram_nll = bigram_validation.get("mean_nll")
    initial_acc = initial_validation.get("accuracy")
    final_acc = final_validation.get("accuracy")
    best_nll = best_validation.get("mean_nll")
    return {
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "unigram_validation": unigram_validation,
        "bigram_validation": bigram_validation,
        "validation_nll_delta": (
            final_nll - initial_nll
            if isinstance(final_nll, (int, float))
            and isinstance(initial_nll, (int, float))
            else None
        ),
        "validation_accuracy_delta": (
            final_acc - initial_acc
            if isinstance(final_acc, (int, float))
            and isinstance(initial_acc, (int, float))
            else None
        ),
        "final_vs_unigram_nll_delta": (
            final_nll - unigram_nll
            if isinstance(final_nll, (int, float))
            and isinstance(unigram_nll, (int, float))
            else None
        ),
        "final_vs_bigram_nll_delta": (
            final_nll - bigram_nll
            if isinstance(final_nll, (int, float))
            and isinstance(bigram_nll, (int, float))
            else None
        ),
        "best_validation": best_validation,
        "best_validation_epoch": best_epoch,
        "best_validation_mean_nll": best_nll,
        "final_minus_best_validation_nll": (
            final_nll - best_nll
            if isinstance(final_nll, (int, float))
            and isinstance(best_nll, (int, float))
            else None
        ),
        "best_vs_unigram_nll_delta": (
            best_nll - unigram_nll
            if isinstance(best_nll, (int, float))
            and isinstance(unigram_nll, (int, float))
            else None
        ),
        "best_vs_bigram_nll_delta": (
            best_nll - bigram_nll
            if isinstance(best_nll, (int, float))
            and isinstance(bigram_nll, (int, float))
            else None
        ),
    }


def _write_completion_summary(
    run_dir: pathlib.Path,
    run_meta: dict[str, Any],
    *,
    weights_path: pathlib.Path,
    metrics_path: pathlib.Path,
    final_sample_path: pathlib.Path,
    save_weights: pathlib.Path | None,
    validation: dict[str, Any] | None = None,
) -> None:
    meta_path = _meta_path_for_weights(weights_path)
    external_meta_path = (
        _meta_path_for_weights(save_weights) if save_weights is not None else None
    )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "arch": run_meta.get("arch"),
        "run_schema": run_meta.get("schema"),
        "training_contract": run_meta.get("training_contract"),
        "checkpoint": {
            "weights_path": str(weights_path),
            "weights_exists": weights_path.exists(),
            "meta_path": str(meta_path),
            "meta_exists": meta_path.exists(),
            "save_weights_path": str(save_weights) if save_weights is not None else None,
            "save_weights_exists": (
                save_weights.exists() if save_weights is not None else None
            ),
            "save_meta_path": (
                str(external_meta_path) if external_meta_path is not None else None
            ),
            "save_meta_exists": (
                external_meta_path.exists()
                if external_meta_path is not None
                else None
            ),
            "loaded_from": run_meta.get("weights_loaded_from"),
        },
        "metrics": _metrics_summary(metrics_path),
        "sample": {
            "path": str(final_sample_path),
            "exists": final_sample_path.exists(),
        },
    }
    if validation:
        payload.update(validation)
    _write_json(run_dir / "summary.json", payload)


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
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text_or_dir> [<text_or_dir> ...] "
            "[--load weights.json] [--save weights.json] [--steps N] [--embed-dim N] [--hidden N] "
            "[--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] [--temperature F] "
            "[--gen N] [--topk N] [--seed N] [--prompt STR] [--val-split F] [--eval-samples N] "
            "[--backend cpu|wgpu|cuda|hip|auto] "
            "[--events PATH] [--events-types A,B,C] "
            "[--atlas] [--atlas-bound N] [--atlas-district NAME] "
            "[--desire] [--desire-concepts N] [--desire-prime N] [--desire-blend F] [--desire-drift-gain F] "
            "[--run-dir PATH] "
            f"{_softlogic_cli.usage_flags()}"
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
    val_split = 0.1
    eval_samples = 64
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

    softlogic_cli = _softlogic_cli.pop_softlogic_flags(args)
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
        elif flag == "--val-split":
            val_split = float(next(it))
        elif flag == "--eval-samples":
            eval_samples = int(next(it))
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
    if val_split < 0.0 or val_split >= 0.95:
        raise ValueError("--val-split must be within [0, 0.95)")
    if eval_samples < 0:
        raise ValueError("--eval-samples must be >= 0")

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
    metrics_path = run_dir / "metrics.jsonl"
    weights_path = run_dir / "weights.json"
    if metrics_path.exists():
        metrics_path.unlink()

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
    train_tokens, validation_tokens, validation_start_fraction_actual = _split_token_stream(
        tokens,
        steps=steps,
        val_split=val_split,
    )

    if prompt is None:
        prompt = "".join(list(text)[:steps])

    mode = (
        f"embedding({embed_dim_runtime})" if embed_dim_runtime is not None else "one_hot"
    )
    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "llm_char_finetune",
        "data_paths": [str(path) for path in data_paths],
        "data_file_count": len(data_files),
        "data_files_manifest": str(run_dir / "data_files.txt"),
        "format": FORMAT_V2 if embed_dim_runtime is not None else FORMAT_V1,
        "steps": steps,
        "embed_dim": embed_dim_runtime,
        "hidden": hidden,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "eval_samples": eval_samples,
        "lr": lr,
        "learning_rate": lr,
        "curvature": curvature,
        "temperature": temperature,
        "backend": backend,
        "gen_len": gen_len,
        "top_k": top_k,
        "seed": seed,
        "prompt": prompt,
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "validation_start_fraction_requested": 1.0 - val_split,
        "validation_start_fraction_actual": validation_start_fraction_actual,
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

    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    trainer = st.nn.ModuleTrainer(
        backend=backend,
        curvature=curvature,
        hyper_learning_rate=lr,
        fallback_learning_rate=lr,
    )
    softlogic_meta = _softlogic_cli.apply_softlogic_cli(trainer, softlogic_cli)
    run_meta["softlogic"] = softlogic_meta
    run_meta["training_contract"] = _build_training_contract(
        run_meta,
        weights_path=weights_path,
        metrics_path=metrics_path,
        samples_dir=samples_dir,
        save_weights=save_weights,
    )
    _write_json(run_dir / "run.json", run_meta)
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
    validation_samples = _sample_eval_windows(
        validation_tokens,
        steps=steps,
        count=eval_samples,
        seed=seed + 700_000,
    )
    unigram = _unigram_probs(train_tokens, vocab_size)
    bigram = _bigram_probs(train_tokens, vocab_size)
    unigram_rows, bigram_rows = _baseline_rows(
        validation_samples,
        unigram=unigram,
        bigram=bigram,
    )
    unigram_validation = _validation_from_probability_rows(
        unigram_rows,
        validation_samples,
        vocab_size=vocab_size,
    )
    bigram_validation = _validation_from_probability_rows(
        bigram_rows,
        validation_samples,
        vocab_size=vocab_size,
    )
    initial_validation = _evaluate_model(
        model,
        validation_samples,
        vocab_size=vocab_size,
        steps=steps,
        embed_dim=embed_dim_runtime,
        unigram_rows=unigram_rows,
        bigram_rows=bigram_rows,
    )
    best_validation = initial_validation
    best_validation_epoch: int | None = None

    print(
        f"mode={mode} vocab={vocab_size} files={len(data_files)} chars={len(text)} steps={steps} hidden={hidden} epochs={epochs} "
        f"batch={batch} lr={lr:.3e} curvature={curvature} temp={temperature} backend={backend} eval_samples={eval_samples} run_dir={run_dir}"
    )
    if initial_validation.get("mean_nll") is not None:
        print(
            f"init val_nll={float(initial_validation['mean_nll']):.6f} "
            f"acc={float(initial_validation['accuracy']) * 100.0:.2f}%",
            flush=True,
        )

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
                        train_tokens,
                        vocab_size,
                        steps,
                        batch,
                        rng,
                        embed_dim=embed_dim_runtime,
                    )
                )
            stats = trainer.train_epoch(model, loss, batches, schedule)
            avg_loss = float(stats.average_loss)
            validation = _evaluate_model(
                model,
                validation_samples,
                vocab_size=vocab_size,
                steps=steps,
                embed_dim=embed_dim_runtime,
                unigram_rows=unigram_rows,
                bigram_rows=bigram_rows,
            )
            validation_nll = validation.get("mean_nll")
            best_nll = best_validation.get("mean_nll")
            if (
                isinstance(validation_nll, (int, float))
                and (
                    not isinstance(best_nll, (int, float))
                    or float(validation_nll) < float(best_nll)
                )
            ):
                best_validation = validation
                best_validation_epoch = epoch
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batches": int(stats.batches),
                            "average_loss": avg_loss,
                            "validation": validation,
                        },
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
            if validation.get("mean_nll") is not None:
                print(
                    f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f} "
                    f"val_nll={float(validation['mean_nll']):.6f} "
                    f"acc={float(validation['accuracy']) * 100.0:.2f}%",
                    flush=True,
                )
            else:
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
        final_sample_path = samples_dir / f"epoch_{epochs - 1:03d}.txt"
        sample = final_sample_path.read_text(encoding="utf-8")
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
        final_sample_path = samples_dir / "init.txt"
        final_sample_path.write_text(sample, encoding="utf-8")
    final_validation = (
        validation
        if epochs > 0
        else _evaluate_model(
            model,
            validation_samples,
            vocab_size=vocab_size,
            steps=steps,
            embed_dim=embed_dim_runtime,
            unigram_rows=unigram_rows,
            bigram_rows=bigram_rows,
        )
    )
    _write_completion_summary(
        run_dir,
        run_meta,
        weights_path=weights_path,
        metrics_path=metrics_path,
        final_sample_path=final_sample_path,
        save_weights=save_weights,
        validation=_validation_summary_payload(
            initial_validation=initial_validation,
            final_validation=final_validation,
            unigram_validation=unigram_validation,
            bigram_validation=bigram_validation,
            best_validation=best_validation,
            best_epoch=best_validation_epoch,
        ),
    )
    print("--- sample (prompt + gen) ---")
    print(sample)


if __name__ == "__main__":
    main()
