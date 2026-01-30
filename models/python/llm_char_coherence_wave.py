from __future__ import annotations

import datetime as _dt
import json
import pathlib
import random
import sys
from typing import Any

# Prefer the in-repo development shim when running from a source checkout.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if (_ROOT / "spiraltorch").is_dir():
    sys.path.insert(0, str(_ROOT))

import spiraltorch as st

FORMAT = "st-char-lm-coherence-wave-v1"
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


def _parse_dilations(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"invalid --dilations entry: {part}")
        out.append(value)
    if not out:
        raise ValueError("empty --dilations")
    return out


def _build_model(
    vocab_size: int,
    steps: int,
    embed_dim: int,
    hidden: int,
    memory: int,
    kernel: int,
    dilations: list[int],
    curvature: float,
    temperature: float,
) -> st.nn.Sequential:
    model = st.nn.Sequential()
    model.add(st.nn.Embedding("embed", vocab_size, embed_dim))
    model.add(
        st.nn.ZSpaceCoherenceWaveBlock(
            embed_dim,
            steps,
            memory,
            curvature,
            temperature,
            kernel_size=kernel,
            dilations=dilations,
        )
    )
    model.add(st.nn.Linear("mix", embed_dim, hidden))
    model.add(st.nn.Relu())
    model.add(st.nn.Linear("head", hidden, vocab_size))
    model.add(st.nn.ZSpaceSoftmax(curvature, temperature))
    return model


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
        next_idx = _sample_topk([float(v) for v in row], top_k, rng)
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
            "usage: PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_wave.py <text.txt> "
            "[--load weights.json] [--save weights.json] [--steps N] [--embed-dim N] [--hidden N] [--memory N] "
            "[--kernel N] [--dilations 1,2,4] [--epochs N] [--batches N] [--batch N] [--lr F] [--curvature F] "
            "[--temperature F] [--gen N] [--topk N] [--seed N] [--prompt STR] "
            "[--infuse STR] [--infuse-every once|epoch|batch] [--infuse-mode blend|separate] "
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
    memory = 16
    kernel = 3
    dilations = [1, 2, 4]
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
        elif flag == "--memory":
            memory = int(next(it))
        elif flag == "--kernel":
            kernel = int(next(it))
        elif flag == "--dilations":
            dilations = _parse_dilations(str(next(it)))
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
        elif flag == "--run-dir":
            run_dir = pathlib.Path(str(next(it)))
        else:
            raise ValueError(f"unknown flag: {flag}")

    if memory > steps:
        raise ValueError(f"--memory must be <= --steps (got memory={memory}, steps={steps})")
    if kernel <= 0:
        raise ValueError("--kernel must be > 0")
    if not dilations:
        raise ValueError("empty --dilations")
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

    if run_dir is None:
        run_dir = _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    text = _read_text(text_path)
    if not text:
        raise ValueError("empty text")

    if load_weights is not None:
        meta_path = _meta_path_for_weights(load_weights)
        meta = _load_meta(meta_path)
        fmt = str(meta.get("format", ""))
        if fmt != FORMAT:
            raise ValueError(f"unexpected meta format: {fmt}")
        steps = int(meta["steps"])
        embed_dim = int(meta["embed_dim"])
        hidden = int(meta["hidden"])
        memory = int(meta["memory"])
        kernel = int(meta["kernel"])
        dilations = [int(v) for v in meta["dilations"]]
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
            memory,
            kernel,
            dilations,
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
            memory,
            kernel,
            dilations,
            curvature,
            temperature,
        )

    tokens = _encode(text, index)
    if len(tokens) <= steps:
        raise ValueError(f"text too short for steps={steps}: len={len(tokens)}")

    if prompt is None:
        prompt = "".join(list(text)[:steps])

    run_meta: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "arch": "llm_char_coherence_wave",
        "text_path": str(text_path),
        "format": FORMAT,
        "steps": steps,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "memory": memory,
        "kernel": kernel,
        "dilations": dilations,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch": batch,
        "lr": lr,
        "curvature": curvature,
        "temperature": temperature,
        "gen_len": gen_len,
        "top_k": top_k,
        "seed": seed,
        "prompt": prompt,
        "infuse": infuse,
        "infuse_every": infuse_every,
        "infuse_mode": infuse_mode,
        "vocab_size": vocab_size,
        "symbols_count": len(symbols),
        "weights_loaded_from": str(load_weights) if load_weights is not None else None,
    }
    _write_json(run_dir / "run.json", run_meta)

    model.attach_hypergrad(curvature=curvature, learning_rate=lr)
    trainer = st.nn.ModuleTrainer(
        backend="cpu",
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
    schedule = trainer.roundtable(
        batch,
        vocab_size,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    loss = st.nn.CategoricalCrossEntropy()

    print(
        f"arch=coherence_wave vocab={vocab_size} steps={steps} embed_dim={embed_dim} hidden={hidden} "
        f"memory={memory} kernel={kernel} dilations={dilations} epochs={epochs} batch={batch} "
        f"lr={lr:.3e} curvature={curvature} temp={temperature} run_dir={run_dir}"
    )

    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(epochs):
        rng = random.Random(seed + epoch * 10_000)
        batches = []
        for _ in range(batches_per_epoch):
            batches.append(_build_random_batch(tokens, vocab_size, steps, batch, rng))
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
        )
        (samples_dir / f"epoch_{epoch:03d}.txt").write_text(sample, encoding="utf-8")
        print(f"epoch[{epoch}] batches={stats.batches} avg_loss={avg_loss:.6f}")

    weights_path = run_dir / "weights.json"
    st.nn.save(str(weights_path), model)

    meta = {
        "format": FORMAT,
        "steps": steps,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "memory": memory,
        "kernel": kernel,
        "dilations": dilations,
        "curvature": curvature,
        "temperature": temperature,
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
        final_sample = (samples_dir / f"epoch_{epochs - 1:03d}.txt").read_text(
            encoding="utf-8"
        )
    else:
        final_sample = _generate(
            model,
            symbols,
            index,
            steps,
            prompt,
            gen_len,
            top_k,
            seed + 999,
        )
        (samples_dir / "init.txt").write_text(final_sample, encoding="utf-8")
    print("--- sample (prompt + gen) ---")
    print(final_sample)


if __name__ == "__main__":
    main()
