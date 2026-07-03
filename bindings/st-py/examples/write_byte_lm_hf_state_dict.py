import argparse
import math
from pathlib import Path

import spiraltorch as st

from checkpoint_preflight import HF_KEY_PRESETS, HF_UNUSED_KEYS


VOCAB = st.dataset.BYTE_LM_VOCAB
HIDDEN = 24


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a tiny byte-compatible HF/PyTorch-style LM state dict."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .bin/.pt file path, usually .../pytorch_model.bin.",
    )
    parser.add_argument(
        "--key-preset",
        choices=sorted(HF_KEY_PRESETS),
        default="llama",
        help="HF-style key preset to write.",
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=VOCAB,
        help="Embedding vocabulary rows.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=HIDDEN,
        help="Embedding/head hidden width.",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        default=VOCAB,
        help="LM-head output classes.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.003,
        help="Small deterministic weight scale.",
    )
    parser.add_argument(
        "--trained-source",
        action="store_true",
        help="Train the tiny byte MLP source first, then externalize its checkpoint.",
    )
    parser.add_argument(
        "--source-text",
        default="spiraltorch adapters inherit byte memories; adapters inherit byte memories",
        help="Source text used with --trained-source.",
    )
    parser.add_argument(
        "--include-biases",
        action="store_true",
        help=(
            "Write explicit embed/head biases instead of exercising bias synthesis. "
            "For --trained-source this preserves the trained bias tensors."
        ),
    )
    parser.add_argument(
        "--no-extra-key",
        action="store_true",
        help="Do not include the preset's unused layernorm-style audit key.",
    )
    parser.add_argument(
        "--nested-model",
        action="store_true",
        help="Save under {'model': state_dict} instead of a flat HF state dict.",
    )
    args = parser.parse_args()
    for name in ["vocab", "hidden", "target_classes"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.scale <= 0.0:
        parser.error("--scale must be positive")
    return args


def deterministic_matrix(rows, cols, *, scale, phase):
    values = []
    for row in range(rows):
        row_values = []
        for col in range(cols):
            angle = (row + 1) * 0.017 + (col + 1) * 0.071 + phase
            row_values.append(scale * (math.sin(angle) + 0.5 * math.cos(angle * 0.37)))
        values.append(row_values)
    return values


def deterministic_vector(cols, *, scale, phase):
    return [
        scale * 0.25 * math.sin((col + 1) * 0.113 + phase)
        for col in range(cols)
    ]


def build_byte_hf_state_dict(
    *,
    key_preset,
    vocab=VOCAB,
    hidden=HIDDEN,
    target_classes=VOCAB,
    scale=0.003,
    include_biases=False,
    include_extra_key=True,
):
    keys = HF_KEY_PRESETS[key_preset]
    state = {
        keys["embed_weight_key"]: deterministic_matrix(
            vocab,
            hidden,
            scale=scale,
            phase=0.0,
        ),
        keys["lm_head_weight_key"]: deterministic_matrix(
            target_classes,
            hidden,
            scale=scale,
            phase=1.7,
        ),
    }
    if include_biases:
        state[keys["embed_bias_key"]] = deterministic_vector(
            hidden,
            scale=scale,
            phase=0.5,
        )
        state[keys["lm_head_bias_key"]] = deterministic_vector(
            target_classes,
            scale=scale,
            phase=1.1,
        )
    if include_extra_key:
        state[HF_UNUSED_KEYS.get(key_preset, f"{key_preset}.unused.norm.weight")] = [
            [1.0, 1.0]
        ]
    return state


def tensor_to_rows(tensor):
    data = tensor.data()
    return [
        data[row * tensor.cols : (row + 1) * tensor.cols]
        for row in range(tensor.rows)
    ]


def build_trained_byte_hf_state_dict(
    *,
    key_preset,
    source_text,
    include_biases,
    include_extra_key,
):
    from spiraltorch.nn import SoftmaxCrossEntropy

    from byte_lm_mlp_lora_adapter import (
        ACCUMULATION_STEPS,
        BATCH_WINDOWS,
        CONTEXT,
        VOCAB as ADAPTER_VOCAB,
        externalize_mlp_state,
        loader,
        train_source,
    )

    if ADAPTER_VOCAB != VOCAB:
        raise RuntimeError(f"writer VOCAB={VOCAB} does not match adapter VOCAB={ADAPTER_VOCAB}")

    source_samples = st.dataset.byte_lm_windows(source_text, CONTEXT)
    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.5,
        fallback_learning_rate=0.1,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(2.0)
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    source, source_stats, source_delta = train_source(
        session,
        trainer,
        schedule,
        SoftmaxCrossEntropy(),
        source_samples,
    )
    checkpoint, _rules = externalize_mlp_state(source.state_dict(), key_preset=key_preset)
    keys = HF_KEY_PRESETS[key_preset]
    if not include_biases:
        checkpoint.pop(keys["embed_bias_key"], None)
        checkpoint.pop(keys["lm_head_bias_key"], None)
    if include_extra_key:
        checkpoint[HF_UNUSED_KEYS.get(key_preset, f"{key_preset}.unused.norm.weight")] = (
            st.Tensor(1, 2, [1.0, 1.0])
        )
    state = {key: tensor_to_rows(tensor) for key, tensor in checkpoint.items()}
    return state, {
        "source_windows": len(source_samples),
        "source_batches": source_stats.batches,
        "source_optimizer_steps": source_stats.optimizer_steps,
        "source_loss_delta": source_delta["loss_delta"],
    }


def save_torch_state_dict(path, state, *, nested_model=False):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("writing a PyTorch state dict requires torch") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_state = {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in state.items()
    }
    payload = {"model": tensor_state} if nested_model else tensor_state
    torch.save(payload, path)


def shape_label(value):
    rows = len(value)
    cols = len(value[0]) if rows and isinstance(value[0], list) else 1
    return f"{rows}x{cols}"


def main():
    args = parse_args()
    train_summary = None
    if args.trained_source:
        if args.vocab != VOCAB or args.hidden != HIDDEN or args.target_classes != VOCAB:
            raise RuntimeError(
                "--trained-source currently writes the adapter smoke shape "
                f"(vocab, hidden, target_classes)=({VOCAB}, {HIDDEN}, {VOCAB})"
            )
        state, train_summary = build_trained_byte_hf_state_dict(
            key_preset=args.key_preset,
            source_text=args.source_text,
            include_biases=args.include_biases,
            include_extra_key=not args.no_extra_key,
        )
    else:
        state = build_byte_hf_state_dict(
            key_preset=args.key_preset,
            vocab=args.vocab,
            hidden=args.hidden,
            target_classes=args.target_classes,
            scale=args.scale,
            include_biases=args.include_biases,
            include_extra_key=not args.no_extra_key,
        )
    save_torch_state_dict(args.out, state, nested_model=args.nested_model)
    print(
        "byte_hf_state_dict "
        f"path={args.out} "
        f"key_preset={args.key_preset} "
        f"keys={len(state)} "
        f"vocab={args.vocab} "
        f"hidden={args.hidden} "
        f"target_classes={args.target_classes} "
        f"include_biases={args.include_biases} "
        f"trained_source={args.trained_source} "
        f"nested_model={args.nested_model}"
    )
    if train_summary is not None:
        print(
            "trained_source "
            f"source_windows={train_summary['source_windows']} "
            f"source_batches={train_summary['source_batches']} "
            f"source_optimizer_steps={train_summary['source_optimizer_steps']} "
            f"source_loss_delta={train_summary['source_loss_delta']:.6f}"
        )
    for key in sorted(state):
        print(f"state_key name={key} shape={shape_label(state[key])}")


if __name__ == "__main__":
    main()
