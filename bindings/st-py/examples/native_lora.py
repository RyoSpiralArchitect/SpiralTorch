"""One native GPT-2 LoRA update, adapter reload, and bounded generation.

Uses existing local GPT-2 safetensors and exact token IDs. No downloads, tokenizer,
Torch, or Transformers. This is a mechanics example, not a language-quality study.
"""
import argparse
import json
from pathlib import Path

import spiraltorch as st


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("tokens", type=Path, help="JSON list containing a training window of integer IDs")
    parser.add_argument("output", type=Path, help="adapter JSON destination")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--resume-adapter", type=Path)
    parser.add_argument("--readout", choices=("f32", "f64"), default="f32",
                        help="retain the same policy when resuming training")
    args = parser.parse_args()

    archive = st.SafeTensorArchive.from_bytes(args.weights.read_bytes())
    base = st.NativeCausalLm.from_gpt2(args.config.read_text(), archive).with_readout_accumulator(args.readout)
    model = (base.with_lora_json(args.resume_adapter.read_text()) if args.resume_adapter
             else base.with_lora(args.layer, args.rank, args.alpha))
    window = json.loads(args.tokens.read_text())
    gradients = model.lora_loss_gradients(window, [1, len(window)])
    updated = model.step_lora_sgd(gradients, args.learning_rate)
    args.output.write_text(updated.lora_to_json())
    restored = base.with_lora_json(args.output.read_text())
    assert restored.lora_artifact_id == updated.lora_artifact_id
    generator = st.NativeCausalLmGenerator(restored, window[:1], '{"max_new_tokens": 1}')
    generator.step()
    print(json.dumps({
        "base_model_identity": restored.base_model_identity,
        "lora_artifact_id": restored.lora_artifact_id,
        "pre_update_loss": gradients.loss,
        "target_tokens": gradients.target_tokens,
        "generated_ids": generator.generated_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
