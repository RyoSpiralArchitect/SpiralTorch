"""Post-result logit/probability diagnostic; does not replace the sealed gate."""
import argparse
from pathlib import Path
import runpy

Q = runpy.run_path(str(Path(__file__).with_name("qualify.py")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model-dir", "data", "checkpoint", "cohort", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.mode = "diagnostic"
    data = Q["checked"](args)
    import torch
    import transformers
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = Q["PRIOR"]["model"](args, st)["base"]
    reference = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    results = []
    with torch.inference_mode():
        for length in Q["TIMING_LENGTHS"]:
            ids = data[:length + Q["APPENDS"]]
            actual = torch.tensor(model.forward(ids, [1, len(ids)], last_only=True).values(), dtype=torch.float64)
            hidden = reference.transformer(torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1:]
            expected = reference.lm_head(hidden).flatten().double()
            delta = actual - expected
            results.append({"prompt_length": length, "evaluated_tokens": len(ids),
                "raw_logit_max_abs": float(delta.abs().max()), "mean_logit_shift": float(delta.mean()),
                "centered_logit_max_abs": float((delta-delta.mean()).abs().max()),
                "probability_max_abs": float((actual.softmax(0)-expected.softmax(0)).abs().max()),
                "top1_exact": int(actual.argmax()) == int(expected.argmax()),
                "original_logit_gate_passed": float(delta.abs().max()) <= Q["GATES"]["logit_atol"]})
    Q["save"](args.output, {"kind": "post_result_diagnostic_not_a_replacement_gate", "cases": results,
        "script_sha256": Q["digest"](Path(__file__)), "extension_sha256": Q["digest"](Path(st._rs.__file__))})


if __name__ == "__main__":
    main()
