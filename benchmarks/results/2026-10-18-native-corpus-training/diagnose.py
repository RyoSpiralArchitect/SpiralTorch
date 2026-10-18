"""First-update gradient localization after the sealed corpus comparison failed."""
import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["native", "reference"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-f64-ce", action="store_true")
    args = parser.parse_args()
    if args.mode == "native" and args.reference_f64_ce:
        parser.error("the native math is not changed by this reference diagnostic")
    helper = runpy.run_path(str(Path(__file__).resolve().with_name("qualify.py")))
    helper["cohort"](args)
    data = json.loads((args.cohort / "data.json").read_text())["token_documents"][0]
    recipe = helper["RECIPE"]
    width = recipe["context_length"]
    rows = [data[start:start+width] for start in range(0, width * 4, width)]
    if args.mode == "native":
        helper["forbid_ml"]()
        import spiraltorch as st
        base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(),
                                          st.SafeTensorArchive(args.model_dir / "model.safetensors"))
        artifact = st.GeometryAdapterArtifact.from_json((args.cohort / "initial.json").read_text(),
                                                       helper["ARTIFACT"], helper["BASE"])
        model = base.with_geometry(artifact)
        group = model.geometry_accumulator(label_smoothing=recipe["label_smoothing"])
        for offset in (0, 2):
            group.accumulate_loss(model, [v for row in rows[offset:offset+2] for v in row], [2, width])
        gradients = group.parameter_gradients()
        optimizer = st.NativeGeometryAdamW(artifact, max_gradient_norm=recipe["max_gradient_norm"])
        updated, optimizer = group.step_adamw(model, optimizer, recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        final = updated.geometry_artifact().to_json()
        loss = group.loss
    else:
        import torch
        import transformers
        import spiraltorch as st
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
        artifact = st.GeometryAdapterArtifact.from_json((args.cohort / "initial.json").read_text(),
                                                       helper["ARTIFACT"], helper["BASE"])
        stack = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=helper["BASE"])
        optimizer = torch.optim.AdamW(stack.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"],
                                     betas=(.9, .999), eps=1e-8, foreach=False)
        attach = runpy.run_path(str(helper["HELPER"]))["attach_reference"]
        loss = 0.
        for offset in (0, 2):
            ids = torch.tensor(rows[offset:offset+2])
            with attach(stack, model):
                logits = model(ids[:, :-1], use_cache=False).logits
                if args.reference_f64_ce:
                    logits = logits.double()
                value = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                    reduction="sum", label_smoothing=recipe["label_smoothing"]) / (4 * (width-1))
                value.backward()
            loss += value.item()
        gradients = {path: {name: p.grad.detach().flatten().tolist() for name, p in adapter.named_parameters()}
                     for path, adapter in zip(stack.paths, stack.adapters)}
        torch.nn.utils.clip_grad_norm_(stack.parameters(), recipe["max_gradient_norm"], error_if_nonfinite=True)
        optimizer.step()
        final = stack.export_artifact(base_model_identity=helper["BASE"]).to_json()
    helper["save"](args.output, {"mode": args.mode, "reference_f64_ce": args.reference_f64_ce,
                               "loss": loss, "gradients": gradients,
                               "geometry_json": final, "contract_sha256": helper["digest"](args.cohort / "contract.json")})


if __name__ == "__main__":
    main()
