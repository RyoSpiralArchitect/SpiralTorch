"""Post-failure same-input terminal VJP localization; Torch is reference only."""
import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper = runpy.run_path(str(Path(__file__).resolve().with_name("qualify.py")))
    helper["cohort"](args)
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
    attach = runpy.run_path(str(helper["HELPER"]))["attach_reference"]
    data = json.loads((args.cohort / "data.json").read_text())["token_documents"][0]
    ids = torch.tensor([data[0:32], data[32:64]])
    captures = {}

    def hook(_module, inputs, output):
        captures["input"], captures["normalized"] = inputs[0], output
        inputs[0].retain_grad()
        output.retain_grad()

    handle = model.transformer.ln_f.register_forward_hook(hook)
    with attach(stack, model):
        logits = model(ids[:, :-1], use_cache=False).logits
        logits.retain_grad()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
            label_smoothing=.1) * .5
        loss.backward()
    handle.remove()

    def native(value):
        # Safetensors-backed views need not meet the native DLPack alignment.
        return st.Tensor.from_dlpack(value.detach().contiguous().clone().__dlpack__())

    def tensor(value):
        return torch.utils.dlpack.from_dlpack(value)

    def error(value, expected):
        a, b = value.detach().double().flatten(), expected.detach().double().flatten()
        return {"max_absolute": float((a-b).abs().max()), "l2_error_over_reference": float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b))}

    flat = logits.reshape(-1, logits.shape[-1])
    seed = native(torch.tensor([[.5]]))
    d_logits = native(flat).cross_entropy_with_logits_backward(ids[:, 1:].reshape(-1).tolist(), seed, label_smoothing=.1)
    double_logits = flat.detach().double().requires_grad_()
    double_loss = torch.nn.functional.cross_entropy(double_logits, ids[:, 1:].reshape(-1), label_smoothing=.1) * .5
    double_gradient, = torch.autograd.grad(double_loss, (double_logits,))
    d_norm = d_logits.matmul(native(model.lm_head.weight), backend="cpu_simd")
    gamma = native(model.transformer.ln_f.weight.reshape(1, -1))
    before = captures["input"].reshape(-1, model.config.n_embd)
    d_input = native(before).layer_norm_affine_backward(gamma, d_norm, epsilon=model.config.layer_norm_epsilon)[0]
    exact_head = tensor(d_logits).double() @ model.lm_head.weight.detach().double()
    exact_d_norm = native(exact_head.float())
    exact_d_input = native(before).layer_norm_affine_backward(gamma, exact_d_norm, epsilon=model.config.layer_norm_epsilon)[0]
    report = {"cross_entropy_gradient": error(tensor(d_logits), logits.grad),
        "native_cross_entropy_vs_f64": error(tensor(d_logits), double_gradient),
        "torch_f32_cross_entropy_vs_f64": error(logits.grad, double_gradient),
        "head_matmul_gradient": error(tensor(d_norm), captures["normalized"].grad),
        "head_matmul_vs_f64": error(tensor(d_norm), exact_head),
        "f64_head_vs_torch": error(exact_head, captures["normalized"].grad),
        "final_norm_gradient": error(tensor(d_input), captures["input"].grad),
        "final_norm_after_f64_head": error(tensor(exact_d_input), captures["input"].grad),
        "contract_sha256": helper["digest"](args.cohort / "contract.json")}
    helper["save"](args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
