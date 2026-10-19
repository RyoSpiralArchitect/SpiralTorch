"""Post-result local-kernel attribution; hybrid hooks are diagnostics, not a runtime."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
OLD = runpy.run_path(str(ROOT / "benchmarks/results/2026-10-18-native-lm-cache/qualify.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OLD["PRIOR"]["PRIOR"]["check_model"](args.model_dir)
    if OLD["digest"](args.data) != OLD["PRIOR"]["DATA"]:
        raise ValueError("only previously admitted encoded train data is allowed")
    data = OLD["load"](args.data)["token_documents"][0]
    import torch
    import transformers
    from transformers.pytorch_utils import Conv1D
    import spiraltorch as st
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = transformers.GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(),
                                     st.SafeTensorArchive(args.model_dir / "model.safetensors"))
    if base.base_model_identity != OLD["BASE"]:
        raise ValueError("base model identity mismatch")
    packs = {}

    def native(value):
        # Safetensors views can be contiguous without satisfying the native alignment contract.
        owned = value.detach().contiguous().clone()
        return st.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(owned))

    def foreign(value):
        return torch.from_dlpack(value)

    def compare(actual, expected):
        delta = actual.double() - expected.double()
        return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt())}

    def projected(name, module, inputs):
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).contiguous()
        weight = module.weight if isinstance(module, Conv1D) else module.weight.T.contiguous()
        if name not in packs:
            packs[name] = st.cpu_simd_prepack_rhs(native(weight))
        value = foreign(native(x).matmul_simd_prepacked(packs[name]))
        if module.bias is not None:
            value = value + module.bias
        return value.reshape(*inputs[0].shape[:-1], weight.shape[1])

    def normalized(module, inputs):
        x = inputs[0]
        z, _ = native(x.reshape(-1, x.shape[-1])).layer_norm_stats(epsilon=module.eps)
        return foreign(z).reshape_as(x) * module.weight + module.bias

    @contextmanager
    def replace(linears=False, norms=False, measurements=None):
        handles = []
        def hook(name):
            def execute(module, inputs, output):
                is_linear = isinstance(module, (Conv1D, torch.nn.Linear))
                actual = projected(name, module, inputs) if is_linear else normalized(module, inputs)
                if measurements is not None:
                    row = {"name": name, "kind": "linear" if is_linear else "norm",
                           "shape": list(output.shape), "native_vs_torch": compare(actual, output)}
                    if is_linear:
                        x = inputs[0].double()
                        weight = module.weight if isinstance(module, Conv1D) else module.weight.T
                        high = x @ weight.double()
                        if module.bias is not None:
                            high = high + module.bias.double()
                    else:
                        x = inputs[0].double()
                        high = torch.nn.functional.layer_norm(x, module.normalized_shape,
                            module.weight.double(), module.bias.double(), module.eps)
                    row["native_vs_f64"] = compare(actual, high)
                    row["torch_vs_f64"] = compare(output, high)
                    measurements.append(row)
                if (is_linear and linears) or (not is_linear and norms):
                    return actual
                return None
            return execute
        try:
            for name, module in reference.named_modules():
                if isinstance(module, (Conv1D, torch.nn.Linear, torch.nn.LayerNorm)):
                    handles.append(module.register_forward_hook(hook(name)))
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def forward(ids):
        hidden = reference.transformer(ids, use_cache=False).last_hidden_state[:, -1:]
        return reference.lm_head(hidden).flatten().clone()

    conditions = []
    cached_logits = {}
    with torch.inference_mode():
        for steps in [24, 72]:
            ids = torch.tensor([data[:steps]])
            measurements = []
            with replace(measurements=measurements):
                expected = forward(ids)
            actual = torch.tensor(base.forward(data[:steps], [1, steps], last_only=True).values())
            cached_logits[steps] = {"native": actual, "torch_f32": expected}
            row = {"steps": steps, "native_vs_torch": compare(actual, expected), "primitives": measurements, "hybrids": []}
            for linears, norms in [(True, False), (False, True), (True, True)]:
                with replace(linears=linears, norms=norms):
                    output = forward(ids)
                name = f"native_linears={linears},native_norms={norms}"
                cached_logits[steps][name] = output
                row["hybrids"].append({"name": name, "vs_torch": compare(output, expected), "vs_native": compare(output, actual)})
            conditions.append(row)
            print(f"steps={steps} primitive/hybrid diagnostics complete", flush=True)
        packs.clear()
        del base
        reference.double()
        for row in conditions:
            steps = row["steps"]
            high = forward(torch.tensor([data[:steps]]))
            row["whole_model_f64_control"] = {name: compare(value, high) for name, value in cached_logits[steps].items()}
    OLD["save"](args.output, {"schema": "spiraltorch.native_projection_attribution.v1",
        "scope": "post-result numerical attribution; hybrid hooks and f64 are diagnostics, not replacements for the failed gate",
        "conditions": conditions, "script_sha256": OLD["digest"](Path(__file__)),
        "extension_sha256": OLD["digest"](Path(st._rs.__file__)), "torch": torch.__version__,
        "transformers": transformers.__version__, "model_sha256": OLD["PRIOR"]["PRIOR"]["MODEL"],
        "data_sha256": OLD["PRIOR"]["DATA"], "base_model_identity": OLD["BASE"]})


if __name__ == "__main__":
    main()
