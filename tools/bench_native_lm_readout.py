"""Offline CPU GPT-2 masked-loss benchmark: no geometry, training or quality claim."""
import argparse
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

TOKEN_PATTERN = [464, 3616, 286, 1204, 318, 407, 262, 976, 355, 37573, 4673, 4981, 460, 307, 973, 284, 4331]
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "torch-full", "torch-selected"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      SPIRALTORCH_AUTOTUNE_STORE=str(args.output/"autotune.json"))
    source = {name: digest(args.model_dir/name) for name in ("config.json", "model.safetensors")}
    smoothing = .1
    if args.engine == "native":
        class NoForeignML(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
                    raise AssertionError("native benchmark imported " + fullname)
        sys.meta_path.insert(0, NoForeignML())
        import spiraltorch as st
        weights = st.SafeTensorArchive(args.model_dir/"model.safetensors")
        model = st.NativeCausalLm.from_gpt2((args.model_dir/"config.json").read_text(), weights)
        del weights
        info = {"native_extension_sha256": digest(Path(st._rs.__file__)), "build": st.build_info(),
                "execution_backend": model.execution_backend, "optional_ml_imports_forbidden": True}
        vocabulary = json.loads(model.config_json())["vocabulary"]

        def loss(ids, labels):
            return model.next_token_loss(ids, [1, len(ids)], labels=labels, label_smoothing=smoothing)
    else:
        import torch
        import transformers
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest")
        model = transformers.GPT2LMHeadModel.from_pretrained(
            args.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
        model.config.use_cache = False
        vocabulary = model.config.vocab_size
        info = {"torch": torch.__version__, "transformers": transformers.__version__,
                "torch_threads": torch.get_num_threads(), "device": str(next(model.parameters()).device)}

        def loss(ids, labels):
            # Both engines include host integer transport and a host scalar result.
            tokens = torch.tensor(ids).reshape(1, -1)[:, :-1]
            targets = torch.tensor(labels[1:])
            with torch.inference_mode():
                if args.engine == "torch-selected":
                    hidden = model.transformer(tokens, use_cache=False).last_hidden_state.reshape(-1, model.config.n_embd)
                    selected = targets.ne(-100)
                    logits = model.lm_head(hidden[selected])
                    targets = targets[selected]
                else:
                    logits = model(tokens, use_cache=False).logits.reshape(-1, vocabulary)
                return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=smoothing).item()

    cases = []
    for steps in (16, 64):
        ids = (TOKEN_PATTERN * (steps//len(TOKEN_PATTERN)+2))[:steps+1]
        for active in (1, 4, steps):
            labels = [-100]*(len(ids)-active) + ids[-active:]
            cases.append({"id": f"rows{steps}-selected{active}", "context_rows": steps,
                          "selected_rows": active, "ids": ids, "labels": labels, "timings_ms": []})
    for case in cases:
        start = time.perf_counter_ns()
        case["loss"] = loss(case["ids"], case["labels"])
        case["first_call_ms"] = (time.perf_counter_ns()-start)/1e6
        if not math.isfinite(case["loss"]):
            raise ValueError("non-finite loss")
        loss(case["ids"], case["labels"])
    order = []
    rng = random.Random(7281)
    for repeat in range(5):
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        order.append(indices)
        for index in indices:
            case = cases[index]
            start = time.perf_counter_ns()
            result = loss(case["ids"], case["labels"])
            elapsed = (time.perf_counter_ns()-start)/1e6
            if result != case["loss"]:
                raise ValueError("fixed-input CPU loss changed during timing")
            case["timings_ms"].append(elapsed)
    for case in cases:
        case["median_ms"] = statistics.median(case["timings_ms"])
        print(json.dumps({"engine": args.engine, "case": case["id"], "loss": case["loss"], "median_ms": case["median_ms"]}), flush=True)
    with (args.output/"report.json").open("x") as stream:
        json.dump({"schema": "spiraltorch.native_lm_readout_bench.v1", "engine": args.engine,
                   "source_sha256": source, "script_sha256": digest(Path(__file__)), "vocabulary": vocabulary,
                   "label_smoothing": smoothing, "platform": platform.platform(), "python": platform.python_version(),
                   "thread_env": {name: os.environ[name] for name in THREAD_ENV}, "runtime": info,
                   "boundary": "host_integer_lists_to_mean_loss_scalar_cpu_f32_frozen_no_geometry",
                   "order": order, "cases": cases,
                   "limits": ["six synthetic-ID cases, five warm samples each; not a statistical performance claim",
                              "decoder contexts unchanged; torch-selected final norm still processes all context rows",
                              "no training/backward/geometry speed, GPU, RSS or quality measurement"]},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
