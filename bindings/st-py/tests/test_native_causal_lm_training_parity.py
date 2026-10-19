"""Optional Torch comparison of the complete ordered native corpus training loop."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))
DOCS = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 0, 1], [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2]]
# Independent expected spans, not the trainer's generated plan.
SPANS = [(1, 0), (1, 4), (2, 0), (2, 4), (2, 8)]


@pytest.mark.parametrize("kind", helpers["KINDS"])
@pytest.mark.parametrize("smoothing", [0., .2])
def test_corpus_updates_and_restored_cursor_match_torch(kind, smoothing):
    reference, base = helpers["pair"](width=3, heads=1)
    reference.requires_grad_(False)
    record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"][0]["path"] = "transformer.h.0.attn"
    if kind == "affine":
        record["placements"][0]["recipe"]["max_values"] = 96
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(artifact)
    stack, handles = helpers["attach"](reference, artifact)
    named = dict(stack.adapters[0].named_parameters())
    rate, decay, clip = .02, .01, .001
    optimizer = torch.optim.AdamW(stack.parameters(), lr=rate, betas=(.7, .9), eps=.05,
                                 weight_decay=decay, foreach=False)
    labels = [list(d) for d in DOCS]
    labels[1][1] = labels[2][1] = -100
    native_opt = st.NativeGeometryAdamW(artifact, beta1=.7, beta2=.9, epsilon=.05, max_gradient_norm=clip)
    trainer = st.NativeCausalLmTrainer(native, native_opt, DOCS, 4, labels=labels, learning_rate=rate,
                                     weight_decay=decay, label_smoothing=smoothing,
                                     microbatch_windows=2, accumulation_steps=2, epochs=2)
    try:
        for update, (start, end) in enumerate(((0, 4), (4, 5), (0, 4), (4, 5))):
            optimizer.zero_grad(set_to_none=True)
            count = sum(v != -100 for d, s in SPANS[start:end] for v in labels[d][s+1:s+4])
            expected_loss = 0.
            for offset in range(start, end, 2):
                spans = SPANS[offset:min(offset+2, end)]
                ids = torch.tensor([DOCS[d][s:s+4] for d, s in spans])
                targets = torch.tensor([labels[d][s+1:s+4] for d, s in spans]).reshape(-1)
                logits = reference(ids[:, :-1], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets,
                                                        ignore_index=-100, reduction="sum", label_smoothing=smoothing) / count
                loss.backward()
                expected_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(stack.parameters(), clip)
            optimizer.step()
            report = trainer.step()
            assert (report.update, report.epoch, report.window_start, report.window_end) == (update+1, update//2, start, end)
            assert report.target_tokens == count
            assert report.loss == pytest.approx(expected_loss, abs=3e-6)
            current = json.loads(trainer.model.geometry_artifact().state_json())["placements"][0]
            for parameter in current["tensors"]:
                if parameter["role"] == "parameter":
                    expected = named[parameter["name"]]
                    torch.testing.assert_close(torch.tensor(parameter["values"]).reshape_as(expected), expected, atol=3e-6, rtol=3e-4)
            if update == 0:
                trainer = st.NativeCausalLmTrainer.from_json(base, DOCS, trainer.to_json(), trainer.checkpoint_id, labels=labels)
        assert trainer.exhausted and trainer.step() is None
        assert all(p.grad is None for p in reference.parameters())
    finally:
        for handle in handles:
            handle.remove()
