import json
import pickle
from pathlib import Path
import pytest
import spiraltorch as st


def test_native_windows_preserve_boundaries_and_account_for_tails():
    plan = st.CausalDocumentWindows([0, 3, 5, 12], 4)
    assert [plan.window(i) for i in range(len(plan))] == [(2, 0, 4), (3, 0, 4), (3, 4, 4), (3, 8, 4)]
    state = json.loads(plan.to_json())
    assert (state["total_tokens"], state["dropped_tokens"], state["target_tokens"]) == (20, 4, 12)
    assert plan.plan_id == st.CausalDocumentWindows.create_json(
        '{"document_lengths":[0,3,5,12],"context_length":4}').plan_id
    fixture = json.loads((Path(__file__).resolve().parents[3] / "crates/st-nn/tests/fixtures/causal_document_windows.fixture.json").read_text())
    assert plan.plan_id == fixture["plan_id"]
    with pytest.raises(IndexError):
        plan.window(4)


def test_short_documents_cannot_be_packed_into_a_window():
    assert len(st.CausalDocumentWindows([3, 3], 4)) == 0
    with pytest.raises(ValueError, match="no full document"):
        st.HfGeometryDataset([[1, 2, 3], [4, 5, 6]], 4)


def test_dataset_is_immutable_and_first_label_is_not_a_target():
    docs = [[0, 1, 2, 3, 4], list(range(10, 18))]
    data = st.HfGeometryDataset(docs, 4)
    docs[0][0] = 999
    assert len(data) == 3
    assert data[0] == {"input_ids": [0, 1, 2, 3], "labels": [-100, 1, 2, 3], "attention_mask": [1] * 4}
    data[0]["input_ids"][0] = 888
    assert data[0]["input_ids"][0] == 0
    assert data[-1]["input_ids"] == [14, 15, 16, 17]
    with pytest.raises(IndexError):
        _ = data[-4]


def test_tokenizer_keeps_documents_independent_without_padding_or_truncation():
    class Tokenizer:
        def __call__(self, documents, **options):
            assert documents == ["abcd", "efghi"]
            assert options == dict(padding=False, truncation=False, add_special_tokens=False, return_attention_mask=False)
            return {"input_ids": [[ord(c) for c in text] for text in documents]}
    data = st.HfGeometryDataset.from_texts(Tokenizer(), ["abcd", "efghi"], 4, add_special_tokens=False)
    assert data[1]["input_ids"] == list(map(ord, "efgh"))
    assert json.loads(data.plan.to_json())["dropped_tokens"] == 1
    with pytest.raises(TypeError, match="bare string"):
        st.HfGeometryDataset.from_texts(Tokenizer(), "abcd", 4)


@pytest.mark.parametrize("value", [True, -1, 1.5, 2**32])
def test_invalid_token_ids_are_not_coerced(value):
    with pytest.raises((TypeError, ValueError)):
        st.HfGeometryDataset([[0, value, 2, 3]], 4)


def test_plan_identity_is_not_a_claim_about_token_content():
    a = st.HfGeometryDataset([[0, 1, 2, 3]], 4)
    b = st.HfGeometryDataset([[4, 5, 6, 7]], 4)
    assert a.plan.plan_id == b.plan.plan_id
    assert a[0] != b[0]


def test_document_budget_precedes_sequence_extraction():
    class Oversized:
        def __len__(self):
            return st.CausalDocumentWindows.MAX_DOCUMENTS + 1

        def __getitem__(self, index):
            raise AssertionError("over-budget sequence was accessed")

    for construct in (st.CausalDocumentWindows, st.HfGeometryDataset):
        with pytest.raises(ValueError, match="document count"):
            construct(Oversized(), 4)
    with pytest.raises(ValueError, match="document count"):
        st.HfGeometryDataset.from_texts(None, Oversized(), 4)


def test_native_conversion_is_bounded_even_if_iterator_disagrees():
    class Lengths:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            assert 0 <= index < 2
            return 4

        def __iter__(self):
            raise AssertionError("unbounded iteration used")

    assert len(st.CausalDocumentWindows(Lengths(), 4)) == 2


@pytest.mark.parametrize("values, context", [([True, 4], 4), ([4], True)])
def test_native_constructor_rejects_python_booleans(values, context):
    with pytest.raises(TypeError, match="booleans"):
        st.CausalDocumentWindows(values, context)
    with pytest.raises(TypeError, match="booleans"):
        st.CausalDocumentWindows([4], 4).window(True)


@pytest.mark.parametrize("module", ["numpy", "torch"])
def test_boolean_array_scalars_are_not_token_ids_or_lengths(module):
    library = pytest.importorskip(module)
    value = library.bool_(True) if module == "numpy" else library.tensor(True)
    with pytest.raises(TypeError, match="booleans"):
        st.HfGeometryDataset([[0, value, 2, 3]], 4)
    with pytest.raises(TypeError, match="booleans"):
        st.CausalDocumentWindows([value, 4], 4)
    integer = library.int64(4) if module == "numpy" else library.tensor(4)
    assert len(st.CausalDocumentWindows([integer], 4)) == 1


def test_pickled_dataset_reconstructs_the_same_rust_plan():
    original = st.HfGeometryDataset([list(range(9)), list(range(4))], 4)
    restored = pickle.loads(pickle.dumps(original))
    assert restored.plan.plan_id == original.plan.plan_id
    assert list(restored) == list(original)


def test_context_integer_is_frozen_once_before_planning_and_pickling():
    class ChangingContext:
        calls = 0

        def __index__(self):
            self.calls += 1
            return 4 if self.calls == 1 else 2

    context = ChangingContext()
    data = st.HfGeometryDataset([list(range(8))], context)
    restored = pickle.loads(pickle.dumps(data))
    assert context.calls == 1
    assert len(data) == len(restored) == 2
    assert data.plan.plan_id == restored.plan.plan_id


def test_actual_spawn_worker_receives_document_safe_rows():
    torch = pytest.importorskip("torch")
    data = st.HfGeometryDataset([list(range(8))], 4)
    loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=1,
                                         multiprocessing_context="spawn", timeout=60)
    assert list(loader) == list(data)


def test_example_requires_previous_dataset_identity_before_resume(tmp_path, capsys):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "examples/hf_geometry_finetune.py"
    spec = importlib.util.spec_from_file_location("hf_geometry_finetune_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    with pytest.raises(SystemExit) as error:
        example.main(["--model-dir", "base", "--geometry-artifact", "initial.json",
                      "--geometry-artifact-id", "pin", "--train-text", "train.txt",
                      "--eval-text", "eval.txt", "--output", str(tmp_path / "new"),
                      "--resume-from-checkpoint", "checkpoint-8", "--geometry-checkpoint-id", "pin"])
    assert error.value.code == 2
    assert "--expected-dataset-id" in capsys.readouterr().err


def test_real_hf_trainer_consumes_native_document_windows(tmp_path):
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = hf.GPT2LMHeadModel(hf.GPT2Config(vocab_size=32, n_embd=8, n_layer=1, n_head=2,
                                           n_positions=8, use_cache=False)).requires_grad_(False)
    stack = st.GeometryAdapterStack({"transformer.h.0.mlp": st.WaveGateAdapter(8)})
    data = st.HfGeometryDataset([list(range(16)), list(range(16, 32))], 8)
    trainer = st.hf_geometry_trainer_class()(
        model=model, geometry=stack, base_model_identity="sha256:" + "a" * 64,
        args=hf.TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=2,
                                  per_device_train_batch_size=2, report_to=[], save_strategy="no", disable_tqdm=True),
        train_dataset=data, data_collator=hf.default_data_collator)
    before = stack.adapters[0].gate.detach().clone()
    trainer.train()
    assert trainer.state.global_step == 2
    assert not torch.equal(before, stack.adapters[0].gate)
    assert all(p.grad is None for p in model.parameters())
