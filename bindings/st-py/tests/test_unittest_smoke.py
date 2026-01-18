import json
import tempfile
import unittest

import spiraltorch as st  # noqa: E402


class SpiralTorchSmokeTest(unittest.TestCase):
    def test_plugin_tensor_op(self) -> None:
        events: list[dict] = []
        sub_id = st.plugin.subscribe("TensorOp", lambda e: events.append(e))
        a = st.Tensor.rand(2, 2, seed=1)
        b = st.Tensor.rand(2, 2, seed=2)
        _ = a.add(b)
        st.plugin.unsubscribe("TensorOp", sub_id)
        self.assertTrue(events)
        self.assertEqual(events[-1]["type"], "TensorOp")

    def test_module_trainer_smoke(self) -> None:
        trainer = st.nn.ModuleTrainer(
            backend="cpu",
            curvature=-1.0,
            hyper_learning_rate=1e-2,
            fallback_learning_rate=1e-2,
        )
        schedule = trainer.roundtable(
            2,
            1,
            st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
        )
        model = st.nn.Sequential()
        model.add(st.nn.Linear("l1", 2, 1))
        model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)
        loss = st.nn.MeanSquaredError()
        x = st.Tensor.rand(2, 2, seed=3)
        y = st.Tensor.rand(2, 1, seed=4)
        stats = trainer.train_epoch(model, loss, [(x, y)], schedule)
        self.assertEqual(stats.batches, 1)

    def test_ops_register_execute(self) -> None:
        @st.ops.signature(2, 1)
        def forward(inputs):
            return [inputs[0].add(inputs[1])]

        st.ops.unregister("py_add")
        st.ops.register("py_add", forward, description="python add operator", backends=["python"])
        a = st.Tensor.rand(1, 1, seed=5)
        b = st.Tensor.rand(1, 1, seed=6)
        out = st.ops.execute("py_add", a, b, return_single=True)
        desc = st.ops.describe("py_add")
        self.assertIsInstance(out, st.Tensor)
        self.assertIn("py_add", desc)

    def test_plugin_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/events.jsonl"
            with st.plugin.record(path, event_types="TensorOp"):
                a = st.Tensor.rand(1, 1, seed=7)
                b = st.Tensor.rand(1, 1, seed=8)
                _ = a.add(b)
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line]
        self.assertTrue(lines)
        event = json.loads(lines[-1])
        self.assertEqual(event["type"], "TensorOp")

    def test_state_dict_io(self) -> None:
        model = st.nn.Linear("l1", 2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/linear.json"
            st.nn.save(path, model)
            manifest = path.replace(".json", ".manifest.json")
            loaded = st.nn.load(manifest)
        self.assertIsInstance(loaded, list)
        model.load_state_dict(loaded)


if __name__ == "__main__":
    unittest.main()
