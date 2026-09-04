"""Object-lifetime checks for the packaged native Python bindings."""

import gc
import importlib
import unittest

import spiraltorch as st
import spiraltorch.spiralk as sk


class NativeBindingOwnershipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        native = importlib.import_module("spiraltorch.spiraltorch")
        if not isinstance(st.Tensor(1, 1, [0.0]), native.Tensor):
            raise RuntimeError("ownership checks require the packaged native Tensor")

    @unittest.skipUnless(hasattr(st.nn, "Linear"), "requires the nn feature")
    def test_state_dict_value_extraction_outlives_source_objects(self) -> None:
        source = st.nn.Linear("fc", 2, 1)
        source.load_state_dict(
            [
                ("fc::weight", st.Tensor(2, 1, [2.0, 3.0])),
                ("fc::bias", st.Tensor(1, 1, [1.0])),
            ]
        )
        inputs = st.Tensor(1, 2, [0.5, -0.25])
        expected = source.forward(inputs).tolist()
        self.assertEqual(expected, [[1.25]])
        state = source.state_dict()
        restored = st.nn.Linear("fc", 2, 1)
        self.assertNotEqual(restored.forward(inputs).tolist(), expected)
        restored.load_state_dict(state)
        del source, state
        gc.collect()
        self.assertEqual(restored.forward(inputs).tolist(), expected)

    def test_dlpack_transfer_outlives_the_exporter(self) -> None:
        source = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])
        capsule = source.to_dlpack()
        del source
        gc.collect()
        restored = st.Tensor.from_dlpack(capsule)
        del capsule
        gc.collect()
        self.assertEqual(restored.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_dlpack_capsule_can_only_be_consumed_once(self) -> None:
        source = st.Tensor(1, 2, [3.0, 5.0])
        capsule = source.to_dlpack()
        restored = st.Tensor.from_dlpack(capsule)
        with self.assertRaises(ValueError):
            st.Tensor.from_dlpack(capsule)
        self.assertEqual(restored.tolist(), [[3.0, 5.0]])

    @unittest.skipUnless(
        hasattr(st, "RankAdaptationSession"), "requires the kdsl feature"
    )
    def test_rank_adaptation_keeps_plan_provenance_and_reward_ownership(self) -> None:
        base = st.plan("topk", 2, 256, 8, backend="wgpu")
        session = st.RankAdaptationSession(
            base,
            ["u2: false;", "u2: true;"],
            policy="ucb",
            seed=17,
        )

        selection = session.choose()
        self.assertEqual(selection.selection_id, 1)
        self.assertEqual(session.pending_selection_id, 1)
        self.assertEqual(selection.plan.contract()["requested_backend"], "wgpu")
        self.assertEqual(
            selection.receipt()["semantic_owner"],
            "st-core::runtime::rank_adaptation",
        )
        with self.assertRaisesRegex(RuntimeError, "still awaiting observation"):
            session.choose()

        rejected = session.observe(selection.selection_id, 1.25, False)
        self.assertFalse(rejected["credited"])
        self.assertIsNone(rejected["reward"])
        self.assertEqual(
            sum(session.snapshot()["observation_counts"]["rank_plan_variant"].values()),
            0,
        )

        credited_selection = session.choose()
        credited = session.observe(credited_selection.selection_id, 4.0, True)
        self.assertTrue(credited["credited"])
        self.assertAlmostEqual(credited["reward"], 0.2)
        self.assertEqual(
            sum(session.snapshot()["observation_counts"]["rank_plan_variant"].values()),
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "no pending selection"):
            session.abandon(credited_selection.selection_id)

    def test_torch_bridge_outlives_both_source_wrappers(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            self.skipTest("optional PyTorch interoperability dependency is absent")
        if not hasattr(torch, "Tensor"):
            self.skipTest("a different test installed a torch stub")

        source = st.Tensor(1, 3, [2.0, 4.0, 6.0])
        shared = st.compat.torch.to_torch(source)
        del source
        gc.collect()
        self.assertEqual(shared.tolist(), [[2.0, 4.0, 6.0]])
        restored = st.compat.torch.from_torch(shared)
        del shared
        gc.collect()
        self.assertEqual(restored.tolist(), [[2.0, 4.0, 6.0]])

    @unittest.skipUnless(hasattr(sk, "rewrite_with_ai"), "requires the kdsl feature")
    def test_rust_callback_arguments_can_be_retained_by_python(self) -> None:
        ctx = sk.SpiralKContext(4, 64, 8, True, 32, 256, 64, 4, 1)
        config = sk.SpiralKAiRewriteConfig("local-test", max_hints=2)
        prompt = sk.SpiralKAiRewritePrompt("", ctx)
        retained = []

        def generator(received_config, received_prompt):
            retained.append((received_config, received_prompt))
            return [sk.SpiralKHeuristicHint("workgroup", "64", 0.9, "true")]

        output, script, hints = sk.rewrite_with_ai(
            "", ctx, config, prompt, generator=generator
        )
        del config, prompt
        gc.collect()
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0][0].model, "local-test")
        self.assertEqual(retained[0][1].base_program, "")
        self.assertEqual(output["soft"][0]["value"], 64)
        self.assertIn("soft (wg, 64", script)
        self.assertEqual(hints[0].field, "wg")

    @unittest.skipUnless(hasattr(sk, "rewrite_with_ai"), "requires the kdsl feature")
    def test_rust_callback_exceptions_are_not_swallowed(self) -> None:
        ctx = sk.SpiralKContext(4, 64, 8, True, 32, 256, 64, 4, 1)
        config = sk.SpiralKAiRewriteConfig("local-test")
        prompt = sk.SpiralKAiRewritePrompt("", ctx)

        def generator(_config, _prompt):
            raise RuntimeError("ownership-test-generator-failure")

        with self.assertRaisesRegex(RuntimeError, "ownership-test-generator-failure"):
            sk.rewrite_with_ai("", ctx, config, prompt, generator=generator)
        self.assertEqual(config.model, "local-test")


if __name__ == "__main__":
    unittest.main()
