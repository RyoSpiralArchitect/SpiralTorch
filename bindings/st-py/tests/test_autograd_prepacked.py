"""Actual native packed graph handles, not Python fallback implementations."""
import gc
import unittest

import spiraltorch as st


def node(rows, cols, values, trainable=True):
    return st.AutogradTensor(st.Tensor(rows, cols, values), trainable)


class AutogradPrepackedTests(unittest.TestCase):
    def test_nonleaf_source_lifetime_and_reused_gradients(self):
        x = node(2, 3, [1, 2, -1, 0.5, -2, 3])
        weight = node(3, 2, [2, -1, 0, 3, 1.5, 2])
        rhs = weight.scale(0.5)
        expected = x.matmul(rhs).value().tolist()
        packed = rhs.prepack_rhs()
        self.assertIsInstance(packed, st.AutogradPackedRhs)
        self.assertEqual(packed.source_id(), rhs.id())
        self.assertEqual(packed.shape(), (3, 2))
        self.assertTrue(packed.requires_grad())
        del rhs
        gc.collect()
        first = x.matmul_prepacked(packed)
        self.assertEqual(first.operation_name(), "matmul")
        self.assertEqual(first.value().tolist(), expected)
        report = first.add(x.matmul_prepacked(packed)).sum().backward()
        self.assertEqual(report["leaf_gradient_count"], 2)
        self.assertEqual(x.grad().tolist(), [[1, 3, 3.5], [1, 3, 3.5]])
        self.assertEqual(weight.grad().tolist(), [[1.5, 1.5], [0, 0], [2, 2]])

    def test_snapshot_requires_explicit_refresh_after_step(self):
        x, weight = node(1, 1, [2], False), node(1, 1, [3])
        packed = weight.prepack_rhs()
        optimizer = st.AutogradSgd([weight], 0.5)
        x.matmul_prepacked(packed).backward()
        optimizer.step()
        fresh = optimizer.parameter(0).prepack_rhs()
        self.assertNotEqual(packed.source_id(), fresh.source_id())
        self.assertEqual(x.matmul_prepacked(packed).item(), 6)
        self.assertEqual(x.matmul_prepacked(fresh).item(), 4)

    def test_learning_through_one_reused_frozen_projection(self):
        packed = node(1, 1, [2], False).prepack_rhs()
        self.assertFalse(packed.requires_grad())
        target = node(1, 1, [6], False)
        optimizer = st.AutogradSgd([node(1, 1, [0])], 0.05)
        for _ in range(40):
            x = optimizer.parameter(0)
            loss = x.matmul_prepacked(packed).mean_squared_error(target)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 1e-9)
        self.assertAlmostEqual(optimizer.parameter(0).item(), 3, places=5)

    def test_shared_leaf_empty_and_invalid_inputs(self):
        x = node(1, 1, [2])
        packed = x.prepack_rhs()
        x.matmul_prepacked(packed).backward()
        self.assertEqual(x.grad().tolist(), [[4]])
        with self.assertRaises(RuntimeError):
            node(1, 2, [1, 2]).matmul_prepacked(packed)
        with self.assertRaises(TypeError):
            x.matmul_prepacked(x)
        self.assertEqual(x.grad().tolist(), [[4]])
        for rows, inner, cols in [(0, 3, 2), (2, 0, 3), (2, 3, 0)]:
            lhs = node(rows, inner, [0] * (rows * inner))
            rhs = node(inner, cols, [0] * (inner * cols))
            output = lhs.matmul_prepacked(rhs.prepack_rhs())
            self.assertEqual(output.shape(), (rows, cols))
            output.sum().backward()
            self.assertEqual(lhs.grad().shape(), (rows, inner))
            self.assertEqual(rhs.grad().shape(), (inner, cols))


if __name__ == "__main__":
    unittest.main()
