"""Exercise the real Rust Module GELU, not a Python reference implementation."""
import math
import unittest

import spiraltorch as st


class GeluHostTests(unittest.TestCase):
    def test_forward_and_supplied_seed_derivative(self):
        for rows, cols in [(2, 6), (33, 195)]:
            values = [(i % 131) / 16 - 4 for i in range(rows * cols)]
            seeds = [(i % 29) / 16 - 0.5 for i in range(rows * cols)]
            x, g = st.Tensor(rows, cols, values), st.Tensor(rows, cols, seeds)
            layer = st.nn.Gelu()
            output = sum(layer(x).tolist(), [])
            gradient = sum(layer.backward(x, g).tolist(), [])
            c = math.sqrt(2 / math.pi)
            for value, seed, y, dx in zip(values, seeds, output, gradient):
                t = math.tanh(c * (value + 0.044715 * value ** 3))
                expected = 0.5 * value * (1 + t)
                derivative = (0.5 * (1 + t) + 0.5 * value * (1 - t * t)
                              * c * (1 + 3 * 0.044715 * value * value)) * seed
                self.assertLessEqual(abs(y - expected), 2e-6 * (1 + abs(expected)))
                self.assertLessEqual(abs(dx - derivative), 2e-6 * (1 + abs(derivative)))
            self.assertEqual(sum(x.tolist(), []), values)
            self.assertEqual(sum(g.tolist(), []), seeds)

    def test_checked_forward_retains_errors_and_signed_zero(self):
        for values, label in [([3e38, float("nan")], "gelu_input"),
                              ([3e38], "gelu_square"), ([1e14], "gelu_cubic")]:
            with self.assertRaisesRegex(RuntimeError, label):
                st.nn.Gelu()(st.Tensor(1, len(values), values))
        result = st.nn.Gelu()(st.Tensor(1, 2, [-0.0, 0.0])).tolist()[0]
        self.assertEqual([math.copysign(1, x) for x in result], [-1, 1])


if __name__ == "__main__":
    unittest.main()
