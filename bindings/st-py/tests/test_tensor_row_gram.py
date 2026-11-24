"""The native row Gram uses wide CPU accumulation without changing ordinary matmul."""
import unittest

import spiraltorch as st


@unittest.skipUnless(getattr(st, "_rs", None) is not None, "native CPU extension required")
class WideRowGram(unittest.TestCase):
    def test_cancellation_and_pair_order(self):
        a = st.Tensor(2, 4, [1.] * 4 + [0., 0., 0., 1.])
        b = st.Tensor(3, 4, [1e8, 1., -1e8, 1., 1., 2., 3., 4., 0., 0., 0., 0.])
        self.assertEqual(a.row_gram_wide(b).tolist(), [[2., 10., 0.], [1., 4., 0.]])
        self.assertEqual(a.row_gram_wide().tolist(), [[4., 1.], [1., 1.]])
        self.assertEqual(a.row_gram_wide(a).tolist(), a.row_gram_wide().tolist())

    def test_empty_shapes_and_rejection(self):
        lhs = st.Tensor.zeros(2, 0)
        self.assertEqual(lhs.row_gram_wide(st.Tensor.zeros(3, 0)).tolist(), [[0.] * 3] * 2)
        self.assertEqual(st.Tensor.zeros(0, 4).row_gram_wide(st.Tensor.zeros(3, 4)).shape(), (0, 3))
        with self.assertRaises((ValueError, RuntimeError)):
            lhs.row_gram_wide(st.Tensor.zeros(0, 3))


if __name__ == "__main__":
    unittest.main()
