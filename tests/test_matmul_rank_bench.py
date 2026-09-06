"""Equal scores must not admit different cutoff indices into a matched benchmark."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_matmul_rank_vs_torch import require_canonical_indices


class CanonicalRankTest(unittest.TestCase):
    def test_matching_rows_pass(self):
        require_canonical_indices([[0, 1], [3, 4]], [[0, 1], [3, 4]])

    def test_different_tied_selection_fails(self):
        with self.assertRaisesRegex(RuntimeError, "canonical stable"):
            require_canonical_indices([[0, 2, 3]], [[0, 1, 3]])

    def test_different_order_or_shape_fails(self):
        for actual in [[[1, 0]], [[0]], [[0, 1], []]]:
            with self.assertRaises(RuntimeError):
                require_canonical_indices(actual, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
