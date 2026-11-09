from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_completions import CompletionMenu
from field_head_pairs import rotated_pair_control, suffix_pairs, training_head_rivals


class TrainingContrasts(unittest.TestCase):
    def test_two_directions_not_a_preference_for_long_answers(self):
        m = CompletionMenu(9, 0, ((1, 2, 3), (2, 3), (3,), (4,)))
        self.assertEqual(set(suffix_pairs(m)), {((3,), (1, 2, 3)), ((2, 3), (1, 2, 3)),
                                                ((3,), (2, 3))})
        self.assertEqual(training_head_rivals(m), {(1, 2, 3): (2, 3), (2, 3): (1, 3),
                                                   (3,): (1, 2), (4,): ()})

    def test_shared_head_is_not_a_rival_and_repeated_class_counts_once(self):
        m = CompletionMenu(9, 0, ((1, 2, 3), (1, 3), (3,), (3, 3)))
        self.assertEqual(training_head_rivals(m)[(3,)], (1,))
        self.assertNotIn(((3,), (3, 3)), suffix_pairs(m))

    def test_whole_suffix_in_same_menu_only(self):
        a = CompletionMenu(9, 0, ((1,), (1, 2), (4, 2, 3)))
        b = CompletionMenu(9, 0, ((2,), (2, 3), (5, 2)))
        self.assertEqual(suffix_pairs(a), ())
        self.assertEqual(suffix_pairs(b), (((2,), (5, 2)),))
        self.assertNotIn((4, 2, 3), training_head_rivals(b))

    def test_unrelated_control_has_identical_head_exposure_not_training_pairs(self):
        menu = CompletionMenu(10, 0, ((1,), (2,), (3,), (4, 2), (5, 1), (6, 8, 3), (9,)))
        original = suffix_pairs(menu)
        shifts = rotated_pair_control(menu)
        self.assertFalse(set(original) & set(shifts))
        for i in (0, 1):
            self.assertEqual(sorted(p[i] for p in original), sorted(p[i] for p in shifts))
        for ambiguous in [((1, 2, 3), (2, 3), (3,)), ((1,),), ((1,), (2, 1))]:
            with self.assertRaises(ValueError):
                rotated_pair_control(CompletionMenu(10, 0, ambiguous))


if __name__ == "__main__":
    unittest.main()
