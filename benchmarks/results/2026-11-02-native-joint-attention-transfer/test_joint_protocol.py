import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import CASES, KEYS, STATES, chapters, divergence, roman


def frame(body):
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        + body
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***"
    )


class ProtocolTests(unittest.TestCase):
    def test_roster_keeps_all_arms_and_rotates(self):
        self.assertEqual(len(set(KEYS)), 9)
        self.assertEqual(len(set(STATES)), 13)
        self.assertEqual([CASES[i][1] for i in (0, 3, 6)], ["lora", "affine", "wave"])

    def test_wells_divisions_and_contents(self):
        def volume(n):
            return "\n".join(f"{roman(i)}.\nTitle {i}\nText" for i in range(1, n + 1))

        text = frame(
            " I. Contents\nBOOK ONE\n" + volume(17) + "\nBOOK TWO\n" + volume(10)
        )
        parts = chapters(text, "worlds")
        self.assertEqual(len(parts), 27)
        self.assertEqual(parts[17], "Title 1 Text")
        with self.assertRaises(ValueError):
            chapters(text.replace("BOOK TWO", "BOOK ONE"), "worlds")

    def test_book_heading_order_refuses_missing_chapter(self):
        for book, n in [("mansfield", 48), ("beagle", 21)]:
            body = "\n".join(
                "CHAPTER " + roman(i) + "\nparagraph" for i in range(1, n + 1)
            )
            self.assertEqual(chapters(frame(body), book), ["paragraph"] * n)
            with self.assertRaises(ValueError):
                chapters(frame(body.replace("CHAPTER II\n", "CHAPTER I\n")), book)

    def test_continuation_difference_handles_eos_prefix_and_equality(self):
        self.assertIsNone(divergence([1, 2], [1, 2]))
        self.assertEqual(divergence([1, 2], [1]), 1)
        self.assertEqual(divergence([1, 2], [9, 2]), 0)


if __name__ == "__main__":
    unittest.main()
