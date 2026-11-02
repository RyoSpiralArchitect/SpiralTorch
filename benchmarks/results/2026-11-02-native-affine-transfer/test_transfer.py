import sys
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sample import chapters, roman
from run_transfer import cohort_delta
from transfer import C1, C2, NAMES, TERMINAL, check_fit, save, sha


def frame(text):
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        + text
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***"
    )


class TransferChecks(unittest.TestCase):
    def test_austen_actual_headings_and_volume_markers(self):
        body = "\n".join("CHAPTER " + roman(i) + ".\nbody" for i in range(1, 51))
        result = chapters(
            frame(" CHAPTER I\n" + body + "\nEND OF THE THIRD VOLUME"), "austen"
        )
        self.assertEqual(result, ["body"] * 50)

    def test_shelley_prefatory_letters_and_contents_are_not_chapters(self):
        sections = "\n".join(f"Chapter {i}\ntext {i}" for i in range(1, 25))
        self.assertEqual(
            chapters(frame(" Chapter 1\nLetter 4\n" + sections), "shelley")[-1],
            "text 24",
        )
        with self.assertRaises(ValueError):
            chapters(frame(sections.replace("Chapter 12\n", "Chapter 11\n")), "shelley")

    def test_faraday_ignores_contents_and_appendix_and_rejects_missing_boundary(self):
        sections = "\n".join("LECTURE " + roman(i) + ".\nbody" for i in range(1, 7))
        body = "LECTURE I.\ncontents\nTHE CHEMICAL HISTORY OF A CANDLE\n" + sections
        self.assertEqual(
            chapters(frame(body + "\nLECTURE ON PLATINUM.\nappendix"), "faraday"),
            ["body"] * 6,
        )
        with self.assertRaises(ValueError):
            chapters(frame(body), "faraday")

    def test_cohorts_are_not_pooled_or_relabelled(self):
        seeds = C1 + C2
        rows = {str(s): {"free_minus_affine": (-1 if s in C1 else 2)} for s in seeds}
        self.assertEqual((cohort_delta(rows, C1), cohort_delta(rows, C2)), (-1, 2))
        self.assertEqual(len(set(TERMINAL)), 12)
        self.assertEqual(len(set(NAMES)), 15)

    def test_substituted_terminal_pin_is_refused_before_parsing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terminal.json"
            save(path, {"updates": 255})
            with self.assertRaisesRegex(ValueError, "terminal state"):
                check_fit(path, {"checkpoint_id": "sha256:" + "0" * 64}, "not-used")
            with self.assertRaisesRegex(ValueError, "terminal state"):
                check_fit(path, {"checkpoint_id": "sha256:" + sha(path)}, "not-used")


if __name__ == "__main__":
    unittest.main()
