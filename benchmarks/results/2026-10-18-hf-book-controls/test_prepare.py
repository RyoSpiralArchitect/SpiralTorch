import json
from pathlib import Path
import runpy
import tempfile
import unittest

API = runpy.run_path(str(Path(__file__).with_name("prepare.py")))


def fixture():
    lines = ["ignored preface", API["START"], "ignored front matter"]
    for i in range(1, 62):
        lines.extend([f"CHAPTER {API['roman'](i)}.", f"chapter {i} " * 30,
                      "[Illustration: not novel prose]"])
    lines.extend(["CHISWICK PRESS:--ignored colophon", API["END"], "ignored license"])
    return "\r\n".join(lines).encode()


class PreparationTests(unittest.TestCase):
    def test_nested_illustrations_are_balanced_and_other_brackets_survive(self):
        text, count = API["remove_illustrations"](
            "before [kept] [Illustration: caption [_Copyright_]] after")
        self.assertEqual(text, "before [kept]  after")
        self.assertEqual(count, 1)
        with self.assertRaises(ValueError):
            API["remove_illustrations"]("before [Illustration: nested [inner] after")

    def test_bounded_reads_and_changed_policy_reject_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bounded").write_bytes(b"12345")
            self.assertEqual(API["read_bounded"](root / "bounded", 5), b"12345")
            with self.assertRaises(ValueError):
                API["read_bounded"](root / "bounded", 4)
            raw = fixture()
            (root / "book.txt").write_bytes(raw)
            plan = json.loads(Path(__file__).with_name("plan.json").read_text())
            plan["source"]["raw_sha256"] = API["sha"](raw)
            plan["data_policy"]["normalization"] = "different"
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                API["prepare"](root / "book.txt", root / "plan.json", root / "chapters")
            self.assertFalse((root / "chapters").exists())

    def test_exact_chapters_not_front_matter_license_or_illustrations(self):
        raw = fixture()
        records = API["chapters"](raw, API["sha"](raw))
        self.assertEqual(len(records), 61)
        self.assertTrue(all(removed == 1 for _, removed in records))
        self.assertTrue(all("ignored" not in text and "Illustration" not in text for text, _ in records))
        self.assertTrue(all("\r" not in text for text, _ in records))

    def test_pin_order_duplicates_and_missing_footer_reject(self):
        raw = fixture()
        with self.assertRaises(ValueError):
            API["chapters"](raw + b"x", API["sha"](raw))
        for bad in (raw.replace(b"CHAPTER II.", b"CHAPTER I."),
                    raw.replace(API["END"].encode(), b"missing"),
                    raw + API["END"].encode(),
                    raw.replace(b"CHISWICK PRESS:--", b"missing")):
            with self.assertRaises(ValueError):
                API["chapters"](bad, API["sha"](bad))

    def test_all_splits_written_once_and_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = fixture()
            (root / "book.txt").write_bytes(raw)
            plan = json.loads(Path(__file__).with_name("plan.json").read_text())
            plan["source"]["raw_sha256"] = API["sha"](raw)
            (root / "plan.json").write_text(json.dumps(plan))
            result = API["prepare"](root / "book.txt", root / "plan.json", root / "chapters")
            self.assertEqual([sum(d["split"] == split for d in result["documents"])
                              for split in ("train", "development", "reserve")], [49, 6, 6])
            self.assertFalse(result["reserve_used"])
            for doc in result["documents"]:
                self.assertEqual(API["sha"]((root / "chapters" / doc["relative_path"]).read_bytes()), doc["sha256"])
            before = (root / "chapters/manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                API["prepare"](root / "book.txt", root / "plan.json", root / "chapters")
            self.assertEqual(before, (root / "chapters/manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
