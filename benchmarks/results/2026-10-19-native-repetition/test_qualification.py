"""Publication completeness and pins are separate from model evidence."""
import copy
from pathlib import Path
import runpy
import tempfile
import unittest

Q = runpy.run_path(str(Path(__file__).with_name("qualify.py")))


class PublicationTests(unittest.TestCase):
    def test_all_conditions_exact_booleans_and_retained_checkpoint_pins_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, replay = {}, {}
            for name in Q["CONFIGS"]:
                raw[name] = {"reports": [{"update": 1}, {"update": 2}]}
                for field, number in (("first", 1), ("final", 2)):
                    path = root / (name + "-step" + str(number) + ".json")
                    path.write_text('{"fixture":true}')
                    raw[name][field] = {"file": path.name, "sha256": Q["digest"](path)}
                replay[name] = True
            Q["validate_results"](root, raw, replay)
            name = next(iter(raw))
            for value in (False, "true", 1, None):
                with self.assertRaises(ValueError):
                    Q["validate_results"](root, raw, dict(replay, **{name: value}))
            missing = dict(raw)
            del missing[name]
            with self.assertRaises(ValueError):
                Q["validate_results"](root, missing, replay)
            changed = copy.deepcopy(raw)
            changed[name]["reports"].pop()
            with self.assertRaises(ValueError):
                Q["validate_results"](root, changed, replay)
            changed = copy.deepcopy(raw)
            changed[name]["final"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                Q["validate_results"](root, changed, replay)
            (root / raw[name]["final"]["file"]).write_text("changed")
            with self.assertRaises(ValueError):
                Q["validate_results"](root, raw, replay)


if __name__ == "__main__":
    unittest.main()
