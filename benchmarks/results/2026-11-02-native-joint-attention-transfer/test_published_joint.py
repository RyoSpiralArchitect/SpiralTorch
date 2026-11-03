import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_public import check, check_attempt, check_cross, check_registration, read


class PublishedJointTests(unittest.TestCase):
    def test_duplicate_section_cannot_preserve_the_verified_primary(self):
        primary, public = read(HERE / "results.json"), read(HERE / "score_roster.json")
        check(primary, public)
        fake = copy.deepcopy(public)
        groups = fake["groups"]
        groups["mansfield-5"]["common"]["data_id"] = groups["mansfield-13"]["common"][
            "data_id"
        ]
        with self.assertRaisesRegex(ValueError, "another chapter"):
            check(primary, fake)

    def test_published_paired_projection_and_first_attempt_cannot_be_relabelled(self):
        primary, x = (
            read(HERE / "results.json"),
            read(HERE / "interchange_results.json"),
        )
        attempt = read(HERE / "interchange_attempt.json")
        check_cross(primary, x)
        check_attempt(x, attempt)
        fake = copy.deepcopy(x)
        row = next(
            v
            for v in fake["section_nll"]["mansfield"].values()
            if v["131-AW"] != v["131-WA"]
        )
        row["131-AW"], row["131-WA"] = row["131-WA"], row["131-AW"]
        with self.assertRaisesRegex(ValueError, "attribution"):
            check_cross(primary, fake)
        fake = copy.deepcopy(attempt)
        fake["status"]["attempts"] = 2
        with self.assertRaisesRegex(ValueError, "first-attempt"):
            check_attempt(x, fake)

    def test_registration_cannot_be_reported_as_a_new_run(self):
        x = read(HERE / "interchange_results.json")
        record = read(HERE / "interchange_registration.json")
        check_registration(x, record)
        record["status"]["new_worker_starts"] = 1
        with self.assertRaisesRegex(ValueError, "file-only registration"):
            check_registration(x, record)


if __name__ == "__main__":
    unittest.main()
