"""Synthetic wheel/install identity tests; no native code or GPU is loaded."""
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_artifacts as artifacts


def files(target="macosx_14_0_universal2"):
    native = "spiraltorch/spiraltorch.pyd" if target == "win_amd64" else "spiraltorch/spiraltorch.abi3.so"
    return {
        "spiraltorch/__init__.py": b"# synthetic, never executed\n",
        "spiraltorch/helper.py": b"value = 1\n",
        native: b"synthetic native bytes, never loaded",
        "spiraltorch-0.4.33.dist-info/METADATA": b"Name: spiraltorch\nVersion: 0.4.33\n",
        "spiraltorch-0.4.33.dist-info/WHEEL": ("Wheel-Version: 1.0\nTag: cp38-abi3-" + target + "\n").encode(),
        "spiraltorch-0.4.33.dist-info/RECORD": b"",
    }


def wheel(directory, contents=None, target="macosx_14_0_universal2"):
    path = Path(directory) / f"spiraltorch-0.4.33-cp38-abi3-{target}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in (files(target) if contents is None else contents).items():
            archive.writestr(name, raw)
    return path


def install(root, contents):
    for name, raw in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


class ArtifactIdentity(unittest.TestCase):
    def test_shipped_spiral_and_examples_are_checked_not_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            contents = files()
            contents.update({"spiral/__init__.py": b"# synthetic facade\n",
                             "examples/demo.py": b"# synthetic example\n"})
            inventory = artifacts.wheel_inventory(wheel(temporary, contents))
            install(root, contents)
            artifacts.verify_installed_files(inventory, root, root / "spiraltorch/__init__.py", "0.4.33")
            (root / "spiral/__init__.py").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "file differs"):
                artifacts.verify_installed_files(inventory, root, root / "spiraltorch/__init__.py", "0.4.33")

    def test_all_three_platforms_and_every_original_file_are_frozen(self):
        for target, family in (("macosx_14_0_universal2", "macos"),
                               ("manylinux2014_x86_64", "linux"), ("win_amd64", "windows")):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                inventory = artifacts.wheel_inventory(wheel(temporary, target=target))
                self.assertEqual(inventory["platform"], family)
                self.assertEqual(set(inventory["files"]), set(files(target)))
                self.assertTrue(inventory["native"]["path"].endswith(".pyd" if family == "windows" else ".so"))

    def test_changed_archive_and_wrong_platform_native_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = wheel(temporary)
            with patch.object(artifacts, "sha256", side_effect=["a" * 64, "b" * 64]):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    artifacts.wheel_inventory(path)
            contents = files()
            contents["spiraltorch/spiraltorch.pyd"] = contents.pop("spiraltorch/spiraltorch.abi3.so")
            with self.assertRaisesRegex(ValueError, "does not match"):
                artifacts.wheel_inventory(wheel(temporary, contents))

    def test_compressed_universal2_tags_match_the_filename(self):
        targets = ["macosx_14_0_x86_64", "macosx_14_0_arm64", "macosx_14_0_universal2"]
        contents = files()
        contents["spiraltorch-0.4.33.dist-info/WHEEL"] = (
            "Wheel-Version: 1.0\n" + "".join("Tag: cp38-abi3-" + t + "\n" for t in targets)).encode()
        with tempfile.TemporaryDirectory() as temporary:
            inventory = artifacts.wheel_inventory(wheel(temporary, contents, ".".join(targets)))
            self.assertEqual(len(inventory["tags"]), 3)
            self.assertEqual(inventory["platform"], "macos")

    def test_python_metadata_and_native_changes_are_all_detected(self):
        for changed in ("spiraltorch/helper.py", "spiraltorch/spiraltorch.abi3.so",
                        "spiraltorch-0.4.33.dist-info/METADATA", "spiraltorch-0.4.33.dist-info/WHEEL"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "site"
                inventory = artifacts.wheel_inventory(wheel(temporary))
                install(root, files())
                result = artifacts.verify_installed_files(inventory, root, root / "spiraltorch/__init__.py", "0.4.33")
                self.assertTrue(result["all_original_files_match"])
                (root / changed).write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "file differs"):
                    artifacts.verify_installed_files(inventory, root, root / "spiraltorch/__init__.py", "0.4.33")

    def test_extra_cached_missing_and_symlinked_package_files_reject(self):
        for defect in ("extra", "cached", "missing", "symlink"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "site"
                inventory = artifacts.wheel_inventory(wheel(temporary))
                install(root, files())
                if defect in ("extra", "cached"):
                    (root / "spiraltorch" / ("extra.py" if defect == "extra" else "cached.pyc")).write_bytes(b"x")
                else:
                    (root / "spiraltorch/helper.py").unlink()
                    if defect == "symlink":
                        (root / "spiraltorch/helper.py").symlink_to(root / "spiraltorch/__init__.py")
                with self.assertRaises(ValueError):
                    artifacts.verify_installed_files(inventory, root, root / "spiraltorch/__init__.py", "0.4.33")

    def test_record_rewrite_allowed_but_version_shadow_and_loaded_origin_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            inventory = artifacts.wheel_inventory(wheel(temporary))
            install(root, files())
            (root / "spiraltorch-0.4.33.dist-info/RECORD").write_bytes(b"installer rewritten")
            origin = root / "spiraltorch/__init__.py"
            artifacts.verify_installed_files(inventory, root, origin, "0.4.33", [origin])
            for selected_origin, version, loaded in ((origin, "0.4.32", []),
                                                     (root / "spiraltorch/helper.py", "0.4.33", []),
                                                     (origin, "0.4.33", [Path(__file__)])):
                with self.assertRaises(ValueError):
                    artifacts.verify_installed_files(inventory, root, selected_origin, version, loaded)

    def test_unsafe_missing_and_mismatched_wheel_entries_reject(self):
        changes = {
            "traversal": lambda f: f.update({"../escape": b"x"}),
            "absolute": lambda f: f.update({"/escape": b"x"}),
            "separator": lambda f: f.update({"spiraltorch\\escape": b"x"}),
            "double_separator": lambda f: f.update({"spiraltorch//escape": b"x"}),
            "cached": lambda f: f.update({"spiraltorch/helper.pyc": b"x"}),
            "another_distribution": lambda f: f.update({"other/data": b"x"}),
            "missing_native": lambda f: f.pop("spiraltorch/spiraltorch.abi3.so"),
            "second_native": lambda f: f.update({"spiraltorch/spiraltorch.extra.so": b"x"}),
            "missing_init": lambda f: f.pop("spiraltorch/__init__.py"),
            "version": lambda f: f.update({"spiraltorch-0.4.33.dist-info/METADATA": b"Name: spiraltorch\nVersion: 0.4.34\n"}),
            "abi": lambda f: f.update({"spiraltorch-0.4.33.dist-info/WHEEL": b"Tag: cp312-abi3-macosx_14_0_universal2\n"}),
            "tag": lambda f: f.update({"spiraltorch-0.4.33.dist-info/WHEEL": b"Tag: cp38-abi3-win_amd64\n"}),
        }
        for defect, mutate in changes.items():
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                contents = files()
                mutate(contents)
                with self.assertRaises(ValueError):
                    artifacts.wheel_inventory(wheel(temporary, contents))

    def test_duplicate_symbolic_member_and_size_bounds_reject(self):
        for defect in ("duplicate", "symlink", "size"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                path = wheel(temporary)
                if defect != "size":
                    with zipfile.ZipFile(path, "a") as archive, warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        entry = zipfile.ZipInfo("spiraltorch/helper.py" if defect == "duplicate" else "spiraltorch/link")
                        if defect == "symlink":
                            entry.create_system = 3
                            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                        archive.writestr(entry, b"x")
                with patch.object(artifacts, "MAX_MEMBER", 10 if defect == "size" else artifacts.MAX_MEMBER):
                    with self.assertRaises(ValueError):
                        artifacts.wheel_inventory(path)


if __name__ == "__main__":
    unittest.main()
