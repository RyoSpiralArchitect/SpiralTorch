"""Model-free integrity checks for the native QKV file-run output protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).parents[1] / "spiraltorch" / "native_qkv_runs.py"
spec = importlib.util.spec_from_file_location("native_qkv_runs_under_test", SOURCE)
runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runs)


def encode(header, payload=b""):
    data = json.dumps(header, separators=(",", ":")).encode()
    return len(data).to_bytes(4, "little") + data + payload


def pin(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


class QkvRunVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {"context_length": 4, "batch_windows": 1}
        self.data_id = "sha256:" + "a" * 64
        self.result = {
            "schema": "spiraltorch.native_qkv_run.v1",
            "data_id": self.data_id,
            "config": self.config,
            "initial_update": 0,
            "updates": 1,
            "total_updates": 1,
            "exhausted": True,
            "checkpoint_index": "checkpoints.jsonl",
            "step_log": "steps.jsonl",
        }
        index = []
        for update in (0, 1):
            inner = encode({"schema": "spiraltorch.qkv_adamw.v1", "steps": update, "layer": 0}, b"weight")
            outer = encode({
                "schema": "spiraltorch.native_causal_lm_qkv_trainer.v1",
                "data_id": self.data_id,
                "config": self.config,
                "updates": update,
                "initial_steps": 0,
                "optimizer_id": pin(inner),
            }, inner)
            name = f"checkpoint-{update:09d}.bin"
            (self.root / name).write_bytes(outer)
            index.append({"file": name, "checkpoint_id": pin(outer), "updates": update})
        self.result["final_checkpoint"] = index[-1]
        self.result["qkv"] = {
            "file": "qkv.bin", "checkpoint_id": pin(inner),
            "optimizer_steps": 1, "layer": 0,
        }
        (self.root / "qkv.bin").write_bytes(inner)
        (self.root / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
        (self.root / "steps.jsonl").write_text('{"update":1}\n')
        self.write_result()

    def write_result(self):
        (self.root / "result.json").write_text(json.dumps(self.result) + "\n")

    def test_complete_run(self):
        self.assertEqual(runs.verify_qkv_run(self.root), self.result)

    def test_rejects_missing_truncated_or_extra_log(self):
        path = self.root / "steps.jsonl"
        for data in (b"", b'{"update":1}', b'{"update":1}\n{"update":2}\n'):
            with self.subTest(data=data):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    runs.verify_qkv_run(self.root)

    def test_rejects_incomplete_marker_or_index(self):
        marker = self.root / "result.json"
        marker.write_bytes(marker.read_bytes().rstrip(b"\n"))
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.write_result()
        (self.root / "checkpoints.jsonl").write_bytes(b"")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_result_or_file_changes(self):
        path = self.root / "qkv.bin"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        path.unlink()
        with self.assertRaises(OSError):
            runs.verify_qkv_run(self.root)

    def test_rejects_resealed_export_that_differs_from_final_state(self):
        path = self.root / "qkv.bin"
        data = path.read_bytes() + b"different"
        path.write_bytes(data)
        self.result["qkv"]["checkpoint_id"] = pin(data)
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_resealed_inconsistent_checkpoint(self):
        path = self.root / "checkpoint-000000001.bin"
        data = bytearray(path.read_bytes())
        length = int.from_bytes(data[:4], "little")
        header = json.loads(data[4:4 + length])
        header["data_id"] = "sha256:" + "b" * 64
        data = encode(header, data[4 + length:])
        path.write_bytes(data)
        self.result["final_checkpoint"]["checkpoint_id"] = pin(data)
        index = [json.loads(row) for row in (self.root / "checkpoints.jsonl").read_text().splitlines()]
        index[-1] = self.result["final_checkpoint"]
        (self.root / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_path_substitution_and_duplicate_fields(self):
        self.result["qkv"]["file"] = "../other.bin"
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.result["qkv"]["file"] = "qkv.bin"
        self.write_result()
        (self.root / "steps.jsonl").write_text('{"update":1,"update":1}\n')
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_nonfinite_result_and_symlinked_files(self):
        result = self.root / "result.json"
        result.write_text(result.read_text().replace('"total_updates": 1', '"total_updates": NaN'))
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.write_result()
        qkv = self.root / "qkv.bin"
        target = self.root / "export-copy.bin"
        qkv.rename(target)
        try:
            qkv.symlink_to(target)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)


if __name__ == "__main__":
    unittest.main()
