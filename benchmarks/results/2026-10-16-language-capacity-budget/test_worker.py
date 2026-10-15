import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("worker",Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_input_is_exact_verified_snapshot_not_later_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"request.json"
            path.write_bytes(b"original")
            value = worker.FrozenInput(path.read_bytes(),worker.digest(b"original"))
            path.write_bytes(b"changed")
            self.assertEqual(value.read_bytes(),b"original")
            with self.assertRaises(ValueError):
                worker.FrozenInput(path.read_bytes(),worker.digest(b"original"))

    def test_module_executes_only_verified_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"runner.py"
            raw = b"def probe():\n    return 42\n"
            path.write_bytes(raw)
            module = worker.verified_module(path,worker.digest(raw))
            path.write_bytes(b"raise AssertionError('must not execute changed source')\n")
            self.assertEqual(module.probe(),42)
            with self.assertRaises(ValueError):
                worker.verified_module(path,worker.digest(raw))

    def test_native_extension_is_checked_inside_worker_before_dispatch(self):
        repo = Path(__file__).resolve().parents[3]
        here = Path(worker.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root/"w16-seed11"
            case.mkdir()
            (case/"request.json").write_bytes(b"request")
            extension = root/"native.so"
            extension.write_bytes(b"extension-a")
            manifest = dict(sources=[dict(path=str(here.relative_to(repo)),sha256=worker.digest(here.read_bytes())),
                                    dict(path=str(here.with_name("launch.py").relative_to(repo)),sha256="launch"),
                                    dict(path="tools/run_byte_corpus_segments.py",sha256="runner")],
                            entries=[dict(name=case.name,request_sha256=worker.digest(b"request"))],
                            native_extension_sha256=worker.digest(b"extension-a"))
            raw = json.dumps(manifest).encode()
            (root/"prepared.json").write_bytes(raw)
            launch = SimpleNamespace(validated_inputs=Mock(),names=lambda:[case.name],STOPS=(0,1024),write=Mock())
            consumed = []
            runner = SimpleNamespace(run=lambda request,*args,**kwargs:consumed.append(request.read_bytes()))
            st = SimpleNamespace(_rs=SimpleNamespace(__file__=str(extension)),nn=SimpleNamespace(ByteCorpusStudy=object()))
            with patch.object(worker,"verified_module",side_effect=lambda path,expected:launch if path.name == "launch.py" else runner), \
                 patch.dict(worker.sys.modules,{"spiraltorch":st}):
                worker.execute(repo,root,case.name,"native",worker.digest(raw))
                self.assertEqual(consumed,[b"request"])
                extension.write_bytes(b"extension-b")
                with self.assertRaisesRegex(ValueError,"extension"):
                    worker.execute(repo,root,case.name,"native",worker.digest(raw))
                self.assertEqual(consumed,[b"request"])

    def test_nonisolated_worker_rejects_before_opening_inputs(self):
        with patch.object(worker.sys,"flags",SimpleNamespace(isolated=False)):
            with self.assertRaisesRegex(ValueError,"isolated"):
                worker.execute(Path("missing"),Path("missing"),"missing","native","missing")


if __name__ == "__main__":
    unittest.main()
