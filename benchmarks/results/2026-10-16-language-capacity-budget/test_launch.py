import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("launch", Path(__file__).with_name("launch.py"))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
REPO = Path(__file__).resolve().parents[3]
protocol = launch.load(REPO / "tools/byte_corpus_study.py")


def fake_child(args, **kwargs):
    root = Path(args[args.index("--root")+1])
    name = args[args.index("--name")+1]
    engine = args[args.index("--engine")+1]
    (root/name).mkdir(exist_ok=True)
    launch.write(root/name/(engine+"-child.json"), dict(
        prepared_sha256=launch.digest((root/"prepared.json").read_bytes()),request_sha256="request",
        native_extension_sha256=None,engine=engine,isolated=True,input_bytes_bound=True))


def fixture(width=16, seed=11):
    config = dict(batch=2, steps=32, width=width, hidden=width*2, heads=2,
                  blocks=[False], geometry_cols=4, curvature=-.75)
    documents = [[list(range(256))*16], [list(reversed(range(256)))*16]]
    candidates = protocol.windows(documents[0], 32)
    rng, chosen = random.Random(20261009), []
    while len(chosen) < 2048:
        group = candidates.copy()
        rng.shuffle(group)
        chosen.extend(group)
    valid = protocol.windows(documents[1], 32)
    selected = [list(valid[i*len(valid)//64]) for i in range(64)]
    plain, geometry = protocol.parameters(config, seed)
    cases = [dict(name=f"seed{seed}_ordinary", seed=seed, geometry=False, parameters=plain, geometry_update="train"),
             dict(name=f"seed{seed}_geometry", seed=seed, geometry=True, parameters=geometry, geometry_update="train"),
             dict(name=f"seed{seed}_geometry_frozen", seed=seed, geometry=True, parameters=copy.deepcopy(geometry), geometry_update="frozen")]
    request = dict(schema=protocol.REQUEST_V9, design_schema=protocol.REQUEST_V2,
                   optimizer=launch.RECIPE.copy(), config=config, cases=cases,
                   train_documents=documents[0], validation_documents=documents[1],
                   train_batches=[[list(s) for s in chosen[i:i+2]] for i in range(0,2048,2)],
                   validation_batches=[selected[i:i+2] for i in range(0,64,2)],
                   checkpoint_every=64, rate=protocol.f32(.001))
    return request, documents


class PreflightTests(unittest.TestCase):
    def test_all_declared_widths_and_seeds(self):
        for width in launch.WIDTHS:
            for seed in launch.SEEDS:
                request, documents = fixture(width, seed)
                launch.validate_request(protocol, request, width, seed, documents)

    def test_rejects_changed_data_budget_policy_and_initials(self):
        original, documents = fixture()
        for corrupt in (
            lambda r: r["train_documents"][0].__setitem__(0, 99),
            lambda r: r["train_batches"].pop(),
            lambda r: r["train_batches"].reverse(),
            lambda r: r["validation_batches"].reverse(),
            lambda r: r["optimizer"].__setitem__("weight_decay", 0.),
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][0]["parameters"][0]["values"].__setitem__(0, 0.),
            lambda r: r["cases"][2].__setitem__("geometry_update", "train"),
        ):
            request = copy.deepcopy(original)
            corrupt(request)
            with self.assertRaises(ValueError):
                launch.validate_request(protocol, request, 16, 11, documents)

    def test_canonical_mapping_ignores_order_but_preserves_signed_zero(self):
        self.assertEqual(launch.canonical(dict(a=1, b=2)), launch.canonical(dict(b=2, a=1)))
        self.assertNotEqual(launch.canonical([-0.0]), launch.canonical([0.0]))

    def test_exclusive_matrix_claim_prevents_repeated_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared.json").write_text("{}")
            manifest = dict(entries=[dict(name=name,request_sha256="request") for name in launch.names()])
            with patch.object(launch, "validated_inputs", return_value=manifest), patch.object(launch.subprocess, "run",side_effect=fake_child) as run:
                launch.execute(REPO, root, reference=True)
                self.assertEqual(run.call_count, 6)
                with self.assertRaises(FileExistsError):
                    launch.execute(REPO, root, reference=True)
                self.assertEqual(run.call_count, 6)

    def test_subprocess_failure_keeps_claim_and_no_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared.json").write_text("{}")
            manifest = dict(entries=[dict(name=name) for name in launch.names()])
            with patch.object(launch, "validated_inputs", return_value=manifest), patch.object(launch.subprocess, "run", side_effect=RuntimeError("failed")) as run:
                with self.assertRaises(RuntimeError):
                    launch.execute(REPO, root, reference=True)
                self.assertTrue((root / "reference-launch.json").exists())
                self.assertFalse((root / "reference-completed.json").exists())
                self.assertEqual(run.call_count, 1)

    def test_relative_root_is_resolved_before_child_cwd_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent/"data"
            root.mkdir()
            (root/"prepared.json").write_text("{}")
            manifest = dict(entries=[dict(name=name,request_sha256="request") for name in launch.names()])
            previous = Path.cwd()
            try:
                os.chdir(parent)
                with patch.object(launch,"validated_inputs",return_value=manifest) as validate, \
                     patch.object(launch.subprocess,"run",side_effect=fake_child) as run:
                    launch.execute(REPO,Path("data"),True)
                    for call in validate.call_args_list:
                        self.assertEqual(call.args,(REPO.resolve(),root.resolve()))
                    for call in run.call_args_list:
                        args = call.args[0]
                        self.assertEqual(args[args.index("--root")+1],str(root.resolve()))
            finally:
                os.chdir(previous)

    def test_source_drift_between_conditions_stops_before_second_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"prepared.json").write_text("{}")
            manifest = dict(entries=[dict(name=name,request_sha256="request") for name in launch.names()])
            changed = False
            def validate(*args):
                if changed:
                    raise ValueError("runner/source drift")
                return manifest
            def first(args,**kwargs):
                nonlocal changed
                fake_child(args,**kwargs)
                changed = True
            with patch.object(launch,"validated_inputs",side_effect=validate), \
                 patch.object(launch.subprocess,"run",side_effect=first) as run:
                with self.assertRaisesRegex(ValueError,"drift"):
                    launch.execute(REPO,root,True)
                self.assertEqual(run.call_count,1)
                self.assertTrue((root/"reference-launch.json").exists())
                self.assertFalse((root/"reference-completed.json").exists())

    def test_archived_launcher_is_only_allowed_for_read_only_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"PLAN.md").write_bytes(Path(launch.__file__).with_name("PLAN.md").read_bytes())
            (root/"launch-source.py").write_bytes(b"original launcher")
            entries = []
            for name in launch.names():
                (root/name).mkdir()
                (root/name/"request.json").write_bytes(b"request")
                entries.append(dict(name=name,request_sha256=launch.digest(b"request")))
            manifest = dict(entries=entries,stops=list(launch.STOPS),
                            plan_sha256=launch.digest((root/"PLAN.md").read_bytes()),
                            sources=[dict(path=str(Path(launch.__file__).relative_to(REPO)),
                                          sha256=launch.digest(b"original launcher"))])
            (root/"prepared.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"source drift"):
                launch.validated_inputs(REPO,root)
            self.assertEqual(launch.validated_inputs(REPO,root,allow_recorded_launcher=True),manifest)
            (root/"launch-source.py").write_bytes(b"changed archive")
            with self.assertRaisesRegex(ValueError,"source drift"):
                launch.validated_inputs(REPO,root,allow_recorded_launcher=True)


if __name__ == "__main__":
    unittest.main()
