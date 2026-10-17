import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location("run_once",Path(__file__).with_name("run_once.py"))
RUN=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ClaimTests(unittest.TestCase):
    def test_repeat_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            RUN.claim(path,"request","runner")
            with self.assertRaises(FileExistsError):
                RUN.claim(path,"request","runner")
            self.assertTrue(json.loads((path/"native-run-claim.json").read_text())["before_gpu_launch"])

    def test_existing_output_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            (path/"native-128.json").touch()
            with self.assertRaisesRegex(ValueError,"no training started"):
                RUN.claim(path,"request","runner")
            with self.assertRaises(FileExistsError):
                RUN.claim(path,"request","runner")

    def test_adoption_is_not_a_prelaunch_claim(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)
            RUN.claim(path,"request","runner",123)
            state=json.loads((path/"native-run-claim.json").read_text())
            self.assertFalse(state["before_gpu_launch"])
            self.assertEqual(state["adopted_existing_pid"],123)


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix="pilot with spaces ")
        self.addCleanup(self.directory.cleanup)
        self.output=Path(self.directory.name)
        self.runner=self.output/"run_budget.py"
        self.runner.write_text("# Synthetic pinned runner; never executed.\n")
        (self.output/"request.json").write_text("{}\n")
        (self.output/"PLAN.md").write_text("synthetic plan\n")
        (self.output/"prepared.json").write_text(json.dumps(dict(
            request_sha256=RUN.sha((self.output/"request.json").read_bytes()),
            runner_sha256=RUN.sha(self.runner.read_bytes()),
            plan_sha256=RUN.sha((self.output/"PLAN.md").read_bytes()))))
        self.repo=self.output/"repository"
        self.historical=self.output/"historical request.json"
        self.command=RUN.runner_command(self.runner,self.repo,self.historical,self.output)
        self.argv=["run_once.py","--runner",str(self.runner),"--repo",str(self.repo),
                   "--historical",str(self.historical),"--output",str(self.output)]
        self.claim=self.output/"native-run-claim.json"

    def test_exact_argv_preserves_spaces(self):
        RUN.validate_process(self.command,self.command)

    def test_runner_and_output_prefixes_are_rejected(self):
        wrong=self.command.copy()
        wrong[3]+=".backup"
        wrong[-1]+="-other"
        with self.assertRaisesRegex(ValueError,"arguments"):
            RUN.validate_process(wrong,self.command)

    def test_shared_base_interpreter_does_not_equate_virtualenvs(self):
        first=self.output/"first-venv-python"
        second=self.output/"second-venv-python"
        first.symlink_to(sys.executable)
        second.symlink_to(sys.executable)
        self.assertEqual(first.resolve(),second.resolve())
        expected=self.command.copy()
        expected[0]=str(first)
        actual=expected.copy()
        actual[0]=str(second)
        with self.assertRaisesRegex(ValueError,"arguments"):
            RUN.validate_process(actual,expected)

    def test_changed_flags_paths_and_extra_arguments_are_rejected(self):
        variants=[[],[" ".join(self.command)],self.command[:-1],self.command+["extra"]]
        for index in (0,1,2,3,4,5,6,7,8,9,10):
            changed=self.command.copy()
            changed[index]+="-other"
            variants.append(changed)
        for changed in variants:
            with self.subTest(command=changed):
                with self.assertRaises(ValueError):
                    RUN.validate_process(changed,self.command)

    def test_missing_optional_inspector_fails_closed(self):
        with mock.patch.dict(sys.modules,{"psutil":None}):
            with self.assertRaisesRegex(RuntimeError,"requires psutil"):
                RUN.process_arguments(123)
        self.assertFalse(self.claim.exists())

    def inspector(self, launcher, image="shared-python-image"):
        command=self.command.copy()
        command[0]="/framework/Python"
        target=SimpleNamespace(cmdline=lambda:command,exe=lambda:image,
                               environ=mock.Mock(return_value={"__PYVENV_LAUNCHER__":launcher}))
        current=SimpleNamespace(exe=lambda:"shared-python-image")
        return SimpleNamespace(Error=RuntimeError,Process=lambda pid=None:current if pid is None else target),target

    def test_framework_launcher_keeps_literal_invocation_path(self):
        inspector,target=self.inspector(self.command[0])
        with mock.patch.dict(sys.modules,{"psutil":inspector}),mock.patch.object(sys,"platform","darwin"):
            actual=RUN.process_arguments(123)
        RUN.validate_process(actual,self.command)
        target.environ.assert_called_once()

    def test_framework_launcher_cannot_substitute_another_virtualenv(self):
        for launcher in (None,"/other-venv/bin/python"):
            inspector,_=self.inspector(launcher)
            with mock.patch.dict(sys.modules,{"psutil":inspector}),mock.patch.object(sys,"platform","darwin"):
                actual=RUN.process_arguments(123)
            with self.assertRaisesRegex(ValueError,"arguments"):
                RUN.validate_process(actual,self.command)

    def test_nonmatching_process_image_does_not_read_environment(self):
        inspector,target=self.inspector(self.command[0],"another-python-image")
        with mock.patch.dict(sys.modules,{"psutil":inspector}),mock.patch.object(sys,"platform","darwin"):
            with self.assertRaisesRegex(ValueError,"interpreter"):
                RUN.process_arguments(123)
        target.environ.assert_not_called()

    def test_wrong_process_does_not_create_claim_or_launch(self):
        wrong=self.command.copy()
        wrong[-1]+="-other"
        with mock.patch.object(sys,"argv",self.argv+["--adopt-pid","123"]), \
                mock.patch.object(RUN,"process_arguments",return_value=wrong), \
                mock.patch.object(RUN.subprocess,"run") as launch:
            with self.assertRaisesRegex(ValueError,"arguments"):
                RUN.main()
        self.assertFalse(self.claim.exists())
        launch.assert_not_called()

    def test_uninspectable_process_does_not_create_claim_or_launch(self):
        with mock.patch.object(sys,"argv",self.argv+["--adopt-pid","123"]), \
                mock.patch.object(RUN,"process_arguments",side_effect=RuntimeError("cannot inspect")), \
                mock.patch.object(RUN.subprocess,"run") as launch:
            with self.assertRaisesRegex(RuntimeError,"cannot inspect"):
                RUN.main()
        self.assertFalse(self.claim.exists())
        launch.assert_not_called()

    def test_correct_adoption_does_not_launch(self):
        with mock.patch.object(sys,"argv",self.argv+["--adopt-pid","123"]), \
                mock.patch.object(RUN,"process_arguments",return_value=self.command), \
                mock.patch.object(RUN.subprocess,"run") as launch, \
                mock.patch.object(sys,"stdout",io.StringIO()):
            RUN.main()
        state=json.loads(self.claim.read_text())
        self.assertFalse(state["before_gpu_launch"])
        self.assertEqual(state["adopted_existing_pid"],123)
        launch.assert_not_called()

    def test_fresh_launch_uses_the_same_canonical_command(self):
        with mock.patch.object(sys,"argv",self.argv), \
                mock.patch.object(RUN.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as launch:
            with self.assertRaises(SystemExit) as stopped:
                RUN.main()
        self.assertEqual(stopped.exception.code,0)
        launch.assert_called_once_with(self.command,check=False)
        self.assertTrue(json.loads(self.claim.read_text())["before_gpu_launch"])


if __name__=="__main__":
    unittest.main()
