import contextlib
import json
import os
import shutil
import textwrap
import time
import unittest
import uuid
from pathlib import Path

import spiraltorch as st  # noqa: E402

_TEST_TMP_ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def _temp_dir(prefix: str) -> str:
    """Create a short-lived writable directory for tests.

    Python's `tempfile` uses mode=0o700 for directories; on Windows with Python
    3.14 this can yield ACLs that deny file creation in this environment.
    """

    path = _TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    os.mkdir(path)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class SpiralTorchSmokeTest(unittest.TestCase):
    def test_plugin_tensor_op(self) -> None:
        events: list[dict] = []
        sub_id = st.plugin.subscribe("TensorOp", lambda e: events.append(e))
        a = st.Tensor.rand(2, 2, seed=1)
        b = st.Tensor.rand(2, 2, seed=2)
        _ = a.add(b)
        st.plugin.unsubscribe("TensorOp", sub_id)
        self.assertTrue(events)
        self.assertEqual(events[-1]["type"], "TensorOp")

    def test_module_trainer_smoke(self) -> None:
        trainer = st.nn.ModuleTrainer(
            backend="cpu",
            curvature=-1.0,
            hyper_learning_rate=1e-2,
            fallback_learning_rate=1e-2,
        )
        schedule = trainer.roundtable(
            2,
            1,
            st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
        )
        model = st.nn.Sequential()
        model.add(st.nn.Linear("l1", 2, 1))
        model.attach_hypergrad(curvature=-1.0, learning_rate=1e-2)
        loss = st.nn.MeanSquaredError()
        x = st.Tensor.rand(2, 2, seed=3)
        y = st.Tensor.rand(2, 1, seed=4)
        stats = trainer.train_epoch(model, loss, [(x, y)], schedule)
        self.assertEqual(stats.batches, 1)

    def test_ops_register_execute(self) -> None:
        @st.ops.signature(2, 1)
        def forward(inputs):
            return [inputs[0].add(inputs[1])]

        st.ops.unregister("py_add")
        st.ops.register("py_add", forward, description="python add operator", backends=["python"])
        a = st.Tensor.rand(1, 1, seed=5)
        b = st.Tensor.rand(1, 1, seed=6)
        out = st.ops.execute("py_add", a, b, return_single=True)
        desc = st.ops.describe("py_add")
        self.assertIsInstance(out, st.Tensor)
        self.assertIn("py_add", desc)

    def test_plugin_record_jsonl(self) -> None:
        with _temp_dir("tmp_jsonl") as tmp:
            path = f"{tmp}/events.jsonl"
            with st.plugin.record(path, event_types="TensorOp"):
                a = st.Tensor.rand(1, 1, seed=7)
                b = st.Tensor.rand(1, 1, seed=8)
                _ = a.add(b)
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line]
        self.assertTrue(lines)
        event = json.loads(lines[-1])
        self.assertEqual(event["type"], "TensorOp")

    def test_python_plugin_registry(self) -> None:
        class DemoPlugin:
            def __init__(self) -> None:
                self.events: list[dict] = []
                self.loaded = False
                self.unloaded = False
                self.service: dict[str, object] = {"ok": True}

            def metadata(self) -> dict:
                return {
                    "id": "demo_py_plugin",
                    "version": "0.0.1",
                    "name": "Demo Python Plugin",
                    "capabilities": ["Telemetry", "Custom:python-demo"],
                    "metadata": {"lang": "py"},
                }

            def on_load(self) -> None:
                self.loaded = True
                st.plugin.set_config("demo_py_plugin.key", "demo-value")
                st.plugin.register_service("demo_py_plugin.service", self.service)

            def on_unload(self) -> None:
                self.unloaded = True

            def on_event(self, event: dict) -> None:
                self.events.append(event)

        plugin = DemoPlugin()
        plugin_id = st.plugin.register_python_plugin(plugin)
        self.assertEqual(plugin_id, "demo_py_plugin")
        self.assertTrue(plugin.loaded)
        self.assertIn(plugin_id, st.plugin.list_plugins())

        meta = st.plugin.plugin_metadata(plugin_id)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["id"], plugin_id)
        self.assertEqual(meta["version"], "0.0.1")
        self.assertIn("Telemetry", meta.get("capabilities", []))

        hits = st.plugin.find_by_capability("Telemetry")
        self.assertIn(plugin_id, hits)

        self.assertEqual(st.plugin.get_config("demo_py_plugin.key"), "demo-value")
        self.assertIn("demo_py_plugin.service", st.plugin.list_services())
        service = st.plugin.get_service("demo_py_plugin.service")
        self.assertIs(service, plugin.service)

        plugin.events.clear()
        st.plugin.publish("DemoEvent", {"x": 1})
        self.assertTrue(plugin.events)
        self.assertEqual(plugin.events[-1]["type"], "DemoEvent")
        self.assertEqual(plugin.events[-1]["payload"]["x"], 1)

        st.plugin.unregister_plugin(plugin_id)
        self.assertTrue(plugin.unloaded)
        self.assertNotIn(plugin_id, st.plugin.list_plugins())

    def test_python_plugin_load_path(self) -> None:
        plugin_id = f"demo_path_plugin_{uuid.uuid4().hex}"
        plugin_source = textwrap.dedent(
            f"""
            import spiraltorch as st

            class PathPlugin:
                def __init__(self) -> None:
                    self.events = []

                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.1",
                        "capabilities": ["Telemetry"],
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{plugin_id}.instance", self)

                def on_event(self, event: dict) -> None:
                    self.events.append(event)

            plugin = PathPlugin()
            """
        )

        with _temp_dir("tmp_path") as tmp:
            plugin_path = f"{tmp}/path_plugin.py"
            with open(plugin_path, "w", encoding="utf-8") as handle:
                handle.write(plugin_source)

            loaded = st.plugin.load_path(tmp, recursive=False, strict=True)
            self.assertIn(plugin_id, loaded)

            meta = st.plugin.plugin_metadata(plugin_id)
            self.assertIsInstance(meta, dict)
            self.assertIsInstance(meta.get("metadata"), dict)
            extra = meta["metadata"]
            self.assertEqual(extra.get("spiraltorch.source"), "path")
            self.assertEqual(Path(extra["spiraltorch.source_path"]).resolve(), Path(plugin_path).resolve())
            self.assertTrue(str(extra["spiraltorch.source_module"]).startswith("spiraltorch_path_plugin_"))

            plugin = st.plugin.get_service(f"{plugin_id}.instance")
            self.assertIsNotNone(plugin)

            st.plugin.publish("FromPath", {"v": 123})
            self.assertTrue(plugin.events)
            self.assertEqual(plugin.events[-1]["type"], "FromPath")
            self.assertEqual(plugin.events[-1]["payload"]["v"], 123)

            st.plugin.unregister_plugin(plugin_id)
            self.assertNotIn(plugin_id, st.plugin.list_plugins())

    def test_python_plugin_unload_path(self) -> None:
        plugin_id = f"demo_unload_path_{uuid.uuid4().hex}"
        plugin_source = textwrap.dedent(
            f"""
            class DemoUnloadPlugin:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.1",
                    }}

            plugin = DemoUnloadPlugin()
            """
        )

        try:
            with _temp_dir("tmp_unload") as tmp:
                plugin_path = f"{tmp}/unload_plugin.py"
                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source)

                loaded = st.plugin.load_path(plugin_path, recursive=False, strict=True)
                self.assertIn(plugin_id, loaded)
                self.assertIn(plugin_id, st.plugin.list_plugins())

                unloaded = st.plugin.unload_path(plugin_path, strict=True)
                self.assertIn(plugin_id, unloaded)
                self.assertNotIn(plugin_id, st.plugin.list_plugins())
        finally:
            try:
                st.plugin.unregister_plugin(plugin_id)
            except Exception:
                pass

    def test_python_plugin_load_entrypoints_dependency_order(self) -> None:
        import importlib.metadata

        plugin_a_id = f"demo_entry_a_{uuid.uuid4().hex}"
        plugin_b_id = f"demo_entry_b_{uuid.uuid4().hex}"
        service_a = f"{plugin_a_id}.value"
        service_b = f"{plugin_b_id}.seen"

        class _EntryPoint:
            def __init__(self, name: str, value: str, plugin):
                self.name = name
                self.value = value
                self._plugin = plugin

            def load(self):
                return self._plugin

        def _make_plugins(version: str):
            class PluginA:
                def __init__(self):
                    self.unloaded = False

                def metadata(self) -> dict:
                    return {"id": plugin_a_id, "version": version}

                def on_load(self) -> None:
                    st.plugin.register_service(service_a, version)
                    st.plugin.register_service(f"{plugin_a_id}.instance", self)

                def on_unload(self) -> None:
                    self.unloaded = True

            class PluginB:
                def __init__(self):
                    self.unloaded = False

                def metadata(self) -> dict:
                    return {
                        "id": plugin_b_id,
                        "version": version,
                        "dependencies": {plugin_a_id: ">=0"},
                    }

                def on_load(self) -> None:
                    st.plugin.register_service(service_b, st.plugin.get_service(service_a))
                    st.plugin.register_service(f"{plugin_b_id}.instance", self)

                def on_unload(self) -> None:
                    self.unloaded = True

            return PluginA(), PluginB()

        original = importlib.metadata.entry_points
        state = {"version": "0.0.1"}

        def fake_entry_points():
            plugin_a, plugin_b = _make_plugins(state["version"])
            group = "spiraltorch.plugins"
            return {
                group: [
                    _EntryPoint("demo_b", f"demo_b={state['version']}", plugin_b),
                    _EntryPoint("demo_a", f"demo_a={state['version']}", plugin_a),
                ]
            }

        try:
            importlib.metadata.entry_points = fake_entry_points  # type: ignore[assignment]

            loaded = st.plugin.load_entrypoints(replace=True)
            self.assertIn(plugin_a_id, loaded)
            self.assertIn(plugin_b_id, loaded)
            self.assertLess(loaded.index(plugin_a_id), loaded.index(plugin_b_id))
            self.assertEqual(st.plugin.get_service(service_b), "0.0.1")

            meta = st.plugin.plugin_metadata(plugin_a_id)
            self.assertIsInstance(meta, dict)
            extra = meta.get("metadata", {})
            self.assertEqual(extra.get("spiraltorch.source"), "entrypoint")
            self.assertEqual(extra.get("spiraltorch.entrypoint_group"), "spiraltorch.plugins")

            old_a = st.plugin.get_service(f"{plugin_a_id}.instance")
            old_b = st.plugin.get_service(f"{plugin_b_id}.instance")
            self.assertIsNotNone(old_a)
            self.assertIsNotNone(old_b)

            state["version"] = "0.0.2"
            loaded2 = st.plugin.load_entrypoints(replace=True)
            self.assertIn(plugin_a_id, loaded2)
            self.assertIn(plugin_b_id, loaded2)
            self.assertEqual(st.plugin.get_service(service_b), "0.0.2")

            new_a = st.plugin.get_service(f"{plugin_a_id}.instance")
            new_b = st.plugin.get_service(f"{plugin_b_id}.instance")
            self.assertIsNot(old_a, new_a)
            self.assertIsNot(old_b, new_b)
            self.assertTrue(getattr(old_a, "unloaded", False))
            self.assertTrue(getattr(old_b, "unloaded", False))
        finally:
            importlib.metadata.entry_points = original  # type: ignore[assignment]
            for plugin_id in (plugin_b_id, plugin_a_id):
                try:
                    st.plugin.unregister_plugin(plugin_id)
                except Exception:
                    pass

    def test_python_plugin_unload_entrypoints(self) -> None:
        import importlib.metadata

        plugin_a_id = f"demo_unload_entry_a_{uuid.uuid4().hex}"
        plugin_b_id = f"demo_unload_entry_b_{uuid.uuid4().hex}"
        service_a = f"{plugin_a_id}.instance"
        service_b = f"{plugin_b_id}.instance"

        class _EntryPoint:
            def __init__(self, name: str, value: str, plugin):
                self.name = name
                self.value = value
                self._plugin = plugin

            def load(self):
                return self._plugin

        class PluginA:
            def __init__(self):
                self.unloaded = False

            def metadata(self) -> dict:
                return {"id": plugin_a_id, "version": "0.0.1"}

            def on_load(self) -> None:
                st.plugin.register_service(service_a, self)

            def on_unload(self) -> None:
                self.unloaded = True

        class PluginB:
            def __init__(self):
                self.unloaded = False

            def metadata(self) -> dict:
                return {
                    "id": plugin_b_id,
                    "version": "0.0.1",
                    "dependencies": {plugin_a_id: ">=0"},
                }

            def on_load(self) -> None:
                st.plugin.register_service(service_b, self)

            def on_unload(self) -> None:
                self.unloaded = True

        original = importlib.metadata.entry_points

        def fake_entry_points():
            group = "spiraltorch.plugins"
            return {
                group: [
                    _EntryPoint("demo_b", "demo_b=0.0.1", PluginB()),
                    _EntryPoint("demo_a", "demo_a=0.0.1", PluginA()),
                ]
            }

        try:
            importlib.metadata.entry_points = fake_entry_points  # type: ignore[assignment]
            loaded = st.plugin.load_entrypoints(replace=True)
            self.assertIn(plugin_a_id, loaded)
            self.assertIn(plugin_b_id, loaded)

            a = st.plugin.get_service(service_a)
            b = st.plugin.get_service(service_b)
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)

            unloaded = st.plugin.unload_entrypoints(strict=True)
            self.assertIn(plugin_a_id, unloaded)
            self.assertIn(plugin_b_id, unloaded)
            self.assertNotIn(plugin_a_id, st.plugin.list_plugins())
            self.assertNotIn(plugin_b_id, st.plugin.list_plugins())

            self.assertLess(unloaded.index(plugin_b_id), unloaded.index(plugin_a_id))
            self.assertTrue(getattr(a, "unloaded", False))
            self.assertTrue(getattr(b, "unloaded", False))
        finally:
            importlib.metadata.entry_points = original  # type: ignore[assignment]
            for plugin_id in (plugin_b_id, plugin_a_id):
                try:
                    st.plugin.unregister_plugin(plugin_id)
                except Exception:
                    pass

    def test_python_plugin_reload_path(self) -> None:
        plugin_id = f"demo_reload_path_plugin_{uuid.uuid4().hex}"
        service_name = f"{plugin_id}.instance"

        plugin_source_v1 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class ReloadPlugin:
                def __init__(self) -> None:
                    self.events = []
                    self.unloaded = False

                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.1",
                        "capabilities": ["Telemetry"],
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_name}", self)

                def on_unload(self) -> None:
                    self.unloaded = True

                def on_event(self, event: dict) -> None:
                    self.events.append({{
                        "marker": "v1",
                        "type": event.get("type"),
                        "payload": event.get("payload"),
                    }})

            plugin = ReloadPlugin()
            """
        )

        plugin_source_v2 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class ReloadPlugin:
                def __init__(self) -> None:
                    self.events = []
                    self.unloaded = False

                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.2",
                        "capabilities": ["Telemetry"],
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_name}", self)

                def on_unload(self) -> None:
                    self.unloaded = True

                def on_event(self, event: dict) -> None:
                    self.events.append({{
                        "marker": "v2",
                        "type": event.get("type"),
                        "payload": event.get("payload"),
                    }})

            plugin = ReloadPlugin()
            """
        )

        try:
            with _temp_dir("tmp_reload") as tmp:
                plugin_path = f"{tmp}/reload_plugin.py"
                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v1)

                loaded = st.plugin.load_path(plugin_path, recursive=False, strict=True)
                self.assertIn(plugin_id, loaded)

                old_plugin = st.plugin.get_service(service_name)
                self.assertIsNotNone(old_plugin)

                meta = st.plugin.plugin_metadata(plugin_id)
                self.assertIsInstance(meta, dict)
                self.assertEqual(meta["version"], "0.0.1")
                self.assertIsInstance(meta.get("metadata"), dict)
                extra = meta["metadata"]
                self.assertEqual(extra.get("spiraltorch.source"), "path")
                self.assertEqual(Path(extra["spiraltorch.source_path"]).resolve(), Path(plugin_path).resolve())
                self.assertTrue(str(extra["spiraltorch.source_module"]).startswith("spiraltorch_path_plugin_"))

                st.plugin.publish("ReloadDemo", {"x": 1})
                self.assertTrue(old_plugin.events)
                self.assertEqual(old_plugin.events[-1]["marker"], "v1")

                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v2)

                reloaded = st.plugin.reload_path(plugin_path, recursive=False, strict=True)
                self.assertIn(plugin_id, reloaded)

                meta2 = st.plugin.plugin_metadata(plugin_id)
                self.assertIsInstance(meta2, dict)
                self.assertEqual(meta2["version"], "0.0.2")
                self.assertIsInstance(meta2.get("metadata"), dict)
                extra2 = meta2["metadata"]
                self.assertEqual(extra2.get("spiraltorch.source"), "path")
                self.assertEqual(Path(extra2["spiraltorch.source_path"]).resolve(), Path(plugin_path).resolve())
                self.assertTrue(str(extra2["spiraltorch.source_module"]).startswith("spiraltorch_path_plugin_"))

                new_plugin = st.plugin.get_service(service_name)
                self.assertIsNotNone(new_plugin)
                self.assertIsNot(old_plugin, new_plugin)
                self.assertTrue(old_plugin.unloaded)

                old_plugin.events.clear()
                new_plugin.events.clear()
                st.plugin.publish("ReloadDemo", {"x": 2})
                self.assertFalse(old_plugin.events)
                self.assertTrue(new_plugin.events)
                self.assertEqual(new_plugin.events[-1]["marker"], "v2")

                st.plugin.unregister_plugin(plugin_id)
                self.assertNotIn(plugin_id, st.plugin.list_plugins())
                self.assertTrue(new_plugin.unloaded)
        finally:
            try:
                st.plugin.unregister_plugin(plugin_id)
            except Exception:
                pass

    def test_python_plugin_load_path_dependency_order(self) -> None:
        plugin_a_id = f"demo_dep_a_{uuid.uuid4().hex}"
        plugin_b_id = f"demo_dep_b_{uuid.uuid4().hex}"
        service_a = f"{plugin_a_id}.value"
        service_b = f"{plugin_b_id}.seen"

        plugin_source_v1 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class PluginA:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_a_id}",
                        "version": "0.0.1",
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_a}", "v1")

            class PluginB:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_b_id}",
                        "version": "0.0.1",
                        "dependencies": {{"{plugin_a_id}": ">=0"}},
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service(
                        "{service_b}",
                        st.plugin.get_service("{service_a}"),
                    )

            plugins = [PluginB(), PluginA()]
            """
        )

        plugin_source_v2 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class PluginA:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_a_id}",
                        "version": "0.0.2",
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_a}", "v2")

            class PluginB:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_b_id}",
                        "version": "0.0.2",
                        "dependencies": {{"{plugin_a_id}": ">=0"}},
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service(
                        "{service_b}",
                        st.plugin.get_service("{service_a}"),
                    )

            plugins = [PluginB(), PluginA()]
            """
        )

        try:
            with _temp_dir("tmp_deps") as tmp:
                plugin_path = f"{tmp}/deps_plugin.py"
                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v1)

                loaded = st.plugin.load_path(plugin_path, recursive=False, strict=True)
                self.assertIn(plugin_a_id, loaded)
                self.assertIn(plugin_b_id, loaded)
                self.assertEqual(st.plugin.get_service(service_b), "v1")

                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v2)

                reloaded = st.plugin.reload_path(plugin_path, recursive=False, strict=True)
                self.assertIn(plugin_a_id, reloaded)
                self.assertIn(plugin_b_id, reloaded)
                self.assertEqual(st.plugin.get_service(service_b), "v2")
        finally:
            for plugin_id in (plugin_b_id, plugin_a_id):
                try:
                    st.plugin.unregister_plugin(plugin_id)
                except Exception:
                    pass

    def test_python_plugin_watch_path(self) -> None:
        plugin_id = f"demo_watch_path_plugin_{uuid.uuid4().hex}"
        service_name = f"{plugin_id}.instance"
        errors: list[tuple[str, str]] = []

        plugin_source_v1 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class WatchPlugin:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.1",
                        "capabilities": ["Telemetry"],
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_name}", self)

            plugin = WatchPlugin()
            """
        )

        plugin_source_v2 = textwrap.dedent(
            f"""
            import spiraltorch as st

            class WatchPlugin:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.2",
                        "capabilities": ["Telemetry"],
                    }}

                def on_load(self) -> None:
                    st.plugin.register_service("{service_name}", self)

            plugin = WatchPlugin()
            """
        )

        watcher = None
        try:
            with _temp_dir("tmp_watch") as tmp:
                plugin_path = f"{tmp}/watch_plugin.py"
                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v1)

                watcher = st.plugin.watch_path(
                    tmp,
                    recursive=False,
                    strict=True,
                    poll_interval=0.01,
                    debounce=0.0,
                    missing_grace=0.0,
                    on_error=lambda exc, filename: errors.append(
                        (exc.__class__.__name__, f"{filename}: {exc}")
                    ),
                )

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    meta = st.plugin.plugin_metadata(plugin_id)
                    if meta and meta.get("version") == "0.0.1":
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"watch_path did not load plugin {plugin_id}")

                self.assertIsInstance(meta, dict)
                self.assertIsInstance(meta.get("metadata"), dict)
                extra = meta["metadata"]
                self.assertEqual(extra.get("spiraltorch.source"), "path")
                self.assertEqual(Path(extra["spiraltorch.source_path"]).resolve(), Path(plugin_path).resolve())

                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source_v2)

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    meta = st.plugin.plugin_metadata(plugin_id)
                    if meta and meta.get("version") == "0.0.2":
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"watch_path did not reload plugin {plugin_id}")

                self.assertIsInstance(meta, dict)
                self.assertIsInstance(meta.get("metadata"), dict)
                extra = meta["metadata"]
                self.assertEqual(extra.get("spiraltorch.source"), "path")
                self.assertEqual(Path(extra["spiraltorch.source_path"]).resolve(), Path(plugin_path).resolve())

                self.assertFalse(errors)
                self.assertIsNotNone(st.plugin.get_service(service_name))
        finally:
            if watcher is not None:
                try:
                    watcher.stop(timeout=1.0)
                except Exception:
                    pass
            try:
                st.plugin.unregister_plugin(plugin_id)
            except Exception:
                pass

    def test_python_plugin_watch_path_unload_on_stop(self) -> None:
        plugin_id = f"demo_watch_unload_{uuid.uuid4().hex}"
        plugin_source = textwrap.dedent(
            f"""
            class WatchUnloadPlugin:
                def metadata(self) -> dict:
                    return {{
                        "id": "{plugin_id}",
                        "version": "0.0.1",
                    }}

            plugin = WatchUnloadPlugin()
            """
        )

        watcher = None
        try:
            with _temp_dir("tmp_watch_unload") as tmp:
                plugin_path = f"{tmp}/watch_unload.py"
                with open(plugin_path, "w", encoding="utf-8") as handle:
                    handle.write(plugin_source)

                watcher = st.plugin.watch_path(
                    plugin_path,
                    recursive=False,
                    strict=True,
                    poll_interval=0.01,
                    debounce=0.0,
                    missing_grace=0.0,
                    unload_on_stop=True,
                )

                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if plugin_id in st.plugin.list_plugins():
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"watch_path did not load plugin {plugin_id}")

                watcher.stop(timeout=1.0)
                watcher = None
                self.assertNotIn(plugin_id, st.plugin.list_plugins())
        finally:
            if watcher is not None:
                try:
                    watcher.stop(timeout=1.0)
                except Exception:
                    pass
            try:
                st.plugin.unregister_plugin(plugin_id)
            except Exception:
                pass

    def test_state_dict_io(self) -> None:
        model = st.nn.Linear("l1", 2, 1)
        with _temp_dir("tmp_state") as tmp:
            path = f"{tmp}/linear.json"
            st.nn.save(path, model)
            manifest = path.replace(".json", ".manifest.json")
            loaded = st.nn.load(manifest)
        self.assertIsInstance(loaded, list)
        model.load_state_dict(loaded)


if __name__ == "__main__":
    unittest.main()
