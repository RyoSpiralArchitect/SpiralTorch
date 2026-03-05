# SpiralTorch entry point plugin demo

This folder is a minimal **Python plugin package** that registers itself via `entry_points` so
SpiralTorch can discover it with `st.plugin.load_entrypoints()`.

## Install

1. Install SpiralTorch (wheel) or build the bindings locally:

   - `maturin develop -m bindings/st-py/Cargo.toml`

2. Install this demo plugin (editable):

   - `cd examples/python_entrypoint_plugin_demo`
   - `python -m pip install -e .`

## Try it

```python
import spiraltorch as st

loaded = st.plugin.reload_entrypoints(strict=False)
print("loaded:", loaded)
print("plugins:", st.plugin.list_plugins())
print("hello:", st.plugin.get_service("demo_entrypoint_plugin.hello"))
unloaded = st.plugin.unload_entrypoints()
print("unloaded:", unloaded)
```

Or run:

- `python run_demo.py`
