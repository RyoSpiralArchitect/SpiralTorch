"""CPU-only CPython allocation probe. Run in its own interpreter, never in a caller's tracer."""
import json
from pathlib import Path
import runpy
import tempfile
import tracemalloc

import spiraltorch as st


def measured(call):
    tracemalloc.stop()
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main():
    here = Path(__file__).resolve().parent
    tracemalloc.stop()
    config, weights = runpy.run_path(str(here / "test_native_causal_lm.py"))["fixture"]()
    base = st.NativeCausalLm.from_gpt2(config, weights)
    text = "a" * (1024 * 1024 - 1) + "\U0001f600"
    wrong = "sha256:" + "0" * 64
    def reject():
        try:
            st.NativeCausalLm.from_session_checkpoint_at(base, text, wrong, update=0,
                recorded_data_id=wrong, post_lora_id=None, post_geometry_id=None)
        except ValueError as error:
            if "pinned identity" not in str(error):
                raise
        else:
            raise AssertionError("accepted wrong pin")
    result = {"unicode_peak": measured(reject), "file_peaks": []}
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)/"input"
        sample.write_bytes(b"x" * 4465)
        for name, entry in (("native_training_session.py","read_limited"),("native_geometry_generation.py","read_bytes")):
            read = runpy.run_path(str(here.parent/"examples"/name))[entry]
            def load():
                if len(read(sample, 256*1024*1024)) != 4465:
                    raise ValueError("wrong file size")
            result["file_peaks"].append(measured(load))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
