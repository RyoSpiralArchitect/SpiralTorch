"""Installed Python wheel against the frozen v2 native/WASM QKV learning fixture."""
import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import sys
from pathlib import Path
import traceback
import zipfile

import spiraltorch as st
from validate_module_projective_results import verify
from validate_resident_query_pullback import close, difference, exact, relative, require


def values(tensor):
    return tensor.snapshot().read_values()


def installed_identity(wheel):
    """Reject checkout shadowing and bind the imported API to a concrete wheel."""
    distribution = importlib.metadata.distribution("spiraltorch")
    native = importlib.import_module("spiraltorch.spiraltorch")
    paths = [Path(st.__file__).resolve(), Path(native.__file__).resolve()]
    require(any(str(paths[1]).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES),
            "native owner is not an extension")
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for path in paths:
            member = "spiraltorch/" + path.name
            require(names.count(member) == 1, "missing or duplicated wheel member: " + member)
            require(path == Path(distribution.locate_file(member)).resolve(),
                    "import is shadowing the installed distribution: " + member)
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    "installed file differs from candidate wheel: " + member)
            hashes[member] = expected
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "ambiguous wheel metadata")
        from email.parser import BytesParser
        package = BytesParser().parsebytes(archive.read(metadata[0]))
        require(package["Name"] == "spiraltorch", "candidate is not a SpiralTorch wheel")
        version = package["Version"]
        require(version == distribution.version == st.__version__, "wheel version mismatch")
    return {"version": version, "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "installed_files": hashes, "python": sys.version.split()[0],
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def external_gradient_references(case):
    # The fixture stores only present derivatives, in z/pair order per block.
    flat = iter(case["bias_gradients"])
    missing = object()
    rows = []
    for specification in case["biases"]:
        row = {}
        for key in ("z", "pair"):
            expected = None if specification[key] is None else next(flat, missing)
            require(expected is not missing, "missing external bias derivative")
            row[key] = expected
        rows.append(row)
    require(next(flat, missing) is missing, "extra external bias derivative")
    return rows


def initial_checkpoint(case, native):
    # Reuse validated native topology, reset *every* tensor to the frozen CPU
    # fixture, then verify the uploaded tensors. No model equations live here.
    record = json.loads(native["resume_trajectory"]["checkpoint_json"])
    model = record["model"]
    destinations = [model["token"]["values"], model["position"]["values"]]
    geometry = model["geometry"]
    if geometry is not None:
        destinations.extend(p["values"] for p in geometry["projection"]["parameters"])
        destinations.extend([geometry["raw_decay"], geometry["raw_phase"]])
        destinations.extend(geometry["raw_gains"])
        if "mixture" in geometry:
            mix = geometry["mixture"]
            destinations.extend(mix["raw_logits"])
            for graph in mix.get("query_projections", []):
                destinations.extend(p["values"] for p in graph["parameters"])
    for block in model["blocks"]:
        for name in ("pre", "qkv", "output", "feed_forward"):
            destinations.extend(p["values"] for p in block[name]["parameters"])
    destinations.extend(p["values"] for p in model["head"]["parameters"])
    require(len(destinations) == len(case["parameters"]), "initial parameter family mismatch")
    for destination, parameter in zip(destinations, case["parameters"], strict=True):
        require(len(destination) == len(parameter["values"]), "initial parameter shape mismatch")
        destination[:] = parameter["values"]
    record["attempted_revision"] = "0"
    return st.nn.ByteDecoderCheckpoint.from_json(json.dumps(record, allow_nan=False))


def biases(model, case):
    c = case["config"]
    device = model.tensor_device()
    b, h, t = c["batch"], c["heads"], c["steps"]
    return {
        "z_biases": [None if p["z"] is None else device.upload([b, h, t], p["z"]) for p in case["biases"]],
        "pair_biases": [None if p["pair"] is None else device.upload([b, h, t, t], p["pair"]) for p in case["biases"]],
    }


def submit_step(model, data, external, rate):
    forward = model.forward(data, **external)
    loss = forward.next_byte_loss()
    gradients = model.backward(forward, loss.prediction_gradient_tensor())
    weights = [forward.geometry_mixture_weights_tensor(i) for i in range(len(external["z_biases"]))]
    update = model.sgd(gradients, rate)
    return {"update": update, "loss": loss.loss_tensor(),
            "parameters": model.parameter_tensors(),
            "gradients": gradients.parameter_gradient_tensors(),
            "embedding": gradients.embedding_output_gradient_tensor(), "weights": weights}


def read_step(pending):
    return {"revision": pending["update"].read(), "loss": values(pending["loss"])[0],
            "parameters": [values(t) for t in pending["parameters"]],
            "gradients": [values(t) for t in pending["gradients"]],
            "embedding": values(pending["embedding"]),
            "weights": [values(t) for t in pending["weights"] if t is not None]}


def run_case(case, native, browser):
    require(case["name"] == native["name"] == browser["name"], "case order mismatch")
    model = initial_checkpoint(case, native).restore_wgpu()
    device = model.tensor_device()
    adapter = device.adapter_info()
    require(adapter["device_type"] != "Cpu", "real GPU required")
    p = model.parameter_layout()
    begin, end = p["geometry"]["all"]
    for actual, parameter in zip(model.parameter_tensors(), case["parameters"], strict=True):
        require(values(actual) == parameter["values"], "initial values changed")
    c = case["config"]
    data = model.prepare_batch(st.nn.ByteLmBatch.from_documents(
        [bytes(window) for window in case["windows"]],
        [(i, 0) for i in range(c["batch"])], c["steps"]))
    external = biases(model, case)
    forward = model.forward(data, **external)
    output = values(forward.prediction_tensor())
    for reference in (case, native, browser):
        close(output, reference["output"])
    seed = device.upload([c["batch"], c["steps"], 256], case["cotangent"])
    gradients = model.backward(forward, seed)
    full = [values(t) for t in gradients.parameter_gradient_tensors()]
    for i, (a, e) in enumerate(zip(full, case["parameter_gradients"], strict=True)):
        close(a, e)
        if begin <= i < end:
            relative(a, e)
    embedded = values(gradients.embedding_output_gradient_tensor())
    close(embedded, case["embedding_output_gradient"])
    for i, expected in enumerate(external_gradient_references(case)):
        for key, actual in [("z", gradients.z_bias_gradient_tensor(i)), ("pair", gradients.pair_bias_gradient_tensor(i))]:
            if expected[key] is None:
                require(actual is None, "unexpected external bias derivative")
            else:
                close(values(actual), expected[key])
    paired = None
    if "query_controls" in case:
        stopped = model.backward(forward, seed, query_pullback="stop_coordinates")
        stop_embed = values(stopped.embedding_output_gradient_tensor())
        expected = case["query_controls"]
        close(stop_embed, expected["router_detached_embedding_gradient"])
        stopped_parameters = [values(t) for t in stopped.parameter_gradient_tensors()]
        for i, (a, e) in enumerate(zip(stopped_parameters, expected["router_detached_parameter_gradients"], strict=True)):
            close(a, e)
            if begin <= i < end:
                relative(a, e)
            name = case["parameters"][i]["name"]
            affected = i < 2 or name.startswith(("geometry.projection.", "geometry.raw_decay", "geometry.raw_phase"))
            if not affected:
                require(exact(a, full[i]), "unrelated stopped route changed")
        observed = difference(embedded, stop_embed)
        target = difference(case["embedding_output_gradient"], expected["router_detached_embedding_gradient"])
        paired = relative(observed, target)
        for broken in ([0.] * len(observed), [-v for v in observed]):
            try:
                relative(broken, target)
            except ValueError:
                pass
            else:
                raise ValueError("paired checker accepted inert/reversed pullback")
        require(exact(values(forward.prediction_tensor()), output), "backward changed output")
        repeated = model.backward(forward, seed)
        require(exact([values(t) for t in repeated.parameter_gradient_tensors()], full), "Full repetition changed")
    pending, split_capture = [], None
    for index in range(16):
        pending.append(submit_step(model, data, external, case["learning"]["rate"]))
        if index == 6:
            split_capture = model.checkpoint_snapshot()
    observations = [read_step(step) for step in pending]
    trace = []
    for index, actual in enumerate(observations):
        require(actual["revision"] == index + 1, "update rejected or revision mismatch")
        expected = case["learning"]["trace"][index]
        measured_loss = actual["loss"]
        close([measured_loss], [expected["loss"]])
        parameter_values = actual["parameters"]
        parameter_error = max(close(a, e)["max_abs"] for a, e in zip(parameter_values, expected["parameters"], strict=True))
        embedding_error = close(actual["embedding"], expected["embedding_output_gradient"])
        gradient_errors = []
        for a, e in zip(actual["gradients"][begin:end], expected["geometry_gradients"], strict=True):
            close(a, e)
            gradient_errors.append(relative(a, e)["relative_l2"])
        for a, e in zip(actual["weights"], expected.get("mixture_weights", []), strict=True):
            close(a, e)
        for reference in (native, browser):
            old = reference["learning"]["trace"][index]
            close([measured_loss], [old["loss"]])
            close(actual["embedding"], old["embedding_output_gradient"])
            for a, e in zip(actual["gradients"][begin:end], old["geometry_gradients"], strict=True):
                close(a, e)
            for a, e in zip(actual["weights"], old.get("mixture_weights", []), strict=True):
                close(a, e)
            for a, e in zip(parameter_values, old["parameters"], strict=True):
                close(a, e)
        trace.append({"step": index + 1, "loss": measured_loss,
                      "parameter_max_abs_error": parameter_error,
                      "embedding_max_abs_error": embedding_error["max_abs"],
                      "geometry_relative_l2": gradient_errors})
    saved = split_capture.read().to_json()
    restored = st.nn.ByteDecoderCheckpoint.from_json(saved).restore_wgpu()
    require(restored.attempted_updates == 7, "restore lost attempted revision")
    resumed_data = restored.prepare_batch(st.nn.ByteLmBatch.from_windows(
        [bytes(window[:c["steps"] + 1]) for window in case["windows"]]))
    resumed_external = biases(restored, case)
    continued = [submit_step(restored, resumed_data, resumed_external, case["learning"]["rate"])
                 for _ in range(9)]
    for expected, step in zip(observations[7:], continued, strict=True):
        require(exact(read_step(step), expected), "checkpoint continuation/VJP differs")
    final = model.checkpoint_snapshot().read().to_json()
    require(restored.checkpoint_snapshot().read().to_json() == final, "final checkpoint differs")
    return {"name": case["name"], "passed": True, "adapter": adapter, "trace": trace, "paired_query": paired,
            "resume_revision": 7, "exact_continuation_to": 16, "resumed_steps": 9,
            "checkpoint_sha256": hashlib.sha256(saved.encode()).hexdigest(),
            "final_checkpoint_sha256": hashlib.sha256(final.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    installation = installed_identity(args.wheel)
    inputs = {}
    def read(name):
        data = (args.fixture_dir / name).read_bytes()
        inputs[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    fixture, identity = read("reference.json"), read("fixture-identity.json")
    native, bundle = read("native.json"), read("browser.json")
    require(inputs["reference.json"] == identity["fixture_sha256"] == bundle["fixture_sha256"], "fixture identity mismatch")
    verify(fixture, native)
    verify(fixture, bundle["report"])
    rows = []
    for case, n, b in zip(fixture["cases"], native["checks"], bundle["report"]["checks"], strict=True):
        try:
            row = run_case(case, n, b)
        except Exception as error:
            traceback.print_exc()
            row = {"name": case["name"], "passed": False, "error": str(error)}
        rows.append(row)
        print(json.dumps({"name": row["name"], "passed": row["passed"]}), flush=True)
    passed = all(row["passed"] for row in rows) and len(rows) == 6
    result = {"schema": "spiraltorch.python_byte_decoder_parity.v1",
              "scope": "installed-wheel mechanics against frozen v2; no language-quality or speed claim",
              "passed": passed, "installation": installation, "input_hashes": inputs, "checks": rows}
    with args.output.open("x") as output:
        json.dump(result, output, allow_nan=False)
        output.write("\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
