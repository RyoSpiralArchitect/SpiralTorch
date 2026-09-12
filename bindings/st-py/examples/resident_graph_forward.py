"""Replay frozen Rust NN plans through public Python GPU handles, without Torch.

Provide a resident_graph_forward fixture as JSON or xz. All intermediate model
outputs stay on the GPU until the workspaces are reused and destroyed.
"""
import argparse
import gc
import hashlib
import json
import lzma
import math
from pathlib import Path
import spiraltorch as st


def close(actual, expected):
    if len(actual) != len(expected) or any(not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 2e-5+2e-4*abs(b) for a,b in zip(actual,expected)):
        raise AssertionError("capture differs from frozen core fixture")


def rejected(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError("invalid call was accepted")


def capture(case):
    shape, values = case["shape"], case["input"]
    plan = st.nn.InferencePlan.from_json(json.dumps(case["plan"]))
    graph = plan.compile_graph_wgpu(kernel=case["kernel"], accumulation=case["accumulation"])
    # Separately created public devices and compiled graphs share the runtime.
    device = st.WgpuTensorDevice.create()
    assert graph.adapter_info()["device_type"] != "Cpu"
    graph.upload_values(values)
    assert graph.dispatch() == 1
    host = graph.snapshot()
    assert (host.generation,host.submitted_dispatch)==(1,1)
    rejected(lambda: graph.upload_values([]))
    rejected(lambda: graph.set_input_tensor(device.upload([len(values),1],values)))
    assert graph.generation == 1
    padded_shape = [*shape[:-1],shape[-1]+2]
    padded = [v for i in range(0,len(values),4) for v in [91.,*values[i:i+4],91.]]
    source = device.upload(padded_shape,padded)
    view = source.narrow(len(shape)-1,1,4)
    graph.set_input_tensor(view)
    assert graph.dispatch()==2
    strided = graph.snapshot()
    pre = view.mul(device.upload([4],case["pre_gains"])).relu()
    graph.set_input_tensor(pre)
    for i in range(3,19): assert graph.dispatch()==i
    prediction = graph.snapshot()
    assert (prediction.generation,prediction.submitted_dispatch)==(3,18)
    frozen = graph.output_tensor()
    post = frozen.add(graph.tensor_device().upload([3],case["post_shift"])).gelu()
    next_model = st.nn.Sequential()
    next_model.add(st.nn.Relu())
    next_graph = next_model.inference_plan(graph.output_shape).compile_graph_wgpu()
    next_graph.set_input_tensor(post)
    next_graph.dispatch()
    chained = next_graph.output_tensor()
    next_graph.upload_values([0.]*chained.numel); next_graph.dispatch()
    graph.upload_values([0.]*len(values)); graph.dispatch()
    adapter = graph.adapter_info()
    del graph,next_graph,plan,next_model,device,source,view,pre
    gc.collect()
    # These GPU captures are snapshotted only AFTER both source workspaces die.
    captures = dict(host=host.read_values(),strided=strided.read_values(),prediction=prediction.read_values(),
                    frozen=frozen.snapshot().read_values(),post=post.snapshot().read_values(),chained=chained.snapshot().read_values())
    for s in (host,strided,prediction): rejected(s.read_values)
    for name,actual in captures.items(): close(actual,case[name])
    return {key:case[key] for key in ("seed","shape","kernel","accumulation","plan","input","pre_gains","post_shift","dispatches_before_capture")} | captures, adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    raw = args.fixture.read_bytes()
    if args.fixture.suffix==".xz": raw=lzma.decompress(raw)
    fixture = json.loads(raw)
    if fixture.get("schema")!="spiraltorch.resident_graph_forward.v1" or fixture.get("status")!="passed" or len(fixture.get("cases",[]))!=12:
        raise ValueError("a complete frozen core forward fixture is required")
    report = dict(schema="spiraltorch.resident_graph_forward_client.v1",client="python",status="error",cases=[],
                  source_fixture_sha256=hashlib.sha256(raw).hexdigest(),native_library=st._rs.__file__)
    with args.output.open("x") as output:
        try:
            for case in fixture["cases"]:
                captured,adapter=capture(case)
                report["cases"].append(captured)
                report["adapter"]=adapter
            report.update(status="passed",guards=dict(owned_output=True,late_readback=True,atomic_input=True,
                          single_consumption=True,shared_runtime=True))
        except Exception as error:
            report["error"]=repr(error)
            raise
        finally:
            json.dump(report,output,indent=2,allow_nan=False)
            output.write("\n")


if __name__=="__main__": main()
