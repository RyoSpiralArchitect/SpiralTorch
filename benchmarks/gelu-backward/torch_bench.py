"""Matched ATen tanh-GELU derivative, with packed owning observation on CPU/MPS."""
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol_gelu as p
import torch


def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None,"0"):
        raise RuntimeError("MPS fallback must remain disabled")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("real MPS device required")
    cases = []
    for rows,cols,count in sorted(p.KEYS):
        z,g,r = p.inputs(rows,cols)
        n = rows*cols
        reference = p.oracle(rows,cols,count)
        z64 = torch.tensor(z,dtype=torch.float64).reshape(rows,cols)
        g64 = torch.tensor(g,dtype=torch.float64).reshape(rows,cols)
        d64 = torch.ops.aten.gelu_backward(g64,z64,approximate="tanh")
        expected = [d64.flatten()]
        if count == 3:
            expected += [(d64+torch.tensor(r,dtype=torch.float64).reshape(rows,cols)).flatten(),d64.sum(dim=0)]
        p.close(torch.cat(expected).tolist(),reference,rows,cols)
        prepared = []
        for device in ("cpu","mps"):
            x = torch.tensor(z,dtype=torch.float32,device=device).reshape(rows,cols)
            seed = torch.tensor(g,dtype=torch.float32,device=device).reshape(rows,cols)
            residual = torch.tensor(r,dtype=torch.float32,device=device).reshape(rows,cols)
            flat = torch.empty(n if count==1 else 2*n+cols,dtype=torch.float32,device=device)
            grad = flat[:n].view(rows,cols)
            prepared.append((x,seed,residual,flat,grad))
        torch.mps.synchronize()
        intervals,last = [],[None,None]
        for block in range(12):
            order = p.order(block,rows,cols,count,True)
            for burst in (1,4):
                for route in order:
                    x,seed,residual,flat,grad = prepared[route]
                    start = time.perf_counter_ns()
                    for _ in range(burst):
                        torch.ops.aten.gelu_backward.grad_input(seed,x,approximate="tanh",grad_input=grad)
                        if count == 3:
                            torch.add(residual,grad,out=flat[n:2*n].view(rows,cols))
                            torch.sum(grad,dim=0,out=flat[2*n:])
                    observed = flat.to("cpu",copy=True)
                    elapsed_ms = (time.perf_counter_ns()-start)/1e6
                    output = observed.tolist()
                    absolute,scaled = p.close(output,reference,rows,cols)
                    if block >= 3:
                        intervals.append(dict(block=block-3,burst=burst,route=("cpu","mps")[route],order=order,
                            elapsed_ms=elapsed_ms,max_abs_error=absolute,max_scaled_error=scaled))
                    last[route] = output
        cases.append(dict(rows=rows,cols=cols,count=count,order_scheme="balanced-cycle-v1",input=z,seed=g,residual=r,
            reference=reference,last_outputs=last,intervals=intervals))
    print(json.dumps(dict(schema=p.TORCH,status="passed",blocks=9,warmup=3,bursts=[1,4],cases=cases,
        devices=["cpu","mps"],intra_op_threads=4,inter_op_threads=1,compiled=False,torch_version=torch.__version__,
        observation="one_packed_owning_cpu_copy",operation="aten.gelu_backward.grad_input:tanh"),allow_nan=False))


if __name__ == "__main__":
    main()
