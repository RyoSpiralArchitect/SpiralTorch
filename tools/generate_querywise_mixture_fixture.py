"""Independent CPU-f32 query-local router VJPs and one SGD step."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def case(name, shape):
    b, h, q, s, k = shape
    x = torch.sin(torch.arange(b*h*q*s*k, dtype=torch.float32) * .31).reshape(shape).requires_grad_()
    logits = torch.cos(torch.arange(b*h*q*k, dtype=torch.float32) * .47).reshape(b,h,q,k).requires_grad_()
    seed = torch.cos(torch.arange(b*h*q*s, dtype=torch.float32) * .29).reshape(b,h,q,s)
    weights = logits.softmax(-1)
    output = (x * weights.unsqueeze(-2)).sum(-1)
    gx, gl = torch.autograd.grad(output, (x, logits), seed)
    flat = lambda x: x.detach().reshape(-1).tolist()
    return dict(name=name,shape=shape,input=flat(x),logits=flat(logits),seed=flat(seed),
                output=flat(output),weights=flat(weights),input_gradient=flat(gx),logits_gradient=flat(gl))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = args.corpus_fixture.read_bytes()
    source = json.loads(raw)
    assert source["schema"] == "spiraltorch.querywise_mixture.torch_fixture.v1" and len(source["cases"]) == 24
    cases = [case("synthetic_components_"+str(k),[2,3,4,5,k]) for k in (2,3,8)] + source["cases"]
    for c in cases:
        shape = c["shape"]
        assert len(shape) == 5 and all(type(n) is int and n>0 for n in shape)
        b,h,q,s,k=shape
        x = torch.tensor(c["input"]).reshape(shape)
        logits = torch.tensor(c["logits"]).reshape(b,h,q,k)
        assert torch.isfinite(x).all() and torch.isfinite(logits).all()
        gx = torch.tensor(c["input_gradient"]).reshape(shape)
        gl = torch.tensor(c["logits_gradient"]).reshape(b,h,q,k)
        x_after, logits_after = x-.05*gx, logits-.05*gl
        c["updated_input"] = x_after.reshape(-1).tolist()
        c["updated_logits"] = logits_after.reshape(-1).tolist()
        c["updated_output"] = (x_after*logits_after.softmax(-1).unsqueeze(-2)).sum(-1).reshape(-1).tolist()
    payload = dict(schema="spiraltorch.querywise_mixture.qualification.v1",torch_version=torch.__version__,
        device="cpu",dtype="float32",threads=1,corpus_fixture_sha256=hashlib.sha256(raw).hexdigest(),
        rate=.05,cases=cases)
    with args.output.open("x") as out:
        json.dump(payload,out,separators=(",",":"),allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(cases=len(cases),corpus_cases=24,synthetic_cases=3)))


if __name__ == "__main__":
    main()
