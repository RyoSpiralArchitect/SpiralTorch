"""Generate small analytic review witnesses for the existing local GPU probe."""
import argparse
import json
from pathlib import Path


def cases():
    output = []
    for n in (9, 17, 33):
        output.append({"name":f"compensated_denominator_{n}", "batch_index":0,
                       "kind":"attention", "operation":"review.attention", "shape":[1,1,n,1],
                       "query":[0.]*n, "key":[2.**127]*n, "value":[0.]+[1.]*(n-1),
                       "upstream":[0.]*(n-1)+[2.**120], "scale":1.,
                       "pair":([0.]+[-85.]*(n-1))*n})
    output.append({"name":"normal_output_midpoint", "batch_index":0,
                   "kind":"attention", "operation":"review.attention", "shape":[1,1,4,1],
                   "query":[0.]*4,"key":[0.]*4,"value":[4.,2.**-22,2.**-47,0.],
                   "upstream":[0.]*4,"scale":1.,"pair":None})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump({"schema":"spiraltorch.language_local_vjp.fixture.v1","cases":cases()},stream,allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
