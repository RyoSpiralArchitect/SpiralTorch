"""Score pinned native SGD snapshots on one pinned, labeled corpus without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import spiraltorch as st
from spiraltorch.native_session_io import read_limited, read_pinned_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True, help="JSON object with token_documents and unshifted labels")
    parser.add_argument("--documents-sha256", required=True, help="separately retained SHA-256 of that exact JSON file")
    parser.add_argument("--variant", action="append", nargs=4, required=True,
                        metavar=("NAME", "CHECKPOINT", "PIN", "BOUNDARY"))
    parser.add_argument("--reference", required=True, help="variant to subtract from each other variant")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--microbatch-windows", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    names = [arm[0] for arm in args.variant]
    if (not 2 <= len(names) <= st.NativeCausalLmAnswerComparison.MAX_VARIANTS or len(set(names)) != len(names)
            or not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", name) for name in names)):
        parser.error("select 2 to 16 variants with distinct short names")
    if args.reference not in names:
        parser.error("reference must name a selected variant")
    if os.path.lexists(args.output):
        parser.error("output must be a new file")
    if not args.output.parent.is_dir():
        parser.error("output needs an existing parent directory")
    raw = read_limited(args.documents, 256 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != args.documents_sha256:
        parser.error("evaluation corpus differs from pinned SHA-256")
    data = json.loads(raw)
    del raw
    if (not isinstance(data, dict) or set(data) != {"token_documents", "labels"}
            or not all(isinstance(v, list) for v in data.values())):
        parser.error("answer scoring needs explicit token_documents and labels arrays")

    base = st.NativeCausalLm.from_gpt2(read_limited((args.model_dir/"config.json").resolve(strict=True), 1_048_576).decode(),
                                       st.SafeTensorArchive(args.model_dir/"model.safetensors"))
    models, boundaries = [], []
    for name, checkpoint, pin, metadata in args.variant:
        raw, boundary = read_pinned_session(Path(checkpoint), pin, Path(metadata))
        models.append(st.NativeCausalLm.from_session_checkpoint_at(base, raw, pin, **boundary))
        del raw
        boundaries.append(dict(name=name, checkpoint_id=pin, **boundary))
    if len({(b["update"], b["recorded_data_id"]) for b in boundaries}) != 1:
        parser.error("snapshots must agree on completed update and recorded training corpus")
    scored = st.NativeCausalLm.evaluate_variants_by_answer(models, data["token_documents"], args.context,
                                                        microbatch_windows=args.microbatch_windows, labels=data["labels"])
    ref = names.index(args.reference)
    result = dict(schema="spiraltorch.pinned_answer_comparison.v1", reference=args.reference,
                  documents_sha256=args.documents_sha256, variants=boundaries,
                  contrasts={name:scored.contrast(ref,i) for i,name in enumerate(names) if i != ref},
                  scores=json.loads(scored.to_json()), training_documents_verified=False, split_independence_verified=False)
    # Publish only the complete result; a raced output name is never overwritten.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp)/"result"
        with open(staged, "x", encoding="utf8", opener=lambda p,f: os.open(p,f,0o600)) as stream:
            json.dump(result, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.link(staged, args.output)
    print(json.dumps({"reference":args.reference, "answers":len(scored), "contrasts":result["contrasts"]}, allow_nan=False))


if __name__ == "__main__":
    main()
