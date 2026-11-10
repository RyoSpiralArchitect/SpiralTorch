"""Verify input boundaries with the pinned tokenizer only; no model is loaded or scored."""
import argparse
import json
from pathlib import Path
import sys
from hashlib import sha256

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix as m

TOKEN_INPUT_PIN = "6544ac114a910e9faa5a7fb1e86f661ec7f99796cd0ddb6d7c6e161e18dcbe41"


def admit(path):
    prior = json.loads(m.pinned(m.MANIFEST, m.MANIFEST_PIN))
    with path.open("rb") as f:
        raw = f.read(4 * 1024 * 1024 + 1)
    m.check(len(raw) <= 4 * 1024 * 1024 and sha256(raw).hexdigest() == prior["tokenizer"], "wrong tokenizer")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    pairs = m.matrix()
    rows = []
    for pair in pairs:
        prefixes, (target, lf) = m.prepare_pair(pair, lambda s: tok.encode(s, add_special_tokens=False).ids)
        m.check(lf == 198 and target not in (198, 50256), "unexpected reference or LF token")
        rows.append(dict(key=pair.key, field=pair.field, target=target, lf=lf, prefixes=prefixes))
    lengths = [len(prefix) for r in rows for prefix in r["prefixes"].values()]
    m.check(max(lengths) <= 64, "outside the registered prompt length budget")
    token_id = m.digest(rows)
    m.check(token_id == TOKEN_INPUT_PIN, "tokenized pair matrix differs from the frozen input")
    report = {**m.manifest(pairs), "schema": "spiraltorch.paired_prompt_token_admission.v1",
              "tokenizer_sha256": prior["tokenizer"], "input_sha256": token_id,
              "prefix_range": [min(lengths), max(lengths)], "lf_id": 198,
              "distinct_target_ids": len({r["target"] for r in rows}), "model_calls": 0}
    return report, rows


def verify(path):
    return admit(path)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", type=Path)
    print(json.dumps(verify(parser.parse_args().tokenizer_json), indent=2))
