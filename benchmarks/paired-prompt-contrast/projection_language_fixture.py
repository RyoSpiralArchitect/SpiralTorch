"""Generate only the small synthetic fixture for the paired_projection_language Rust example."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

W, V, INNER, POS, STOP = 32, 64, 48, 16, 1
FIELDS = ("color", "shape", "material")
ARMS = ("absolute_equal", "anchored_equal", "absolute_common", "anchored_common")


def pack(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def exclusive(path, data):
    with path.open("xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def weights():
    header, chunks, cursor = {}, [], 0
    seed = 0x1AC73DF4

    def noise(scale):
        nonlocal seed
        seed = (1664525 * seed + 1013904223) & 0xffffffff
        return scale * (2 * seed / 2**32 - 1.)

    def add(name, shape, value):
        nonlocal cursor
        vals = [value(i) for i in range(math.prod(shape))]
        data = struct.pack("<" + str(len(vals)) + "f", *vals)
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[cursor, cursor+len(data)])
        chunks.append(data)
        cursor += len(data)

    def embedding(i):
        tok, col = divmod(i, W)
        v = noise(.42)
        if 29 <= tok <= 31 and col == tok-29:
            v += 2.0
        if 24 <= tok <= 26 and col == 3+tok-24:
            v += 2.0
        if tok in (27, 28) and col == 6:
            v += 2.0 if tok == 27 else -2.0
        if 40 <= tok <= 42:
            v += .75 if col == 3+tok-40 else .3 if col == 6 else 0.
        if 2 <= tok <= 10 and col == 16:
            v += 2.5
        if 14 <= tok <= 22 and col == 7+tok-14:
            v += 2.5
        return v

    def head(i):
        tok, col = divmod(i, W)
        v = noise(.12)
        if tok == STOP and col == 16:
            v += 1.8
        if 2 <= tok <= 10 or 14 <= tok <= 22:
            short = tok <= 10
            j = tok - (2 if short else 14)
            if col == j % 3:
                v += 1.05
            if col == 3+j//3:
                v += .7
            if col == 6:
                v += .9 if short else -.9
            if col == 7+j and short:
                v += 1.45
            if col == 17+j:
                v += .25 if short else -.25
        return v

    add("wte.weight", [V, W], embedding)
    add("wpe.weight", [POS, W], lambda _: noise(.07))
    for l in range(2):
        p = f"h.{l}."
        for norm in ("ln_1", "ln_2"):
            add(p+norm+".weight", [W], lambda _: 1.)
            add(p+norm+".bias", [W], lambda _: 0.)
        def qkv(i):
            row, col = divmod(i, 3*W)
            return noise(.015) + ((.11, .11, .65)[col//W] if row == col%W else 0.)
        add(p+"attn.c_attn.weight", [W, 3*W], qkv)
        add(p+"attn.c_attn.bias", [3*W], lambda _: noise(.01))
        add(p+"attn.c_proj.weight", [W, W], lambda i: noise(.02) + (.46 if i//W == i%W else 0.))
        add(p+"attn.c_proj.bias", [W], lambda _: noise(.01))
        add(p+"mlp.c_fc.weight", [W, INNER], lambda _: noise(.05))
        add(p+"mlp.c_fc.bias", [INNER], lambda _: noise(.01))
        add(p+"mlp.c_proj.weight", [INNER, W], lambda _: noise(.05))
        add(p+"mlp.c_proj.bias", [W], lambda _: noise(.01))
    add("ln_f.weight", [W], lambda _: 1.)
    add("ln_f.bias", [W], lambda _: 0.)
    add("lm_head.weight", [V, W], head)
    raw = pack(header)
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"".join(chunks)


def roster():
    pairs, queries, truth = [], [], []
    for form in range(2):
        for key in range(3):
            for f, field in enumerate(FIELDS):
                k = (key+f+form) % 3
                j = 3*f+k
                short, long = [2+j], [14+j, 2+j]
                def sample(side, answer):
                    cue = 27+side
                    prompt = ([32,24+f,29+k,cue,40+f] if form == 0 else
                              [33,cue,29+k,24+f,40+f])
                    return dict(prompt=prompt, answer=answer)
                pairs.append(dict(field=field, short=sample(0,short), long=sample(1,long)))
                if form == 0:
                    for side, a in enumerate((short,long)):
                        slot = len(queries)
                        queries.append(dict(slot=slot, field=field, prompt=[34,29+k,24+f,27+side,40+f]))
                        truth.append(dict(slot=slot, answer=a))
    return pairs, queries, truth


def prepare(folder):
    folder.mkdir(mode=0o700)
    pairs, queries, truth = roster()
    cfg = dict(model_type="gpt2", vocab_size=V, n_positions=POS, n_embd=W, n_head=4, n_layer=2,
               n_inner=INNER, tie_word_embeddings=False)
    pieces = {"model.safetensors":weights(), "config.json":pack(cfg), "train.json":pack(pairs),
              "queries.json":pack(queries), "truth.json":pack(truth)}
    pins = {k:exclusive(folder/k,v) for k,v in pieces.items()}
    plan = dict(schema="spiraltorch.synthetic_paired_projection_language.v1", files=pins, arms=ARMS,
        data="36 training examples (two templates); 18 new-order prompts, same synthetic labels; not pretrained or natural language.",
        model="Seeded, partly hand-structured two-block GPT-format decoder. Frozen untied readout, base and nonzero LoRA.",
        objective="Equal example means. Full content plus LF; half full CE and half reciprocal binary CE on the first answer token only. Later rows use full CE.",
        rate=.035, mix=.5, stop_id=STOP, vector_and_length_limit=1e-4, actual_change_threshold=1e-6,
        step_control="Absolute equal supplies each actual length. All other arms match it on their own current weights. Never rescale endpoints.",
        stop="First numerical refusal retires that nonreference arm without retry; no partial endpoint score. Missing reference or unexpected error stops the entire study.",
        mechanism_gate="Each chart: both observed example losses decrease on more steps than its equal arm, and at least one loss increases on no more steps; both arms must complete.",
        quality_gate="Each chart: more correct full lines on the 18 queries, and no fewer correct full lines on the nine one-token golds; both arms must complete.",
        budgets=dict(pairs=18, arms=4, updates_max=72, gradients_max=144, independent_train_forwards_max=288,
                     queries=18, lines_per_query=6, owners_max=6, root_and_prefix_rows_max=1080, cpu_wall_seconds=120, retries=0))
    return exclusive(folder/"fixture.json",pack(plan))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: projection_language_fixture.py NEW_DIRECTORY")
    print(prepare(Path(sys.argv[1])))
