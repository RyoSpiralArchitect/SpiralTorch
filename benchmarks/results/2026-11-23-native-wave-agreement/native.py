"""All model operations are owned by SpiralTorch's native extension."""
import json
import math
import struct


def tiny(st):
    # A disjoint two-layer GPT-2-shaped fixture, not pretrained language evidence.
    entries, payload = {}, bytearray()
    def add(name, shape):
        salt = len(entries) + 1
        n = math.prod(shape)
        norm = "ln_" in name
        vals = [float(name.endswith(".weight")) if norm else
                (((i*i+13*i+salt*17) % 127)-63)/512. for i in range(n)]
        start = len(payload)
        payload.extend(struct.pack("<%df" % n, *vals))
        entries[name] = dict(dtype="F32", shape=shape, data_offsets=[start, len(payload)])
    add("wte.weight", [32, 16])
    add("wpe.weight", [16, 16])
    for i in range(2):
        for norm in ("ln_1", "ln_2"):
            for p in ("weight", "bias"):
                add("h.%d.%s.%s" % (i, norm, p), [16])
        for proj, a, b in (("attn.c_attn",16,48), ("attn.c_proj",16,16),
                           ("mlp.c_fc",16,64), ("mlp.c_proj",64,16)):
            add("h.%d.%s.weight" % (i, proj), [a,b])
            add("h.%d.%s.bias" % (i, proj), [b])
    for p in ("weight", "bias"):
        add("ln_f."+p, [16])
    header = json.dumps(entries, separators=(",", ":")).encode()
    blob = struct.pack("<Q", len(header)) + header + payload
    cfg = dict(model_type="gpt2", vocab_size=32, n_positions=16, n_embd=16, n_head=2, n_layer=2)
    return st.NativeCausalLm.from_gpt2(json.dumps(cfg), st.SafeTensorArchive.from_bytes(blob))


def geometry(st, base, kind, shape, strength):
    layers, width, _, _ = shape
    recipe = dict(features=width, strength=strength)
    params = [dict(name=name, shape=[width], role="parameter", requires_grad=True,
                   values=[((i % n)-n//2)/div for i in range(width)])
              for name, n, div in (("gate",11,16.), ("bias",7,128.))]
    if kind == "wave":
        recipe.update(schema="spiraltorch.wave_gate_adapter.v2", learnable_radius=False,
                      kernel=dict(curvature=-1., saturation=1., porosity=.05, max_values=8*width))
        params.append(dict(name="log_radius", shape=[], role="buffer", requires_grad=False, values=[0.]))
    elif kind == "affine":
        recipe.update(schema="spiraltorch.channel_affine_adapter.v1", max_values=8*width)
    else:
        raise ValueError("unknown geometry")
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps(dict(
        schema="spiraltorch.geometry_adapter_artifact.v1", base_model_identity=base.base_model_identity,
        policy="full_prefix_f32_no_cache.v1", training=True, placements=[dict(
            path="transformer.h.%d.attn" % (layers-1), training=True, recipe=recipe, tensors=params)])))
    return base.with_geometry(artifact)


def train_one(model, examples, order, lr, frozen):
    history = []
    for i in order:
        row = examples[i]
        window = row["prefix"] + [row["gold"]]
        before = model.geometry_artifact_id
        grad = model.geometry_loss_gradients(window, [1,len(window)], labels=[-100]*(len(window)-1)+[row["gold"]])
        if (grad.target_tokens != 1 or not math.isfinite(grad.loss) or
                grad.geometry_artifact_id != before or grad.base_model_identity != model.base_model_identity):
            raise ValueError("wrong objective, geometry or gradient")
        if not frozen:
            model = model.step_geometry_sgd(grad, lr)
        history.append(dict(key=row["key"], ce=grad.loss, before=before, after=model.geometry_artifact_id))
    return model, history
