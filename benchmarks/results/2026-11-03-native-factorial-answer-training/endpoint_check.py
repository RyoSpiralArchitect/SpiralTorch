"""File-only checks for the frozen single-answer, one-window-per-document SFT endpoint."""
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def nonnegative(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf


def rounded_batch_bounds(mean, targets):
    # Each answer mean has had its positive f64 token sum divided by span length.
    # Reconstructing from means can cross an f32 midpoint by a few f64 ulps. Bound
    # both nonnegative summation orders before rounding; do not compare macro NLL.
    slack = (4 * targets + 8) * math.ulp(mean)
    low = f32(max(0.0, mean - slack))
    require(math.isfinite(low), "batch NLL exceeds native f32 range")
    return low, min(f32(mean + slack), float.fromhex("0x1.fffffep127"))


@dataclass(frozen=True)
class EndpointPlan:
    stats: tuple
    context: int
    batch: int
    groups: tuple
    plan_id: str
    data_id: str | None

    @classmethod
    def build(cls, stats, context, batch, *, content=None, vocabulary=50257):
        stats = tuple(tuple(v) for v in stats)
        require(0 < len(stats) and type(context) is int and 2 <= context <= 64 and
                type(batch) is int and batch > 0, "endpoint plan bounds")
        require(all(type(key) is str and all(type(x) is int for x in (width, start, n)) and
                    0 < start < width <= context and n == width-start for key,width,start,n in stats)
                and len({v[0] for v in stats}) == len(stats), "expected one content span per short document")
        widths = [v[1] for v in stats]
        # Field order and compact UTF-8 encoding are part of the native plan ID.
        wire = dict(schema="spiraltorch.causal_document_windows.v1", policy="overlap_one_token_include_tail_per_document.v1",
                    context_length=context, document_lengths=widths,
                    windows=[dict(document=i, start=0, length=w) for i,w in enumerate(widths)],
                    total_tokens=sum(widths), dropped_tokens=0, target_tokens=sum(widths)-len(widths))
        pid = "sha256:" + hashlib.sha256(json.dumps(wire,separators=(",",":")).encode()).hexdigest()
        groups = []
        i = 0
        while i < len(widths):
            end = i + 1
            while end < min(i+batch, len(widths)) and widths[end] == widths[i]:
                end += 1
            groups.append((i,end))
            i = end
        data_id = None
        if content is not None:
            require(content["keys"] == [v[0] for v in stats], "different original content order")
            h = hashlib.sha256(b"spiraltorch.native_causal_evaluation_data.v1\0" + pid.encode() + b"\1")
            for (_,w,s,_),doc,mask in zip(stats,content["token_documents"],content["labels"],strict=True):
                require(len(doc) == w and all(type(x) is int and 0 <= x < vocabulary for x in doc)
                        and all(type(x) is int for x in mask) and mask == [-100]*s + doc[s:], "content IDs or label mask")
                h.update(struct.pack(f"<{w+1}Q",w,*doc))
                h.update(struct.pack(f"<{w}q",*mask))
            data_id = "sha256:" + h.hexdigest()
        return cls(stats, context, batch, tuple(groups), pid, data_id)

    def verify_scores(self, raw, variants):
        expected = len(self.stats)
        comp, rows = raw["comparison"], raw["answers"]
        evaluations = comp["evaluations"]
        require(raw["schema"] == "spiraltorch.native_causal_lm_answer_comparison.v1" and
                comp["schema"] == "spiraltorch.native_causal_lm_evaluation_comparison.v1" and
                raw["document_count"] == raw["scored_documents"] == len(rows) == expected and
                len(evaluations) == variants, "unexpected answer comparison")
        for i,(row,(_,width,start,n)) in enumerate(zip(rows,self.stats,strict=True)):
            require((row["document"],row["start"],row["end"]) == (i,start,width) and len(row["scores"]) == variants,
                    "content span or variant alignment")
            for cell in row["scores"]:
                require(nonnegative(cell["nll"]) and cell["nll"] <= 2**130 and
                        type(cell["top1_correct"]) is int and 0 <= cell["top1_correct"] <= n, "invalid answer cell")
        targets = sum(v[3] for v in self.stats)
        counts = dict(window_count=expected, skipped_windows=0, microbatches=len(self.groups),
                      target_tokens=targets, ignored_targets=sum(v[2]-1 for v in self.stats))
        for j,e in enumerate(evaluations):
            require(e["schema"] == "spiraltorch.native_causal_lm_evaluation.v1" and
                    e["config"] == dict(context_length=self.context,microbatch_windows=self.batch) and
                    all(type(e[k]) is int and e[k] == v for k,v in counts.items()), "native evaluation coverage")
            require(e["plan_id"] == self.plan_id and re.fullmatch("sha256:[0-9a-f]{64}",e["data_id"]) and
                    e["data_id"] == evaluations[0]["data_id"] and
                    (self.data_id is None or e["data_id"] == self.data_id), "native evaluation plan or content ID")
            low = high = 0.0
            for start,end in self.groups:
                n = sum(self.stats[i][3] for i in range(start,end))
                mean = math.fsum(rows[i]["scores"][j]["nll"] * self.stats[i][3] for i in range(start,end)) / n
                lo,hi = rounded_batch_bounds(mean,n)
                low += lo * n
                high += hi * n
            nll = e["nll"]
            require(nonnegative(nll) and low/targets <= nll <= high/targets, "token NLL contradicts answer cells")
            try:
                ppl = math.exp(nll)
            except OverflowError:
                ppl = math.inf
            actual = e["perplexity"]
            require(actual is None if math.isinf(ppl) else
                    nonnegative(actual) and math.isclose(actual,ppl,rel_tol=2e-15), "perplexity contradicts NLL")
