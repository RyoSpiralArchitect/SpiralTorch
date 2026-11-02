"""Prompt-only greedy generation and an independent, delimiter-aware line scorer."""
import json
import math


def greedy_completions(st, model, prompts, *, max_new_tokens, stop_ids, max_cache_values):
    """No reference, target length or vocabulary of permitted answers enters the decoder."""
    config = json.dumps(dict(max_new_tokens=max_new_tokens, eos_token_ids=stop_ids,
                             decoding={"selection":"greedy"}, execution={"kind":"kv_cache","max_values":max_cache_values}))
    for ids in prompts:
        session = st.NativeCausalLmGenerator(model, ids, config)
        while not session.finished:
            if session.step() is None:
                raise ValueError("generator stopped without a finish reason")
        actual = session.generated_ids
        if not 1 <= len(actual) <= max_new_tokens or session.token_ids != list(ids) + actual:
            raise ValueError("invalid generation history")
        if any(t in stop_ids for t in actual[:-1]):
            raise ValueError("decoder continued past a stop token")
        reason = "eos" if actual[-1] in stop_ids else "length"
        if session.finish_reason != reason or (reason == "length" and len(actual) != max_new_tokens):
            raise ValueError("incorrect generation boundary")
        yield dict(generated_ids=actual, finish_reason=reason)


def score_line(decoded, reference):
    """Score emitted text, not a canonical tokenization. Neither trim nor complete an unterminated line."""
    if not reference or "\n" in reference or "\r" in reference:
        raise ValueError("expected one nonempty reference line without a delimiter")
    first, lf, _ = decoded.partition("\n")
    return dict(exact=bool(lf) and first == reference, terminated=bool(lf), prefix_exact=decoded.startswith(reference))


def paired_lines(left, right):
    if not left or len(left) != len(right) or any(type(v) is not bool for v in [*left,*right]):
        raise ValueError("strict matched booleans required")
    pairs = list(zip(left, right, strict=True))
    return dict(reference=sum(left), candidate=sum(right), gained=sum(not a and b for a,b in pairs),
                lost=sum(a and not b for a,b in pairs), delta=sum(right)-sum(left))


def factorial_effects(affine, saturation, projection, both):
    """Balanced 2x2 contrasts, not four independent samples. Preserve direction and interaction."""
    if not all(map(math.isfinite, (affine,saturation,projection,both))):
        raise ValueError("nonfinite factorial response")
    return dict(saturation_without_projection=saturation-affine, saturation_with_projection=both-projection,
                projection_without_saturation=projection-affine, projection_with_saturation=both-saturation,
                interaction=(both-saturation)-(projection-affine))
