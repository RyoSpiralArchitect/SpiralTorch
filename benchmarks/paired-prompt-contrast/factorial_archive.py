"""Authenticate and rejoin a local diagonal journal. This module never loads model weights."""
from hashlib import sha256
import io
import json
import math

import factorial as grid
import matrix as m


def pinned_bytes(path, pin, limit):
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    m.check(len(raw) <= limit and sha256(raw).hexdigest() == pin, f"unexpected archive member {path.name}")
    return raw


def load_cached(root, completion_pin, pairs, inputs, models, wheel, native, *, base_id, archive_id):
    """The caller must supply independent source pins and the expected ordered checkpoint descriptors."""
    done = json.loads(pinned_bytes(root / "complete.json", completion_pin, 8192))
    attempt = json.loads(pinned_bytes(root / "attempt.json", done["attempt"], 65536))
    audit = json.loads(pinned_bytes(root / "audit.json", done["audit"], 262144))
    names = [item["name"] for item in models]
    expected = len(models) * len(pairs) * 2
    m.check(names and len(set(names)) == len(names) and set(names) == audit["models"].keys(), "invalid model roster")
    m.check(attempt["models"] == models and attempt["base"] == base_id and attempt["archive"] == archive_id
            and attempt["wheel"] == audit["wheel"] == wheel and attempt["native"] == native
            and done["source"] == audit["source"] == attempt["source"], "different model or runtime lineage")
    m.check(done["calls"] == attempt["calls"] == audit["calls"] == expected and done["attempts"] == 1 and
            done["model_updates"] == attempt["new_updates"] == audit["new_training_updates"] == 0,
            "unqualified historical read")
    legacy = [dict(key=r["key"], field=r["field"], target=r["target"], lf=r["lf"],
                   prefixes={old: r["prefixes"][cell] for old, cell in grid.CACHED.items()}) for r in inputs]
    m.check(m.digest(legacy) == attempt["matrix"] == audit["input"], "diagonal token matrix does not match parent")
    data = pinned_bytes(root / "events.jsonl", audit["trace"], 16 * 1024**2)
    stream = io.BytesIO(data)
    result, cursor = {}, 0
    for model in names:
        evidence, parent_obs = {}, []
        for pair, row in zip(pairs, inputs, strict=True):
            m.check((pair.key, pair.field) == (row["key"], row["field"]), "crossed input identity")
            def observe(form, ids, target, lf):
                nonlocal cursor
                cursor += 1
                lines = stream.readline(), stream.readline()
                m.check(all(lines), "incomplete archived model call")
                start, end = map(json.loads, lines)
                identity = (cursor, model, pair.key, pair.field, form)
                m.check(all((e["n"], e["model"], e["key"], e["field"], e["form"]) == identity for e in (start, end))
                        and (start["event"], end["event"]) == ("start", "returned"), "wrong archived call identity")
                cell = grid.CACHED[form]
                m.check((start["target"], start["lf"]) == (target, lf) and start["prefix_sha256"] == m.digest(ids),
                        "wrong archived candidate or prefix")
                raw = dict(model=model, source="cached", prefix_sha256=start["prefix_sha256"], score=end["score"])
                grid.checked_score(raw, row, cell)
                evidence[pair.key, pair.field, cell] = raw
                return raw["score"]

            parent_obs.append(m.inspect_prepared(pair, {f: row["prefixes"][c] for f,c in grid.CACHED.items()},
                                                row["target"], row["lf"], observe))
        prior = audit["models"][model]
        summary = m.summarize(pairs, parent_obs)
        counts = summary["transition_pairs"]
        m.check(math.isclose(summary["mean_record_delta"], prior["mean_delta"], rel_tol=0., abs_tol=1e-12) and
                all(sum(counts[a].values()) == prior["question"].get(a, 0) and
                    sum(counts[b][a] for b in grid.CATEGORIES) == prior["record"].get(a, 0) for a in grid.CATEGORIES),
                "rejoined diagonals disagree with archived summary")
        result[model] = evidence
    m.check(cursor == expected and not stream.read(1), "extra archived observations")
    return result
