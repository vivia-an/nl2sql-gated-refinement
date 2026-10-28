"""M1 intent modelling: one JSON-producing call (+1 retry on parse failure, counted as a call)."""
import json

from ... import prompts_util as P
from ...util import parse_json
from .knowledge import split_evidence


def intent(llm, model, item, seed, cfg):
    """-> (graph dict | None, ok). Evidence is normalised deterministically (split on ';')."""
    ev_norm = "; ".join(split_evidence(item.evidence))
    msgs = [{"role": "system", "content": P.render("m1.system")},
            {"role": "user", "content": P.render("m1.user", db_id=item.db_id, question=item.question, evidence=ev_norm or "(none)")}]
    mt = cfg["max_tokens"]["m1"]
    r = llm.chat(model, msgs, temperature=0.0, seed=seed, max_tokens=mt, tag="m1")
    g = parse_json(r["text"])
    if g is None:
        msgs2 = msgs + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": P.render("m1.retry")}]
        r = llm.chat(model, msgs2, temperature=0.0, seed=seed, max_tokens=mt, tag="m1")
        g = parse_json(r["text"])
    if g is None:
        return None, False
    g.setdefault("objects", [])
    g.setdefault("filters", [])
    return g, True


def graph_text(g):
    if not g:
        return ""
    return "Task graph: " + json.dumps(g, ensure_ascii=False, separators=(",", ":"))


def mentions(g):
    if not g:
        return []
    out = []
    for o in g.get("objects") or []:
        if isinstance(o, dict) and o.get("mention"):
            out.append(str(o["mention"]))
    for f in g.get("filters") or []:
        if isinstance(f, dict) and f.get("text"):
            out.append(str(f["text"]))
    return out
