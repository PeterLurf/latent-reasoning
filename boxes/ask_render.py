"""§262 NK_v1 §2 one-hop co-training: ST_v1 P1-ASK+NAME's class rule, s1_sentence / ask_of copied byte for byte from
docs/research_steptold_2026-10-24/st_ask_name_local.py (the docs file is not shipped to hosts; scratch/s269_build/gate_ask_equiv.py asserts the copy is identical
and gives the same (class, X, v, text) on 4,000 stream items). Class per item: sha1(sig + "|askname") mod (4 if d == 2 else 3):
  (i) S_1 -> v_1; (ii) S_0 -> v_0; (iii) a distractor (else a given G_t, else S_0) -> its value; (iv) d 2 only: S_2 with S_1's defining sentence replaced by
  "S_1 has v_1 u." -> v_2."""
import hashlib
import re

import boxes.ow_words as OW


def s1_sentence(it):
    """(start, end) of S_1's defining sentence: names S_1 and S_0, no later spine name, not the question (st_cue_check C6's rule; asserted unique)."""
    text = it["text"]; sp = it["names"]["spine"]; later = sp[2:]
    has = lambda n, a, b: re.search(r"\b%s\b" % re.escape(n), text[a:b]) is not None  # noqa: E731
    cand = [(m.start(), m.end()) for m in re.finditer(r"[^.?]*[.?]", text)]
    cand = [(a, b) for a, b in cand if has(sp[1], a, b) and has(sp[0], a, b) and not any(has(n, a, b) for n in later) and "How many" not in text[a:b]]
    assert len(cand) == 1, ("S_1's defining sentence not unique", it["sig"], len(cand)); return cand[0]


def ask_of(it):
    """-> (class, X name, target value, text used, fallback note)."""
    d = it["d"]; sp = it["names"]["spine"]; ncls = 4 if d == 2 else 3
    c = int(hashlib.sha1((it["sig"] + "|askname").encode()).hexdigest()[:8], 16) % ncls; text = it["text"]
    if c == 0:
        return "i", sp[1], int(it["values"][1]), text, None
    if c == 1:
        return "ii", sp[0], int(it["values"][0]), text, None
    if c == 2:
        dn = it["names"]["distractors"]
        if dn:
            return "iii", dn[0], int(it["distractor_values"][0]), text, "distractor"
        gv = it.get("givens") or []
        if it["names"]["givens"] and gv:
            g0 = gv[0]; return "iii", g0["name"] if isinstance(g0, dict) else it["names"]["givens"][0], int(g0["v"] if isinstance(g0, dict) else g0), text, "given_G"
        return "iii", sp[0], int(it["values"][0]), text, "S_0"
    a, b = s1_sentence(it); lead = " " if a > 0 and text[a] == " " else ""
    t2 = text[:a] + lead + OW.GIVEN.format(A=sp[1], v=it["values"][1], u=it["unit"]) + text[b:]
    return "iv", sp[2], int(it["values"][2]), t2, None
