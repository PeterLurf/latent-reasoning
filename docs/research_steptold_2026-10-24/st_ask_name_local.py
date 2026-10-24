"""STEP-TOLD $0 LOCAL gate P1-ASK+NAME (ST_v1 §2 at da924dca; REVIEW_ST fix 1; GPU, local RTX 4060). Replaces st_ask_local.py's P1-ASK (kept unchanged).
§228 P-DIR2's recipe (boxes/inject_ow.py --arm direct, imported read-only, no file edited; items, d-sequence U{1, 2} seed + 22331, EXCL, lr, bs, steps as
P-DIR2), with every item's think input cued (APP form) "How many <u> does <X> have?" appended and the target v_X. The class of X per item is fixed by the
item's sig (sha1(sig + "|askname") mod #classes; no rng drawn, so the d-sequence and the item stream are P-DIR2's):
  (i)   S_1 (one hop)                               target v_1
  (ii)  S_0 (given)                                 target v_0
  (iii) a distractor; else a given G_t; else S_0    target its stated value (the fallback is recorded per item)
  (iv)  d = 2 only: S_2, with S_1's defining sentence (the unique sentence naming S_1 and S_0, no later spine name, not the question; asserted)
        replaced by GIVEN "S_1 has v_1 u."          target v_2
  d = 1 items draw from (i)-(iii), d = 2 items from (i)-(iv). TEST2 (d 1..12) uses the same rule; the summary reads d 1 and 2.
Pre-stated rule (ST_v1 §2 / REVIEW_ST fix 1): PASS iff (i) >= 0.80 and (iv) >= 0.80 and (ii) >= 0.90 and (iii) >= 0.90 (union also reported), read only if
d = 1 (i) >= 0.80 (d = 1 overall also reported); UNDERTRAINED -> one fresh 16k run, same rule; FAIL -> ST stops at $0. Theory scores it.
After training: scratch/gate_s233/p1_ask_name.json (per class x d test acc, n, the per-item class / fallback list) - written, not read, by Experiments.
    python -B docs/research_steptold_2026-10-24/st_ask_name_local.py --d_max 2 --steps 8000 --auto_micro --mem_budget 7.0 --device cuda:0 --out boxes/runs_fsq/s233_askname_s1
    python -B docs/research_steptold_2026-10-24/st_ask_name_local.py --fixtures      (20 decoded fixtures per class -> scratch/gate_s233/askname_fixtures.json; no GPU)"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT  # noqa: E402

_ORIG = IO.tok_ow
GATE = os.path.join(ROOT, "scratch", "gate_s233")


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


def tok_askname(it, aux_target="value"):
    """§228's tok_ow, then think / dec = [EOT] + enc(" " + text' + " " + QUESTION(u, X)) (+ ' The answer is:'), target v_X (aux unused by the direct arm)."""
    o = _ORIG(it, aux_target); cls, X, v, text, fb = ask_of(it)
    th = [EOT] + IO.ENC.encode(" " + text + " " + OW.QUESTION.format(u=it["unit"], A=X)); o["think"] = th; o["dec"] = th + IO.SFX; o["ans"] = v
    o["ask_class"] = cls; o["ask_fallback"] = fb; return o


def fixtures():
    s = OW.Stream(22997, frames=OW.FRAME_TRAIN, **IO.OWKW); out = {}; seen = {}
    for d in (1, 2):
        for _ in range(4000):
            it = s.draw(d); o = tok_askname(it); k = (o["ask_class"], d)
            if len(seen.setdefault(k, [])) < 20:
                seen[k].append({"d": d, "class": o["ask_class"], "fallback": o["ask_fallback"], "target": o["ans"], "decoded_dec": IO.ENC.decode(o["dec"][1:]),
                                "values": it["values"], "spine": it["names"]["spine"], "distractors": list(zip(it["names"]["distractors"], it["distractor_values"]))})
    out = {"%s_d%d" % k: v for k, v in sorted(seen.items())}; os.makedirs(GATE, exist_ok=True)
    json.dump(out, open(os.path.join(GATE, "askname_fixtures.json"), "w", newline="\n"), indent=1); print("fixtures", {k: len(v) for k, v in out.items()})


def summary(out):
    """per class x d (1, 2) test acc from eval_acc.json, the classes recomputed from TEST2 (the same rule)."""
    e = json.load(open(os.path.join(out, "eval_acc.json"))); S = IO.build_sets(); res = {"by_d_class": {}, "items": {}}
    for d in (1, 2):
        its = S["test2"][d]; it_ = e["items"][str(d)]; assert [x["sig"] for x in its] == it_["sig"], "TEST2 order"
        cl = [ask_of(x) for x in its]; ok = [p == a for p, a in zip(it_["pred"], it_["ans"])]
        assert [c[2] for c in cl] == it_["ans"], "targets"
        byc = {}
        for (c, _, _, _, fb), k in zip(cl, ok):
            byc.setdefault(c, []).append(k)
        res["by_d_class"][str(d)] = {c: {"acc": sum(v) / len(v), "n": len(v)} for c, v in sorted(byc.items())}; res["by_d_class"][str(d)]["all"] = {"acc": sum(ok) / len(ok), "n": len(ok)}
        g = byc.get("ii", []) + byc.get("iii", []); res["by_d_class"][str(d)]["ii_iii_union"] = {"acc": sum(g) / len(g) if g else None, "n": len(g)}
        res["items"][str(d)] = {"sig": it_["sig"], "class": [c[0] for c in cl], "fallback": [c[4] for c in cl], "target": it_["ans"], "pred": it_["pred"]}
    res["rule"] = ("PASS iff (i) >= 0.80 and (iv) >= 0.80 and (ii) >= 0.90 and (iii) >= 0.90, read only if d = 1 (i) >= 0.80 (ST_v1 §2); pooled over the "
                   "TEST2 d 1 and 2 items of each class unless Theory states otherwise")
    os.makedirs(GATE, exist_ok=True); json.dump(res, open(os.path.join(GATE, "p1_ask_name.json"), "w", newline="\n"), indent=1); print("SUMMARY written", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--fixtures" in argv:
        fixtures(); return 0
    assert "--arm" not in argv, "P1-ASK+NAME is the direct arm only"
    out = argv[argv.index("--out") + 1]; os.makedirs(out, exist_ok=True)
    meta = {"check": "STEP-TOLD P1-ASK+NAME (ST_v1 §2)", "wrapper_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
            "input": "think = [EOT] + enc(' ' + text' + ' ' + QUESTION(u, X)); dec = think + ' The answer is:'; target v_X; class by sha1(sig + '|askname')"}
    json.dump(meta, open(os.path.join(out, "ask_meta.json"), "w", newline="\n"), indent=1)
    IO.tok_ow = tok_askname                                                 # main() / the eval's tokd() look tok_ow up at call time
    rc = IO.main(["--arm", "direct"] + argv)
    if rc == 0 and os.path.exists(os.path.join(out, "eval_acc.json")):
        summary(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
