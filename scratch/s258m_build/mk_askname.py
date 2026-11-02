"""§258 S1 ASKNAME-M (LB2_v1 §2 + Note 258-1 condition 2): scratch/s258_askname.py = docs/research_steptold_2026-10-24/st_ask_name_local.py (the s233 recipe's wrapper, docs/
is not shipped) with two edits (each asserted once): ROOT from scratch/ (one level up), and summary() ALSO writes <out>/eval_st_classes.json = {"pooled": {cls: {"acc", "n"}},
"by_d": {d: {cls: {"acc", "n"}}}} (ST_v1's classes on TEST2 d 1 / d 2 from the run's eval_acc.json, pooled over d 1-2 as the wrapper's rule states) - score_lb2's S1 field.
--base passes through to inject_ow.main (argv).
    python -B scratch/s258m_build/mk_askname.py"""
s = open("docs/research_steptold_2026-10-24/st_ask_name_local.py", encoding="utf-8").read()
E = [('ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))',
      'ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # §258: scratch/ copy (docs/ is not shipped)'),
     ('    os.makedirs(GATE, exist_ok=True); json.dump(res, open(os.path.join(GATE, "p1_ask_name.json"), "w", newline="\\n"), indent=1); print("SUMMARY written", flush=True)',
      '    os.makedirs(GATE, exist_ok=True); json.dump(res, open(os.path.join(GATE, "p1_ask_name.json"), "w", newline="\\n"), indent=1); print("SUMMARY written", flush=True)\n'
      '    pooled = {}                                                             # §258 S1 (Note 258-1 (2)): eval_st_classes.json in the run dir, classes pooled over d 1-2\n'
      '    for d in ("1", "2"):\n'
      '        for c, r in res["by_d_class"][d].items():\n'
      '            if c in ("i", "ii", "iii", "iv"):\n'
      '                p = pooled.setdefault(c, [0.0, 0]); p[0] += r["acc"] * r["n"]; p[1] += r["n"]\n'
      '    st = {"pooled": {c: {"acc": v[0] / v[1], "n": v[1]} for c, v in sorted(pooled.items())}, "by_d": {d: {c: r for c, r in res["by_d_class"][d].items() if c in ("i", "ii", "iii", "iv")}\n'
      '                                                                       for d in ("1", "2")}, "rule": "ST_v1 P1-ASK+NAME classes (ask_of) on TEST2 d 1 / d 2, pooled"}\n'
      '    json.dump(st, open(os.path.join(out, "eval_st_classes.json"), "w", newline="\\n"), indent=1)')]
for o, n in E:
    assert s.count(o) == 1, ("askname copy edit", s.count(o), o[:70]); s = s.replace(o, n)
s = s.replace('"""STEP-TOLD $0 LOCAL gate P1-ASK+NAME', '"""§258 S1 copy (scratch/s258m_build/mk_askname.py: ROOT from scratch/, + eval_st_classes.json). STEP-TOLD $0 LOCAL gate P1-ASK+NAME', 1)
open("scratch/s258_askname.py", "w", encoding="utf-8", newline="\n").write(s); print("scratch/s258_askname.py written")
