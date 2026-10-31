"""§252 UNSCAFFOLDED: append parts_s252.py to the eval parts source (once); edit mk_inject_aw_eval.py (each asserted once): test_perm in PARTS / AW_ONLY / the
dispatcher; run() drops cue_only / cue_wrong for --cue generic (REVIEW_NS #3); passes() PLAN-FREE assert (NS_v1 §3).
    python -B scratch/s252_build/edit_eval_s252.py"""
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
EDITS = [
    ('"halthead", "transfer", "frames")\\nAW_ONLY = ("probe",', '"halthead", "transfer", "frames", "test_perm")\\nAW_ONLY = ("test_perm", "probe",'),
    ('"frames": lambda: part_frames(m, g, sets, a, dev, bs)}[p_]()',
     '"frames": lambda: part_frames(m, g, sets, a, dev, bs),\\n'
     '              "test_perm": lambda: part_test_perm(m, g, sets, a, dev, bs)}[p_]()'),
    ("parts = [p_ for p_ in parts if p_ not in AW_ONLY]'),",
     "parts = [p_ for p_ in parts if p_ not in AW_ONLY]\\n"
     "    if a.get(\"cue\") == \"generic\":   # §252 GP (REVIEW_NS #3): CUE-ONLY / CUE-WRONG are named-cue builders - dropped\\n"
     "        parts = [p_ for p_ in parts if p_ not in (\"cue_only\", \"cue_wrong\")]'),"),
    ("E = [\n",
     "E = [\n"
     "('        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(ch), dev)\\n',\n"
     " '        if seqs_of_t is seq_std and any(it.get(\"cue\") == \"generic\" for it in ch):   # §252 PLAN-FREE (NS_v1 §3): GP items read one identical input at every pass\\n'\n"
     " '            assert all(it.get(\"cue\") == \"generic\" for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), \"PLAN-FREE: generic-cue pass inputs differ across t\"\\n'\n"
     " '        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = m.mem.zero_state(len(ch), dev)\\n'),\n"),
]
for o, n in EDITS:
    assert s.count(o) == 1, ("§252 eval edit", s.count(o), o[:80]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "scratch/s234_build/inject_aw_eval_parts.py"; t = open(q, encoding="utf-8").read(); add = open("scratch/s252_build/parts_s252.py", encoding="utf-8").read()
assert "def part_test_perm" not in t, "parts_s252 already appended"
o = '''    raw = it["raw"]; return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + " " + ST.cue_str(raw, t)) + IO.SFX'''
n = '''    raw = it["raw"]
    if it.get("cue") == "generic":                                          # §252 GP (REVIEW_NS #3): the plan-free cue at every pass
        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.GEN_CUE) + IO.SFX
    return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + " " + ST.cue_str(raw, t)) + IO.SFX'''
assert t.count(o) == 1, "rh_input"; t = t.replace(o, n)
open(q, "w", encoding="utf-8", newline="\n").write(t + add); print("§252 eval edits applied")
