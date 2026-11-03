"""§261 TARGET-CUE: --cue target (each edit asserted once; the step / generic paths are textually unchanged apart from widened tuples).
  target cue = TGT_FMT.format(u = the unit, X = S_d = names.spine[-1]) = " To find how many <u> <S_d> has, what is the next value?" (+ SFX ' The answer is:' by write_src answer),
  byte-identical at every pass; o['cue'] = 'target'.
  scratch/s234_build/inject_aw_head.py  TGT_FMT / tgt_cue; tok_aw's target branch (text prefix == think, tail == enc(tgt) asserted); write_src answer accepts target.
  E268 (main(), applied by mk_inject_aw.py after E258M): --cue choices + target; the cue-mode asserts accept target; the args 'format' string for target.
  scratch/s234_build/inject_aw_eval_parts.py  passes()' identical-input assert also for target; rh_input's target branch; run() drops cue_only / cue_wrong for target.
then regenerates boxes/inject_aw.py / boxes/inject_aw_eval.py.
    python -B scratch/s268_build/edits_s268.py"""
import subprocess
import sys


def edit(p, E):
    s = open(p, encoding="utf-8").read()
    for o, n in E:
        assert s.count(o) == 1, ("§261 edit", p, s.count(o), o[:90]); s = s.replace(o, n)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


h = "scratch/s234_build/inject_aw_head.py"; assert "TGT_FMT" not in open(h, encoding="utf-8").read(), "already applied"
edit(h, [
    ('GEN_CUE, GEN_LEN = " What is the next value?", 6',
     'TGT_FMT = " To find how many {u} {X} has, what is the next value?"            # §261 TC: the target cue (S_d = the asked entity, u = the unit), every pass\n\n\n'
     'def tgt_cue(raw):\n    return TGT_FMT.format(u=raw["unit"], X=raw["names"]["spine"][-1])\n\n\n'
     'GEN_CUE, GEN_LEN = " What is the next value?", 6'),
    ('    o = tok_st(it, aux_target, "step" if cue == "generic" else cue)\n',
     '    o = tok_st(it, aux_target, "step" if cue in ("generic", "target") else cue)\n'),
    ('        assert all(q[:len(o["think"])] == o["think"] and q[len(o["think"]):] == gt for q in o["think_cues"]), "GEN-CUE: the text rows precede the cue"\n',
     '        assert all(q[:len(o["think"])] == o["think"] and q[len(o["think"]):] == gt for q in o["think_cues"]), "GEN-CUE: the text rows precede the cue"\n'
     '    if cue == "target":                                                     # §261 TC: the target cue at every pass (byte-identical)\n'
     '        tt = ENC.encode(tgt_cue(it)); o["think_cues"] = [[EOT] + ENC.encode(" " + it["text"] + tgt_cue(it)) for _ in o["think_cues"]]; o["cue"] = "target"\n'
     '        assert all(q[:len(o["think"])] == o["think"] and q[len(o["think"]):] == tt for q in o["think_cues"]), "TGT-CUE: the text rows precede the cue"\n'),
    ('        assert cue in ("step", "generic") and "think_cues" in o, "AW: --write_src answer needs --cue step / generic"',
     '        assert cue in ("step", "generic", "target") and "think_cues" in o, "AW: --write_src answer needs --cue step / generic / target"')])
ARGS = 'ap.add_argument("--cue", default="none", choices=["none", "step", "all", "generic"],'
FMT = "(CUE_FMT[a.cue] if a.cue != \"generic\" else \" | §252 GP: think_t = [EOT] + enc(' ' + text + ' What is the next value?') (every pass)\")"
E268 = [(ARGS, 'ap.add_argument("--cue", default="none", choices=["none", "step", "all", "generic", "target"],'),
        ('assert a.cue not in ("step", "generic") or (a.arm == "inject"', 'assert a.cue not in ("step", "generic", "target") or (a.arm == "inject"'),
        ('assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic"))', 'assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic", "target"))'),
        ('assert not HOLD or (a.arm == "inject" and a.cue in ("step", "generic") and', 'assert not HOLD or (a.arm == "inject" and a.cue in ("step", "generic", "target") and'),
        (FMT, "(CUE_FMT[a.cue] if a.cue not in (\"generic\", \"target\") else {\"generic\": \" | §252 GP: think_t = [EOT] + enc(' ' + text + ' What is the next value?') (every pass)\", "
              "\"target\": \" | §261 TC: think_t = [EOT] + enc(' ' + text + ' To find how many <u> <S_d> has, what is the next value?') (every pass)\"}[a.cue])")]
open("scratch/s268_build/edits_s268m.py", "w", encoding="utf-8", newline="\n").write('"""§261 TC: inject_aw main() edits (applied by mk_inject_aw.py after E258M)."""\nE268 = %r\n' % (E268,))
m = "scratch/s234_build/mk_inject_aw.py"; t = open(m, encoding="utf-8").read(); a = 'assert "tok_st(" not in main and "inject_st_eval" not in main'; assert t.count(a) == 1 and "E268" not in t
open(m, "w", encoding="utf-8", newline="\n").write(t.replace(a, 'for o, n in runpy.run_path("scratch/s268_build/edits_s268m.py")["E268"]:   # §261 TC: --cue target\n'
                                                              '    assert main.count(o) == 1, ("§261 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)\n' + a, 1))
edit("scratch/s234_build/inject_aw_eval_parts.py", [
    ('        if seqs_of_t is seq_std and any(it.get("cue") == "generic" for it in ch):',
     '        if seqs_of_t is seq_std and any(it.get("cue") in ("generic", "target") for it in ch):   # §261: the target cue too'),
    ('            assert all(it.get("cue") == "generic" for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), "PLAN-FREE: generic-cue pass inputs differ across t"',
     '            assert all(it.get("cue") in ("generic", "target") for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), "PLAN-FREE: generic / target-cue pass inputs differ across t"'),
    ('        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.GEN_CUE) + IO.SFX\n',
     '        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.GEN_CUE) + IO.SFX\n'
     '    if it.get("cue") == "target":                                           # §261 TC: the target cue at every pass\n'
     '        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.tgt_cue(raw)) + IO.SFX\n'),
    ('    if a.get("cue") == "generic":   # §252 GP (REVIEW_NS #3): CUE-ONLY / CUE-WRONG are named-cue builders - dropped',
     '    if a.get("cue") in ("generic", "target"):   # §252 GP (REVIEW_NS #3) / §261 TC: CUE-ONLY / CUE-WRONG are named-cue builders - dropped')])
for g_ in ("scratch/s234_build/mk_inject_aw.py", "scratch/s234_build/mk_inject_aw_eval.py"):
    assert subprocess.call([sys.executable, "-B", g_]) == 0, g_
print("§261 TC edits applied")
