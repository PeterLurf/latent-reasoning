"""§264 SCORECARD-CHECK + §265 DENSE-DISTRACT build (each edit asserted once; flags off = the pre-edit computation):
  E271  (inject_aw main(), applied by mk_inject_aw.py after E269): --distract_nmax (default 2: IR.distract / distract_mix draw rng.integers(0, 3), bitwise the pre-edit draw);
        the memory-check items carry max(2, nmax) insert pairs (the d_max + nmax-insert shape); the bin logs the capped draws (distract_capped) next to distract_n.
  E271E (inject_aw_eval run(), applied by mk_inject_aw_eval.py after E270E): the parts cue_clock / op_count / attn_fade (boxes/sc_parts.py).
then regenerates boxes/inject_aw.py / boxes/inject_aw_eval.py.  (boxes/insert_render.py's nmax / n / name budget: scratch/s271_build/edits_ir.py.)
    python -B scratch/s271_build/edits_s271.py"""
import subprocess
import sys

E271 = [
    ('    ap.add_argument("--distract_p", type=float, default=0.0, help="Note 255-5: P(an item gets U{0,1,2} irrelevant new-name op sentences) (0 = off, unchanged)")\n',
     '    ap.add_argument("--distract_p", type=float, default=0.0, help="Note 255-5: P(an item gets U{0,1,2} irrelevant new-name op sentences) (0 = off, unchanged)")\n'
     '    ap.add_argument("--distract_nmax", type=int, default=2, help="§265: inserts n ~ U{0..nmax} (default 2 = the pre-§265 draw, bitwise); capped at the 40-name budget")\n'),
    ('    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin\n',
     '    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin\n'
     '    DCAP = {}                                                               # §265: {drawn n: count} of the name-budget-capped draws in the open 1k bin\n'),
    ('IR.distract_mix(r_, a.seed, step, k_, MIXT) if MIXT else IR.distract(r_, a.seed, step, k_)   # §257 --distract_mix',
     'IR.distract_mix(r_, a.seed, step, k_, MIXT, nmax=a.distract_nmax) if MIXT else IR.distract(r_, a.seed, step, k_, nmax=a.distract_nmax)   # §257 --distract_mix; §265 nmax'),
    ('                DSTAT[n_] = DSTAT.get(n_, 0) + 1; out_.append(r_)\n',
     '                DSTAT[n_] = DSTAT.get(n_, 0) + 1; out_.append(r_)\n'
     '                if r_.get("distract_capped") is not None:                       # §265: a draw the name budget capped\n'
     '                    DCAP[r_["distract_capped"]] = DCAP.get(r_["distract_capped"], 0) + 1\n'),
    ('IR.draw_pairs(r_, 2, np.random.default_rng((a.seed, IR.SEED_DISTRACT + 2, i_)))',
     'IR.draw_pairs(r_, max(2, a.distract_nmax), np.random.default_rng((a.seed, IR.SEED_DISTRACT + 2, i_)))   # §265: the nmax-insert shape'),
    ('                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()\n',
     '                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()\n'
     '                if DCAP:                                                    # §265: the capped draws (drawn n: count)\n'
     '                    br["distract_capped"] = {str(k_): v_ for k_, v_ in sorted(DCAP.items())}; DCAP.clear()\n')]
NEW = ("cue_clock", "op_count", "attn_fade")
E271E = [
    ('"late_premise", "name_swap", "shuffle_graded")\nAW_ONLY = (', '"late_premise", "name_swap", "shuffle_graded", %s)\nAW_ONLY = (' % ", ".join('"%s"' % p for p in NEW)),
    ('"frames", "nk_pilot", "late_premise", "name_swap", "shuffle_graded")\n', '"frames", "nk_pilot", "late_premise", "name_swap", "shuffle_graded", %s)\n' % ", ".join('"%s"' % p for p in NEW)),
    ('.part_shuffle_graded(m, g, sets, a, dev, bs)}[p_]()   # §263 ORDER-FREE',
     '.part_shuffle_graded(m, g, sets, a, dev, bs),   # §263 ORDER-FREE\n'
     + "".join('              "%s": lambda: __import__("boxes.sc_parts", fromlist=["part_%s"]).part_%s(m, g, sets, a, dev, bs),\n' % (p, p, p) for p in NEW[:-1])
     + '              "%s": lambda: __import__("boxes.sc_parts", fromlist=["part_%s"]).part_%s(m, g, sets, a, dev, bs)}[p_]()   # §264 / §265' % (NEW[-1], NEW[-1], NEW[-1]))]
open("scratch/s271_build/edits_s271m.py", "w", encoding="utf-8", newline="\n").write('"""§265: inject_aw main() edits (applied by mk_inject_aw.py after E269)."""\nE271 = %r\n' % (E271,))
open("scratch/s271_build/edits_s271e.py", "w", encoding="utf-8", newline="\n").write('"""§264 / §265: inject_aw_eval run() edits (applied by mk_inject_aw_eval.py after E270E)."""\nE271E = %r\n' % (E271E,))
m = "scratch/s234_build/mk_inject_aw.py"; t = open(m, encoding="utf-8").read(); a_ = 'assert "tok_st(" not in main and "inject_st_eval" not in main'; assert t.count(a_) == 1 and "E271" not in t
open(m, "w", encoding="utf-8", newline="\n").write(t.replace(a_, 'for o, n in runpy.run_path("scratch/s271_build/edits_s271m.py")["E271"]:   # §265: --distract_nmax\n'
                                                               '    assert main.count(o) == 1, ("§265 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)\n' + a_, 1))
me = "scratch/s234_build/mk_inject_aw_eval.py"; t = open(me, encoding="utf-8").read(); b_ = 'parts = open("scratch/s234_build/inject_aw_eval_parts.py", encoding="utf-8").read()\n'
assert t.count(b_) == 1 and "E271E" not in t
open(me, "w", encoding="utf-8", newline="\n").write(t.replace(b_, 'for o, n in __import__("runpy").run_path("scratch/s271_build/edits_s271e.py")["E271E"]:   # §264 / §265 parts\n'
                                                                 '    assert body.count(o) == 1, ("§264 eval edit not applied exactly once", body.count(o), o[:90]); body = body.replace(o, n)\n' + b_, 1))
for g_ in ("scratch/s234_build/mk_inject_aw.py", "scratch/s234_build/mk_inject_aw_eval.py"):
    assert subprocess.call([sys.executable, "-B", g_]) == 0, g_
print("§264 / §265 edits applied")
