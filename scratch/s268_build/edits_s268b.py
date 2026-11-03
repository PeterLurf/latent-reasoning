"""§261 TC, part 2 (edits_s268.py stopped at the eval edits: passes()' PLAN-FREE assert and run()'s cue drop live as string literals in mk_inject_aw_eval.py, not in
inject_aw_eval_parts.py; the head / mk_inject_aw E268 edits had landed). Each edit asserted once; then regenerates boxes/inject_aw.py / boxes/inject_aw_eval.py.
    python -B scratch/s268_build/edits_s268b.py"""
import subprocess
import sys

sys.path.insert(0, "scratch/s268_build")
from edits_s268_fn import edit  # noqa: E402

assert "tgt_cue" not in open("scratch/s234_build/inject_aw_eval_parts.py", encoding="utf-8").read(), "already applied"
edit("scratch/s234_build/inject_aw_eval_parts.py", [
    ('        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.GEN_CUE) + IO.SFX\n',
     '        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.GEN_CUE) + IO.SFX\n'
     '    if it.get("cue") == "target":                                           # §261 TC: the target cue at every pass\n'
     '        return [EOT] + IO.ENC.encode(" " + raw["text"] + aug + AW.tgt_cue(raw)) + IO.SFX\n')])
edit("scratch/s234_build/mk_inject_aw_eval.py", [
    ('        if seqs_of_t is seq_std and any(it.get("cue") == "generic" for it in ch):   # §252 PLAN-FREE (NS_v1 §3): GP items read one identical input at every pass',
     '        if seqs_of_t is seq_std and any(it.get("cue") in ("generic", "target") for it in ch):   # §252 PLAN-FREE (NS_v1 §3) / §261 TC: one identical input at every pass'),
    ('            assert all(it.get("cue") == "generic" for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), "PLAN-FREE: generic-cue pass inputs differ across t"',
     '            assert all(it.get("cue") in ("generic", "target") for it in ch) and all(torch.equal(x_, xs[0]) for x_ in xs), "PLAN-FREE: generic / target-cue pass inputs differ across t"'),
    ('    if a.get("cue") == "generic":   # §252 GP (REVIEW_NS #3): CUE-ONLY / CUE-WRONG are named-cue builders - dropped',
     '    if a.get("cue") in ("generic", "target"):   # §252 GP (REVIEW_NS #3) / §261 TC: CUE-ONLY / CUE-WRONG are named-cue builders - dropped')])
for g_ in ("scratch/s234_build/mk_inject_aw.py", "scratch/s234_build/mk_inject_aw_eval.py"):
    assert subprocess.call([sys.executable, "-B", g_]) == 0, g_
print("§261 TC edits (part 2) applied")
