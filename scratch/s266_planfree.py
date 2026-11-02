"""§260 GENERIC-BUN full stage: queue line after the TRAIN + long lines (the runner appends --device <d> --out <dir>; <dir> = the run dir):
  <dir> scratch.s266_planfree
asserts the run's args cue == generic, then on 50 REG TEST2 items per d 1..4 under the run's own tokenization (inject_aw_eval.tokd): (a) every pass input
seq_std(it, t), t = 1..think_k, is token-identical to t = 1 (the plan-free cue at every pass), and (b) inject_aw_eval.passes() runs with its §252 PLAN-FREE assert
armed; (c) the cue part after the item text names none of the item's entities (Note 260-3). -> <dir>/gates_pass.json {"plan_free": true, ...} only if all hold
(else no file, exit 1); host_shas.json refreshed (s257_guarded.manifest)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402


def main():
    import torch
    from boxes import inject_aw_eval as EV
    from boxes import inject_ow as IO
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out")
    a, m, g = EV.load(out, dev); assert a.get("cue") == "generic", ("plan-free gate: the run's cue", a.get("cue")); K = a["think_k"]
    import re
    from boxes.inject_ow import ENC
    sets = IO.build_sets(mul_s_p=EV.law(a)); T = EV.tokd({"test2": {d: sets["test2"][d][:50] for d in (1, 2, 3, 4)}}, a, (1, 2, 3, 4)); same = True; noname = True; n = 0
    for d in (1, 2, 3, 4):
        for it in T[d]:
            tc = it["think_cues"]; same = same and all(q == tc[0] for q in tc) and len(tc) >= 1 and it.get("cue") == "generic"   # byte-identical cue at every pass
            tail = ENC.decode(tc[0][len(it["think"]):]); nm = it["raw"]["names"]                     # the cue part after the item text (GEN_CUE + ' The answer is:')
            noname = noname and not any(re.search(r"\b%s\b" % re.escape(x), tail) for x in list(nm["spine"]) + list(nm["givens"]) + list(nm["distractors"])); n += 1
        with torch.no_grad():
            EV.passes(m, T[d], dev, 50, {K})                                # raises on the PLAN-FREE assert if any batch's pass inputs differ
    o = {"plan_free": bool(same and noname), "cue_identical_all_passes": bool(same), "cue_has_no_entity_name": bool(noname), "items_checked": n, "think_k": K,
         "cue": a.get("cue"), "passes_assert": "armed (inject_aw_eval.passes, §252)"}
    same = same and noname
    if same:
        json.dump(o, open(out + "/gates_pass.json", "w"))
    SG.manifest(out); print("PLAN_FREE", o, flush=True); return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
