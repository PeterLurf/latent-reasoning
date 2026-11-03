"""§261 TARGET-CUE gate: queue line after the stopland line, in the run's dir (the runner appends --device <d> --out <dir>):
  <dir> scratch.s268_target
asserts the run's args cue == target, then on 50 REG TEST2 items per d 1..6 under the run's own tokenization (inject_aw_eval.tokd): (a) every pass input is token-identical
to pass 1 (the target cue at every pass) and inject_aw_eval.passes() runs with its PLAN-FREE assert armed (§252 / §261); (b) the cue tail (the tokens after the item text)
contains S_d's name (names.spine[-1]) and (c) no other entity name of the item (spine / givens / distractors) nor any other boxes.ow_words NAMES entry.
-> <dir>/gates_pass.json {"target_only": true, ...} only if all hold (else no file, exit 1); host_shas.json refreshed (s257_guarded.manifest)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import scratch.s257_guarded as SG  # noqa: E402

DS = (1, 2, 3, 4, 5, 6)


def main():
    import torch
    from boxes import inject_aw_eval as EV
    from boxes import inject_ow as IO
    from boxes import ow_words as OW
    from boxes.inject_ow import ENC
    argv = sys.argv[1:]; dev = SG.take(argv, "--device"); out = SG.take(argv, "--out")
    a, m, g = EV.load(out, dev); assert a.get("cue") == "target", ("target gate: the run's cue", a.get("cue")); K = a["think_k"]
    sets = IO.build_sets(mul_s_p=EV.law(a)); T = EV.tokd({"test2": {d: sets["test2"][d][:50] for d in DS}}, a, DS); same = has_sd = no_other = True; n = 0
    hit = lambda x, s: re.search(r"\b%s\b" % re.escape(x), s) is not None  # noqa: E731
    for d in DS:
        for it in T[d]:
            tc = it["think_cues"]; same = same and len(tc) >= 1 and all(q == tc[0] for q in tc) and it.get("cue") == "target"   # byte-identical cue at every pass
            tail = ENC.decode(tc[0][len(it["think"]):]); nm = it["raw"]["names"]; sd = nm["spine"][-1]
            others = set(nm["spine"][:-1]) | set(nm["givens"]) | set(nm["distractors"]) | set(OW.NAMES); others.discard(sd)
            has_sd = has_sd and hit(sd, tail); no_other = no_other and not any(hit(x, tail) for x in others); n += 1
        with torch.no_grad():
            EV.passes(m, T[d], dev, 50, {K})                                # raises on the PLAN-FREE assert if any batch's pass inputs differ
    ok = bool(same and has_sd and no_other)
    o = {"target_only": ok, "cue_identical_all_passes": bool(same), "cue_has_sd": bool(has_sd), "cue_has_no_other_entity": bool(no_other), "items_checked": n,
         "think_k": K, "cue": a.get("cue"), "ds": list(DS), "passes_assert": "armed (inject_aw_eval.passes, §252 / §261)"}
    if ok:
        json.dump(o, open(out + "/gates_pass.json", "w"))
    SG.manifest(out); print("TARGET_ONLY", o, flush=True); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
