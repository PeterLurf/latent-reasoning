"""STEP-TOLD $0 LOCAL pre-gate CD-local (ST_v1 §2 at da924dca; GPU, local RTX 4060): CUED-DIRECT at d <= 2, 8k. One pass on the plain trunk with every cue
LAT-ST sees, in order (ST_draft §5): dec = [EOT] + enc(" " + text + sum_{t=1..d} " " + QUESTION(u, S_t)) + enc(" The answer is:"), target v_d (the answer).
§228 P-DIR2's recipe otherwise (boxes/inject_ow.py --arm direct, imported read-only, no file edited; items, d-sequence U{1, 2} seed + 22331, EXCL, lr,
bs, steps as P-DIR2). TEST2 (d 1..12) gets the same cues (S_1..S_d).
Pre-stated rule (ST_v1 §2): CUED-DIRECT d = 2 >= 0.80 -> the plan alone solves d = 2 one-pass -> ST stops (a finding). Theory scores it.
After training: scratch/gate_s233/cd_local.json (TEST2 acc by d 1, 2 + n) - written, not read, by Experiments.
    python -B docs/research_steptold_2026-10-24/st_cd_local.py --d_max 2 --steps 8000 --auto_micro --mem_budget 7.0 --device cuda:0 --out boxes/runs_fsq/s233_cdloc_s1
    python -B docs/research_steptold_2026-10-24/st_cd_local.py --fixtures      (decoded fixtures -> scratch/gate_s233/cd_fixtures.json)"""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT  # noqa: E402

_ORIG = IO.tok_ow
GATE = os.path.join(ROOT, "scratch", "gate_s233")


def tok_cd(it, aux_target="value"):
    """§228's tok_ow, then think / dec with the cues S_1..S_d appended (APP, in order); target = the answer (unchanged)."""
    o = _ORIG(it, aux_target); sp = it["names"]["spine"]
    cues = "".join(" " + OW.QUESTION.format(u=it["unit"], A=sp[t]) for t in range(1, it["d"] + 1))
    th = [EOT] + IO.ENC.encode(" " + it["text"] + cues); o["think"] = th; o["dec"] = th + IO.SFX; assert o["ans"] == int(it["values"][it["d"]]); return o


def fixtures():
    s = OW.Stream(22997, frames=OW.FRAME_TRAIN, **IO.OWKW); out = {}
    for d in (1, 2, 3):
        out[str(d)] = [{"decoded_dec": IO.ENC.decode(tok_cd(it)["dec"][1:]), "target": int(it["answer"]), "spine": it["names"]["spine"]} for it in (s.draw(d) for _ in range(5))]
    os.makedirs(GATE, exist_ok=True); json.dump(out, open(os.path.join(GATE, "cd_fixtures.json"), "w", newline="\n"), indent=1); print("fixtures", {k: len(v) for k, v in out.items()})


def summary(out):
    e = json.load(open(os.path.join(out, "eval_acc.json"))); res = {"by_d": {d: e["by_d"][d] for d in ("1", "2")}, "n": {d: len(e["items"][d]["ans"]) for d in ("1", "2")},
                                                                "rule": "CUED-DIRECT d = 2 >= 0.80 -> ST stops (ST_v1 §2)", "run": out}
    os.makedirs(GATE, exist_ok=True); json.dump(res, open(os.path.join(GATE, "cd_local.json"), "w", newline="\n"), indent=1); print("SUMMARY written", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--fixtures" in argv:
        fixtures(); return 0
    assert "--arm" not in argv, "CUED-DIRECT is the direct arm only"
    out = argv[argv.index("--out") + 1]; os.makedirs(out, exist_ok=True)
    meta = {"check": "STEP-TOLD CD-local (ST_v1 §2)", "wrapper_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
            "input": "dec = [EOT] + enc(' ' + text + sum_t ' ' + QUESTION(u, S_t), t = 1..d) + ' The answer is:'; target v_d"}
    json.dump(meta, open(os.path.join(out, "cd_meta.json"), "w", newline="\n"), indent=1)
    IO.tok_ow = tok_cd
    rc = IO.main(["--arm", "direct"] + argv)
    if rc == 0 and os.path.exists(os.path.join(out, "eval_acc.json")):
        summary(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
