"""STEP-TOLD $0 LOCAL check P1-ASK (GPU, local RTX 4060, ~30 min; for Experiments): can ONE pass learn the told pass-1 job?

The told pass 1 of LAT-ST sees think_1 = [EOT] + enc(" " + text + " " + QUESTION(u, S_1)) and must compute v_1 (ST_draft sec. 3, step P1). The closed
§228 models ignore the question's name (st_told_zs.json: ASK0 / REPL at chance), so this job must be learned. This wrapper trains §228's DIRECT
arm (boxes/inject_ow.py, imported read-only, no file edited) on exactly that input: every item, any d, gets the S_1 cue appended and the target v_1
(d = 1: the cue repeats the question, target = the answer). Items, d-sequence (U{1, 2}, seed + 22331), EXCL, lr, bs, steps: §228's P-DIR2. TEST2 eval
by d then reads one-hop accuracy with 0..7 downstream steps present (target v_1 at every d).
Pre-stated rule (ST_draft sec. 6.1): PASS iff TEST2 acc(d = 2) >= 0.70 (target v_1), read only if acc(d = 1) >= 0.80. If acc(d = 1) < 0.80 at 8k:
UNDERTRAINED -> one fresh run at --steps 16000 (same args otherwise; --resume would change the lr schedule), read by the same rule. FAIL -> no cloud pilot.
    python -B docs/research_steptold_2026-10-24/st_ask_local.py --d_max 2 --steps 8000 --device cuda:0 --out boxes/runs_fsq/st_ask1_s1 [inject_ow args]
"""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT  # noqa: E402

ASK_T = 1
_ORIG = IO.tok_ow


def tok_ask(it, aux_target="value"):
    """§228's tok_ow, then the think / dec inputs with the S_1 cue appended and the target v_1 (the aux list is unused by the direct arm)."""
    o = _ORIG(it, aux_target); q = it["question"]; assert it["text"].endswith(q) and q == OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][it["d"]])
    th = [EOT] + IO.ENC.encode(" " + it["text"] + " " + OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][ASK_T]))
    o["think"] = th; o["dec"] = th + IO.SFX; o["ans"] = int(it["values"][ASK_T]); return o


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    assert "--arm" not in argv, "P1-ASK is the direct arm only"
    out = argv[argv.index("--out") + 1]; os.makedirs(out, exist_ok=True)
    meta = {"check": "STEP-TOLD P1-ASK", "ask_t": ASK_T, "wrapper_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
            "input": "think = [EOT] + enc(' ' + text + ' ' + QUESTION(u, S_1)); dec = think + ' The answer is:'; target v_1 (every d)"}
    with open(os.path.join(out, "ask_meta.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1); f.write("\n")
    IO.tok_ow = tok_ask                                                     # main() / the eval's tokd() look tok_ow up at call time
    return IO.main(["--arm", "direct"] + argv)


if __name__ == "__main__":
    sys.exit(main())
