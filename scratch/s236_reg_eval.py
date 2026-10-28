"""§236 / §238 eval-only references on the REG (--law 0) or DF (--law 2) sets (RC_v1 §1 + Note 236-2): a trained run's model.pt evaluated by boxes/inject_aw_eval.py on the REG sets
(build_sets(mul_s_p=0) + the eval module's own REG streams), written to boxes/runs_fsq/<run>_reg (the original run dir is never written).
<run>_reg/args.json = the run's args.json with mul_s_p 0, reg_of <run>, and the REG set shas / pair_stats (the eval asserts them); model.pt = a hard link to the
run's (no copy). fp32 (the eval's dtype), local card, memory-checked by the gate (eval_bs as given).
    python -B scratch/s236_reg_eval.py --run s228_pdir2c_s1 --parts acc --eval_bs 100 --device cuda:0
    python -B scratch/s236_reg_eval.py --run aw_plaw4_s1 --parts ws,acc,cue_only,swap_kd --eval_bs 250 --device cuda:0"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--src_dir", default="", help="gates only: the source run dir (default boxes/runs_fsq/<run>)")
    ap.add_argument("--out", default="", help="gates only: the output dir (default boxes/runs_fsq/<run>_reg)")
    ap.add_argument("--set_d_max", type=int, default=0, help="§237 step-0 refs: evaluate at this d_max (aw_plaw2_s1 trained at 2 -> the P-4 parts at 4); 0 = the run's")
    ap.add_argument("--law", type=int, default=0, choices=[0, 2], help="§238: the sets' law: 0 = REG (out <run>_reg, §236), 2 = DF (out <run>_df)")
    a = ap.parse_args()
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    src = a.src_dir or os.path.join(ROOT, "boxes", "runs_fsq", a.run); out = a.out or os.path.join(ROOT, "boxes", "runs_fsq", a.run + {0: "_reg", 2: "_df"}[a.law])
    assert os.path.exists(os.path.join(src, "model.pt")) and os.path.exists(os.path.join(src, "args.json")), ("source run incomplete", src)
    os.makedirs(out, exist_ok=True); S = IO.build_sets(mul_s_p=a.law)
    ra = json.load(open(os.path.join(src, "args.json")))                    # §238: any source law (REG-trained rcb_* evaluated on the DF sets, DF-trained on REG)
    na = {**ra, "mul_s_p": a.law, "src_mul_s_p": ra.get("mul_s_p", 1), "reg_of": a.run, "reg_src_dir": os.path.relpath(src, ROOT).replace(os.sep, "/"), "pair_stats": S["pair_stats"], **S["shas"]}
    if a.set_d_max:                                                         # §237: the eval's d range only (ds = 2..d_max, swap_kd / lp5h d 3..d_max, ws d 1..d_max)
        na["reg_d_max_override"] = [ra.get("d_max"), a.set_d_max]; na["d_max"] = a.set_d_max
    json.dump(na, open(os.path.join(out, "args.json"), "w", newline="\n"), indent=1)
    mp = os.path.join(out, "model.pt")
    if not os.path.exists(mp):
        os.link(os.path.join(src, "model.pt"), mp)
    if str(a.device).startswith("cuda"):
        torch.zeros(1, device=a.device)
    AE.run(out, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs, sets=S)
    ev = json.load(open(os.path.join(out, "eval.json"))); ev["reg_of"] = a.run
    ev["reg_note"] = "eval-only %s reference (§236 / §238); train = the source run's" % {0: "REG", 2: "DF"}[a.law]
    json.dump(ev, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1); print("REG-EVAL DONE", out, flush=True)


if __name__ == "__main__":
    main()
