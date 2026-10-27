"""§236 REG-CHAIN eval-only references on the REG sets (RC_v1 §1 + Note 236-2): a trained run's model.pt evaluated by boxes/inject_aw_eval.py on the REG sets
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
    a = ap.parse_args()
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    src = a.src_dir or os.path.join(ROOT, "boxes", "runs_fsq", a.run); out = a.out or os.path.join(ROOT, "boxes", "runs_fsq", a.run + "_reg")
    assert os.path.exists(os.path.join(src, "model.pt")) and os.path.exists(os.path.join(src, "args.json")), ("source run incomplete", src)
    os.makedirs(out, exist_ok=True); S = IO.build_sets(mul_s_p=0)
    ra = json.load(open(os.path.join(src, "args.json"))); assert int(ra.get("mul_s_p", 1)) == 1, "the source run must be a default-law run"
    na = {**ra, "mul_s_p": 0, "reg_of": a.run, "reg_src_dir": os.path.relpath(src, ROOT).replace(os.sep, "/"), "pair_stats": S["pair_stats"], **S["shas"]}
    json.dump(na, open(os.path.join(out, "args.json"), "w", newline="\n"), indent=1)
    mp = os.path.join(out, "model.pt")
    if not os.path.exists(mp):
        os.link(os.path.join(src, "model.pt"), mp)
    if str(a.device).startswith("cuda"):
        torch.zeros(1, device=a.device)
    AE.run(out, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs, sets=S)
    ev = json.load(open(os.path.join(out, "eval.json"))); ev["reg_of"] = a.run; ev["reg_note"] = "eval-only REG reference (§236 RC_v1 §1); train = the source run's"
    json.dump(ev, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1); print("REG-EVAL DONE", out, flush=True)


if __name__ == "__main__":
    main()
