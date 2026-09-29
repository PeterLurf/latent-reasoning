"""Print one STAT line per run name given (used by pod_status.sh on cloud pods)."""
import json, os, sys
for r in sys.argv[1:]:
    f = f"runs/{r}/log.jsonl"
    if not os.path.exists(f):
        print("STAT", r, "not-started"); continue
    L = [json.loads(l) for l in open(f)]
    T = [x for x in L if "loss" in x]; E = [x for x in L if "eval" in x]
    A = [x for x in L if "curriculum_advance_to" in x]
    ms = (T[-1]["sec"] - T[-6]["sec"]) / (T[-1]["step"] - T[-6]["step"]) * 1000 if len(T) > 6 else float("nan")
    fails = len([d for d in os.listdir("runs") if d.startswith(r + "_stage1fail")])
    step = T[-1]["step"] if T else 0
    print("STAT", r, "step", step, "stage", A[-1]["curriculum_advance_to"] if A else 1,
          "acc", round(E[-1]["eval"]["delayed"]["bal_acc"], 3) if E else "-", "ms", round(ms) if ms == ms else "-",
          "fails", fails, "DONE" if step >= 30000 else "")
