"""Theory 10-27 (before §236 scoring): batch-size / card / path invariance of the CUE-ONLY bar's source. s228_pdir2c_s1's acc on the OLD TEST2 (default law),
re-run on the 4060 by boxes/inject_aw_eval.py at eval_bs 100, fp32, into scratch/_tmp/s236/s228_old_bs100 (args.json copied, model.pt hard-linked; the run dir is
never written), diffed per item against the run's original eval_acc.json (3090, eval_bs 250, fp32, inject_ow_eval) -> scratch/gate_s236/ref_old_test2_invariance.json.
    python -B scratch/s236_bs_check_old.py"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    src = os.path.join(ROOT, "boxes", "runs_fsq", "s228_pdir2c_s1"); out = os.path.join(ROOT, "scratch", "_tmp", "s236", "s228_old_bs100")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out); shutil.copy(os.path.join(src, "args.json"), os.path.join(out, "args.json")); os.link(os.path.join(src, "model.pt"), os.path.join(out, "model.pt"))
    torch.zeros(1, device="cuda:0"); AE.run(out, ["acc"], "cuda:0", 100, sets=IO.build_sets())
    a = json.load(open(os.path.join(src, "eval_acc.json"))); b = json.load(open(os.path.join(out, "eval_acc.json"))); o = {"by_d": {}}
    for d in a["items"]:
        pa, pb = a["items"][d]["pred"], b["items"][d]["pred"]; sa, sb = a["items"][d]["sig"], b["items"][d]["sig"]
        o["by_d"][d] = {"n": len(pa), "same_items": sa == sb, "n_pred_differ": sum(x != y for x, y in zip(pa, pb))}
    o["n_pred_differ_total"] = sum(v["n_pred_differ"] for v in o["by_d"].values()); o["all_same_items"] = all(v["same_items"] for v in o["by_d"].values())
    o["original"] = {"eval_gpu": a.get("eval_gpu"), "eval_bs": a.get("eval_bs"), "eval_dtype": a.get("eval_dtype"), "module_sha256": a.get("module_sha256")}
    o["rerun"] = {"eval_gpu": b.get("eval_gpu"), "eval_bs": b.get("eval_bs"), "eval_dtype": b.get("eval_dtype"), "module_sha256": b.get("module_sha256")}
    o["def"] = "s228_pdir2c_s1 acc on the OLD TEST2: original (3090, eval_bs 250, inject_ow_eval) vs re-run (4060, eval_bs 100, inject_aw_eval), per-item preds"
    os.makedirs(os.path.join(ROOT, "scratch", "gate_s236"), exist_ok=True)
    json.dump(o, open(os.path.join(ROOT, "scratch", "gate_s236", "ref_old_test2_invariance.json"), "w", newline="\n"), indent=1)
    print("OLD-TEST2 INVARIANCE same_items", o["all_same_items"], "differ", o["n_pred_differ_total"], flush=True)
    os.remove(os.path.join(out, "model.pt"))


if __name__ == "__main__":
    main()
