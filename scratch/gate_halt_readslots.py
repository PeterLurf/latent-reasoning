"""note 198-3 (Theory): halt173.load_model builds nested170's ReadSlotsN for a nested --arm readslots run (not §171's ReadSlots). Check on a tiny trained run
(argv[1] device, default cuda:1): (1) nested170 --arm readslots, 2 steps + the full eval -> model.pt / eval.json; (2) halt173.load_model: kind 'nested',
a ReadSlotsN, state_dict strict; (3) s175 aux_acc_by_pass recomputed on the loaded model (scratch/probe_pathindep.py, the same items / definition)
== eval.json's s175.aux_acc_by_pass; (4) halt173 --deepaux --deepaux_items_only --items_n 9,10,11,12 runs (rc 0, the items file written); (5) halt173.think_tokens == ReadSlotsN
encode valid.sum() on the device (40 n 9..12 items)
    python scratch/gate_halt_readslots.py [cuda dev]   -> boxes/runs_fsq/gate_halt_readslots/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_halt_readslots"); OUT = {}
RUN = os.path.join("scratch", "_ghr_run")
CELL = ["--arm", "readslots", "--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--kprimes", "0,1,4,8,12,16,24,32",
        "--path_indep", "--add_norm", "rms", "--steps", "2", "--bs", "8", "--per_n", "20", "--serial_pool", "40", "--eval_bs", "64", "--micro", "1"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; env = {**os.environ, "PYTHONPATH": ROOT}; shutil.rmtree(RUN, ignore_errors=True)
    try:
        r = subprocess.run([sys.executable, "-m", "boxes.nested170"] + CELL + ["--device", dev, "--out", RUN], cwd=ROOT, capture_output=True, text=True, env=env)
        OUT["train"] = {"rc": r.returncode, "err": r.stderr[-600:] if r.returncode else ""}; assert r.returncode == 0, r.stderr[-1500:]
        import torch
        import boxes.halt173 as H
        import boxes.nested170 as N
        kind, m, a = H.load_model(RUN, dev)
        sd = torch.load(os.path.join(RUN, "model.pt"), map_location=dev, weights_only=False); miss = m.load_state_dict(sd, strict=True)
        OUT["load"] = {"kind": kind, "class": type(m).__name__, "is_ReadSlotsN": isinstance(m, N.ReadSlotsN), "strict_missing": list(miss.missing_keys), "strict_unexpected": list(miss.unexpected_keys),
                       "n_slots": int(m.slots.shape[0]), "nopass": bool(m.nopass)}
        r2 = subprocess.run([sys.executable, "scratch/probe_pathindep.py", RUN, os.path.join(RUN, "model.pt"), dev], cwd=ROOT, capture_output=True, text=True, env=env)
        pj = json.load(open(os.path.join(RUN, "pathindep_final.json"))); ev = json.load(open(os.path.join(RUN, "eval.json")))
        OUT["aux"] = {"probe_rc": r2.returncode, "recomputed": pj["s175"]["aux_acc_by_pass"], "eval_json": ev["s175"]["aux_acc_by_pass"],
                      "path_indep_recomputed": pj["path_indep"], "path_indep_eval_json": ev.get("s182", {}).get("path_indep")}
        r3 = subprocess.run([sys.executable, "-m", "boxes.halt173", "--run", RUN, "--deepaux", "--deepaux_items_only", "--items_n", "9,10,11,12", "--items_tmax", "14",
                             "--items_file", "eval_s187_deepaux_items_n9_12.json", "--bs", "64", "--device", dev], cwd=ROOT, capture_output=True, text=True, env=env)
        fi = os.path.join(RUN, "eval_s187_deepaux_items_n9_12.json")
        OUT["deepaux_items"] = {"rc": r3.returncode, "err": r3.stderr[-600:] if r3.returncode else "", "file": os.path.exists(fi),
                                "n_rows": len(json.load(open(fi)).get("items", [])) if os.path.exists(fi) else 0}
        import numpy as np
        TE = N.test_set(20); ck_ = [it for it in TE if it["n"] in (9, 10, 11, 12)]; ck_ = [ck_[i] for i in np.random.default_rng(5).permutation(len(ck_))[:40]]
        tt_ = [(H.think_tokens(m, it), int(m.think_inputs([it], dev)[1].sum())) for it in ck_]
        OUT["think_tokens"] = {"n_checked": len(tt_), "all_equal_encode_valid": all(u == v for u, v in tt_), "first": tt_[:5]}
        OUT["all_pass"] = bool(OUT["think_tokens"]["all_equal_encode_valid"] and OUT["load"]["is_ReadSlotsN"] and kind == "nested" and not OUT["load"]["strict_missing"] and not OUT["load"]["strict_unexpected"]
                               and OUT["aux"]["recomputed"] == OUT["aux"]["eval_json"] and r3.returncode == 0 and OUT["deepaux_items"]["file"])
    finally:
        dump(); shutil.rmtree(RUN, ignore_errors=True); print(json.dumps(OUT, default=str)[:2500], flush=True)


if __name__ == "__main__":
    main()
