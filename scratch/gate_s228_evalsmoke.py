"""§228 eval smoke (functional, small sizes; the pilot's own eval is the full-scale run): a 20-step inject d_max 2 model (scratch/_tmp/s228/evsmoke,
--steps 20 --eval_parts '' on the 4060) and a 20-step direct one; every inject_ow_eval part on truncated sets (test2 / transfer 20 per d, pairs 12 per
d with their floors). Checks: each part writes its file; SELF bitwise True for state / cache / leak; leak exposure (rows differ in every pair);
pred shapes (inject 15 K' per item); K'0-NOBIAS == K'0 when wo.bias is 0 (it is not after training: reported). -> scratch/gate_s228/evalsmoke.json
    python -B scratch/gate_s228_evalsmoke.py"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); TMP = ROOT + "/scratch/_tmp/s228"; DEV = "cuda:0"
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def train(name, fl):
    d = TMP + "/" + name
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    with open(d + "/stdout.log", "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", "boxes.inject_ow"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", "50", "--eval_parts", "", "--log_every", "10",
                                                                              "--ckpt_every", "0", "--device", DEV, "--out", d], stdout=f_, stderr=subprocess.STDOUT, cwd=ROOT, env=ENV).returncode
    return d, rc


def main():
    import boxes.inject_ow as IO
    import boxes.inject_ow_eval as EV
    o = {}
    di, ri = train("evsmoke_lat", ["--arm", "inject", "--think_k", "8", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", "2"])
    dd, rd = train("evsmoke_dir", ["--arm", "direct", "--d_max", "2"])
    o["train_rc"] = [ri, rd]
    S = IO.build_sets()
    S["test2"] = {d: v[:20] for d, v in S["test2"].items()}; S["transfer"] = {d: v[:20] for d, v in S["transfer"].items()}
    S["pairs"] = {d: v[:12] for d, v in S["pairs"].items()}; S["floors"] = {d: v[:12] for d, v in S["floors"].items()}
    EV.run(di, list(EV.PARTS), DEV, 50, sets=S); EV.run(dd, list(EV.PARTS), DEV, 50, sets=S)
    ld = lambda p: json.load(open(p))  # noqa: E731
    o["files_lat"] = sorted(f_ for f_ in os.listdir(di) if f_.startswith("eval")); o["files_dir"] = sorted(f_ for f_ in os.listdir(dd) if f_.startswith("eval"))
    sw = ld(di + "/eval_swaps.json"); ac = ld(di + "/eval_acc.json"); ad = ld(dd + "/eval_acc.json")
    o["self_bitwise"] = sw["self_bitwise"]; o["leak_rows_differ_all"] = all(r_["leak_rows_differ"] for d in sw["items"] for r_ in sw["items"][d])
    o["last7_same_share"] = {d: v.get("last7_tokens_same_share") for d, v in sw["by_d"].items()}
    o["pairs_n"] = {d: v.get("n_pairs") for d, v in sw["by_d"].items()}; o["floor_missing"] = sum(p_ < 0 for d in sw["items"] for r_ in sw["items"][d] for p_ in r_["floor_preds"])
    o["state_swap_ts"] = {d: sorted(sw["items"][d][0]["state_swap"]) for d in sw["items"] if sw["items"][d]}
    o["budget_front_len"] = {d: len(sw["items"][d][0]["budget_front"]) for d in sw["items"] if sw["items"][d]}
    o["lat_pred_len"] = len(ac["items"]["2"]["pred"][0]); o["dir_pred_type"] = type(ad["items"]["2"]["pred"][0]).__name__
    o["wo_bias_norm"] = ac.get("wo_bias_norm"); o["sec"] = {p_: ld(di + "/eval_%s.json" % p_)["sec"] for p_ in EV.PARTS}
    o["pass"] = (o["train_rc"] == [0, 0] and all(o["self_bitwise"].values()) and o["leak_rows_differ_all"] and o["lat_pred_len"] == 15 and o["dir_pred_type"] == "int"
                 and set(o["files_lat"]) >= {"eval.json"} | {"eval_%s.json" % p_ for p_ in EV.PARTS} and set(o["files_dir"]) == {"eval.json", "eval_acc.json", "eval_transfer.json"}
                 and o["budget_front_len"] == {str(d): 8 - -(-d // 2) + 1 for d in range(4, 9)})
    json.dump(o, open(ROOT + "/scratch/gate_s228/evalsmoke.json", "w", newline="\n"), indent=1); print("evalsmoke pass", o["pass"], flush=True)


if __name__ == "__main__":
    main()
