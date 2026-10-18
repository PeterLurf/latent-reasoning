#!/usr/bin/env bash
# §216 b315 s216_r2_s2 pre-launch gates (Audit 09:3x 10-18): after the r2-s2 pool exists (s216_pool_r2_s2/pool_meta.json) - G-S1 post hoc (CPU), G-S2 at
# Kb 20 (first 1,000 records; + GATE-2-s2 / GATE-1'-s2 gs2_compose), G-S4 init + resume on s216_r1_s2/model.pt with this pool (K 20), G-S17 part 2 (full pool load
# + deep-shape selection); the 4060 (cuda:0), micro / self_micro 16 for the 8 GB card (the host's G-S9 picks the real micro from {1, 2, 4, 8}).
#   nohup bash scratch/s216_s2r2_gates.sh >> logs/s216_s2r2_gates.out 2>&1 &
cd "$(dirname "$0")/.."
say() { echo "$(date '+%F %T') $*"; }
until [ -f boxes/runs_fsq/s216_pool_r2_s2/pool_meta.json ]; do sleep 120; done
OMP_NUM_THREADS=3 python scratch/s216_gs1.py --lineage s2 --round 2 > logs/s216_gs1_s2r2.log 2>&1; say "G-S1 s2 r2 rc $? $(tail -n 1 logs/s216_gs1_s2r2.log | cut -c1-12)"
OMP_NUM_THREADS=2 python - <<'EOF' 2>&1 | sed "s/^/$(date '+%F %T') /"
import sys, json, os; sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch, boxes.vbank as VB, boxes.selfdepth as SD, boxes.nested170 as N
R = "boxes/runs_fsq/"; recs, _ = SD.load_pool(R + "s216_pool_r2_s2/pool_r2.jsonl", R + "s216_pool_r2_s2/pool_meta.json"); n_eq = 0; dl = 0
for r in recs[:1000]:
    tree = SD._parse_tree(r["syms"]); T = len(r["syms"]) + 2
    Es = VB.rec_targets([SD.self_item(r, 20)], 20, T, "cpu"); Eg = VB.rec_targets([{"root": tree, "syms": r["syms"], "n": r["n"]}], 20, T, "cpu")
    if torch.equal(Es, Eg):
        n_eq += 1
    else:
        dl += int(r["chain_label"] == int(N.value(tree)))
eg = json.load(open(R + "s216_gate2_s2/eval_gate.json")); o = {"pool_first_1000": {"n_equal": n_eq, "n_diff": 1000 - n_eq, "n_diff_label_correct": dl}, "Kb": 20, "gate_gs2": eg["gs2_compose"]}
o["pass"] = all(v["PASS"] for v in eg["gs2_compose"].values()) and (1000 - n_eq) <= 10
json.dump(o, open("scratch/gate_s216/s2_s2r2.json", "w", newline="\n"), indent=1); print("G-S2 s2 r2", "PASS" if o["pass"] else "FAIL", o)
EOF
F="--mode slva --table train --seed 2 --steps 12000 --lr 3e-5 --cur 8 --c_sched 0,1 --kmax 12 --kextra 4 --init_from boxes/runs_fsq/s216_r1_s2/model.pt --self_pool boxes/runs_fsq/s216_pool_r2_s2/pool_r2.jsonl --p_self 0.5 --self_kmax 20 --self_kextra 4 --draws_log_steps 20000 --bs 64 --micro 16 --self_micro 16 --device cuda:0"
O=D:/Latent/scratch/_tmp/gs4_r2s2; mkdir -p $O/A $O/B
python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --keep_ckpt_steps 0 --stop_at 2 --out $O/A > $O/A.log 2>&1 && cp $O/A/ckpt_prev.pt $O/A_prev1.pt && cp $O/A/ckpt_last.pt $O/A_ckpt2.pt \
  && python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --resume $O/A/ckpt_last.pt --stop_at 3 --out $O/A > $O/A_resume.log 2>&1 \
  && python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --stop_at 3 --out $O/B > $O/B.log 2>&1; say "G-S4 runs rc $?"
python - <<'EOF' 2>&1 | sed "s/^/$(date '+%F %T') /"
import torch, json
G = "D:/Latent/scratch/_tmp/gs4_r2s2"; I = "D:/Latent/boxes/runs_fsq/s216_r1_s2"; L = lambda p: torch.load(p, map_location="cpu", weights_only=False)
base = torch.load(I + "/model.pt", map_location="cpu", weights_only=True); c0 = L(G + "/A/ckpt_model_0.pt")["model"]; a, b = L(G + "/A/ckpt_last.pt"), L(G + "/B/ckpt_last.pt"); al = open(G + "/A.log").read()
o = {"ckpt0_bitwise": sorted(c0) == sorted(base) and all(torch.equal(c0[k], base[k]) for k in base), "init_line_ok": "(missing: [])" in al and "calibrate() NOT called" in al,
     "calibration_equal": json.load(open(G + "/A/calibration.json")) == json.load(open(I + "/calibration.json")),
     "ckpt_steps": [L(G + "/A_prev1.pt")["step"], L(G + "/A_ckpt2.pt")["step"], a["step"], b["step"]], "resume_line": "RESUME" in open(G + "/A_resume.log").read(),
     "self_state_equal": a["self_state"] == b["self_state"], "rng_s_equal": a["rng_s"] == b["rng_s"], "np_rngs_equal": a["np_rng"] == b["np_rng"] and a["rng_k"] == b["rng_k"],
     "adam_steps": [float(max(v["step"] for v in a["opt"]["state"].values())), float(max(v["step"] for v in b["opt"]["state"].values()))],
     "draws_log_equal": json.load(open(G + "/A/draws_log.json")) == json.load(open(G + "/B/draws_log.json"))}
o["pass"] = (o["ckpt0_bitwise"] and o["init_line_ok"] and o["calibration_equal"] and o["ckpt_steps"] == [1, 2, 3, 3] and o["resume_line"] and o["self_state_equal"]
             and o["rng_s_equal"] and o["np_rngs_equal"] and o["adam_steps"] == [3.0, 3.0] and o["draws_log_equal"])
json.dump(o, open("D:/Latent/scratch/gate_s216/gs4_r2_s2.json", "w", newline="\n"), indent=1); print("G-S4 r2_s2", "PASS" if o["pass"] else "FAIL", o)
EOF
O2=D:/Latent/scratch/_tmp/gs17_r2s2; mkdir -p $O2; t0=$(date +%s)
python -m boxes.vbank $F --mem_check --mem_check_self --mem_check_steps 2 --mem_budget 7.0 --eval_bs 8 --out $O2 > $O2/run.log 2>&1; rc=$?
python -c "
import json; L = open('$O2/run.log').read().splitlines(); o = {'rc': $rc, 'sec': $(( $(date +%s) - t0 )), 'self_pool_line': [l for l in L if l.startswith('SELF_POOL')], 'memcheck': [l for l in L if l.startswith('MEMCHECK')], 'note': '4060, micro / self_micro 16'}
o['pass'] = $rc == 0 and bool(o['self_pool_line']) and bool(o['memcheck']); json.dump(o, open('scratch/gate_s216/gs17_r2_s2.json', 'w', newline='\n'), indent=1); print('G-S17 r2_s2', o)" | sed "s/^/$(date '+%F %T') /"
rm -f D:/Latent/scratch/_tmp/gs17_r2s2/*.pt
say "§216 b315 pre-launch gates done"
