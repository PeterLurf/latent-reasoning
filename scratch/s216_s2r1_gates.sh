#!/usr/bin/env bash
# §216 b314 s216_r1_s2 pre-launch gates (Audit's conditions 07:0x 10-18): after the r1-s2 pool exists (s216_pool_r1_s2/pool_meta.json) - G-S1 post hoc (CPU),
# G-S4 init + resume on its init (s216_d3t_s2/model.pt) and pool, and G-S17 part 2 (the trainer's full pool load + deep-shape selection) on the 4060 (cuda:0;
# micro / self_micro 16 for the 8 GB card - the host's G-S9 picks the real micro). Results -> scratch/gate_s216/{gs1_s2_r1, gs4_r1_s2, gs17_r1_s2}.json.
#   nohup bash scratch/s216_s2r1_gates.sh >> logs/s216_s2r1_gates.out 2>&1 &
cd "$(dirname "$0")/.."
say() { echo "$(date '+%F %T') $*"; }
until [ -f boxes/runs_fsq/s216_pool_r1_s2/pool_meta.json ]; do sleep 120; done
OMP_NUM_THREADS=3 python scratch/s216_gs1.py --lineage s2 --round 1 > logs/s216_gs1_s2r1.log 2>&1; say "G-S1 s2 r1 rc $? $(tail -n 1 logs/s216_gs1_s2r1.log | cut -c1-12)"
F="--mode slva --table train --seed 2 --steps 12000 --lr 3e-5 --cur 8 --c_sched 0,1 --kmax 12 --kextra 4 --init_from boxes/runs_fsq/s216_d3t_s2/model.pt --self_pool boxes/runs_fsq/s216_pool_r1_s2/pool_r1.jsonl --p_self 0.5 --self_kmax 16 --self_kextra 4 --draws_log_steps 20000 --bs 64 --micro 16 --self_micro 16 --device cuda:0"
O=D:/Latent/scratch/_tmp/gs4_r1s2; mkdir -p $O/A $O/B
python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --keep_ckpt_steps 0 --stop_at 2 --out $O/A > $O/A.log 2>&1 && cp $O/A/ckpt_prev.pt $O/A_prev1.pt && cp $O/A/ckpt_last.pt $O/A_ckpt2.pt \
  && python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --resume $O/A/ckpt_last.pt --stop_at 3 --out $O/A > $O/A_resume.log 2>&1 \
  && python -m boxes.vbank $F --log_every 1 --ckpt_every 1 --stop_at 3 --out $O/B > $O/B.log 2>&1; say "G-S4 runs rc $?"
python - <<'EOF' 2>&1 | sed "s/^/$(date '+%F %T') /"
import torch, json
G = "D:/Latent/scratch/_tmp/gs4_r1s2"; I = "D:/Latent/boxes/runs_fsq/s216_d3t_s2"; L = lambda p: torch.load(p, map_location="cpu", weights_only=False)
base = torch.load(I + "/model.pt", map_location="cpu", weights_only=True); c0 = L(G + "/A/ckpt_model_0.pt")["model"]; a, b = L(G + "/A/ckpt_last.pt"), L(G + "/B/ckpt_last.pt"); al = open(G + "/A.log").read()
o = {"ckpt0_bitwise": sorted(c0) == sorted(base) and all(torch.equal(c0[k], base[k]) for k in base), "init_line_ok": "(missing: [])" in al and "calibrate() NOT called" in al,
     "calibration_equal": json.load(open(G + "/A/calibration.json")) == json.load(open(I + "/calibration.json")),
     "ckpt_steps": [L(G + "/A_prev1.pt")["step"], L(G + "/A_ckpt2.pt")["step"], a["step"], b["step"]], "resume_line": "RESUME" in open(G + "/A_resume.log").read(),
     "self_state_equal": a["self_state"] == b["self_state"], "rng_s_equal": a["rng_s"] == b["rng_s"], "np_rngs_equal": a["np_rng"] == b["np_rng"] and a["rng_k"] == b["rng_k"],
     "adam_steps": [float(max(v["step"] for v in a["opt"]["state"].values())), float(max(v["step"] for v in b["opt"]["state"].values()))],
     "draws_log_equal": json.load(open(G + "/A/draws_log.json")) == json.load(open(G + "/B/draws_log.json"))}
o["pass"] = (o["ckpt0_bitwise"] and o["init_line_ok"] and o["calibration_equal"] and o["ckpt_steps"] == [1, 2, 3, 3] and o["resume_line"] and o["self_state_equal"]
             and o["rng_s_equal"] and o["np_rngs_equal"] and o["adam_steps"] == [3.0, 3.0] and o["draws_log_equal"])
json.dump(o, open("D:/Latent/scratch/gate_s216/gs4_r1_s2.json", "w", newline="\n"), indent=1); print("G-S4 r1_s2", "PASS" if o["pass"] else "FAIL", o)
EOF
O2=D:/Latent/scratch/_tmp/gs17_r1s2; mkdir -p $O2; t0=$(date +%s)
python -m boxes.vbank ${F/--bs 64/--bs 64} --mem_check --mem_check_self --mem_check_steps 2 --mem_budget 7.0 --eval_bs 8 --out $O2 > $O2/run.log 2>&1; rc=$?
python -c "
import json; L = open('$O2/run.log').read().splitlines(); o = {'rc': $rc, 'sec': $(( $(date +%s) - t0 )), 'self_pool_line': [l for l in L if l.startswith('SELF_POOL')], 'memcheck': [l for l in L if l.startswith('MEMCHECK')], 'note': '4060, micro / self_micro 16'}
o['pass'] = $rc == 0 and bool(o['self_pool_line']) and bool(o['memcheck']); json.dump(o, open('scratch/gate_s216/gs17_r1_s2.json', 'w', newline='\n'), indent=1); print('G-S17 r1_s2', o)" | sed "s/^/$(date '+%F %T') /"
rm -f D:/Latent/scratch/_tmp/gs17_r1s2/*.pt
say "§216 b314 pre-launch gates done"
