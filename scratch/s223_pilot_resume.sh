#!/bin/bash
# §223 Note 223-4 pilot RECOVERY after the 10-21 19:24 crash (0x133; the 3070 fell off the bus while running T2 textself). The PS (2070) and m1_4060
# (4060) cells died at 3,100 / 3,500 steps; both resume from their full-state ckpt_last.pt (step 2,000; resume path gated bitwise in
# logs/s223_resume.log) with their original flags + --resume + the micro auto_micro chose at launch (8; on --resume auto_micro is skipped, so --micro
# is passed explicitly). Pre-crash args.json / log.jsonl / stages.jsonl are kept as *.precrash; the resumed run appends to log.jsonl / stages.jsonl
# and s223_pilot_summary.py keeps the last record per step (rows 2,001..crash are re-run). Registered kernel on the 4060; math SDP on the 2070 (Turing);
# neither is bitwise across a resume, so the pilot is descriptive (resumed_from 2000 in cell_meta.json). Only 2 cards run (crash note: limit
# concurrent load); direct_4060 runs after m1_4060 on the 4060.
#   nohup bash scratch/s223_pilot_resume.sh ps  >> logs/s222_chain.out 2>&1 &    nohup bash scratch/s223_pilot_resume.sh 4060 >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
T=scratch/_tmp/s223_pilot; L=logs/s222_chain.out
keep() { for f in args.json log.jsonl stages.jsonl; do [ -e "$1/$f.precrash" ] || cp "$1/$f" "$1/$f.precrash"; done; }
case $1 in
  ps)
    O=$T/ps; keep $O; echo "$(date '+%F %T') s223 pilot ps RESUME from ckpt (cuda:2, kernel math, micro 8)" >> $L
    python -B scratch/s223_mathsdp_wrap.py boxes/codi_ps.py --recode --lam 1 --bottleneck --device cuda:2 --out $O --stop_at 4600 --micro 8 --resume > logs/s223_pilot_ps_resume.log 2>&1; rc=$?
    printf '{"card_device": "cuda:2", "card": "RTX 2070", "kernel": "math", "rc": %s, "resumed_from": 2000, "crash": "10-21 19:24 at step 3100"}\n' "$rc" > $O/cell_meta.json
    echo "$(date '+%F %T') s223 pilot ps rc $rc (9 = stopped at 4600 as registered)" >> $L
    python -B scratch/s223_pilot_summary.py cuda:2 math >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 pilot summary rc $? -> scratch/gate_s223/pilot.json" >> $L;;
  4060)
    O=$T/m1_4060; keep $O; echo "$(date '+%F %T') s223 pilot m1_4060 RESUME from ckpt (cuda:0, kernel registered, micro 8)" >> $L
    python -B boxes/codi_ps.py --recode --lam 1 --device cuda:0 --out $O --stop_at 4600 --micro 8 --resume > logs/s223_pilot_m1_4060_resume.log 2>&1; rc=$?
    printf '{"card_device": "cuda:0", "card": "RTX 4060", "kernel": "registered", "rc": %s, "resumed_from": 2000, "crash": "10-21 19:24 at step 3500"}\n' "$rc" > $O/cell_meta.json
    echo "$(date '+%F %T') s223 pilot m1_4060 rc $rc (9 = stopped at 4600 as registered)" >> $L
    O=$T/direct_4060; [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"
    echo "$(date '+%F %T') s223 pilot direct_4060 start (cuda:0, kernel registered)" >> $L
    python -B boxes/codi_ps.py --direct --device cuda:0 --out $O --stop_at 4600 --auto_micro --mem_budget 7.0 > logs/s223_pilot_direct_4060.log 2>&1; rc=$?
    printf '{"card_device": "cuda:0", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$rc" > $O/cell_meta.json
    echo "$(date '+%F %T') s223 pilot direct_4060 rc $rc (9 = stopped at 4600 as registered)" >> $L
    python -B scratch/s223_pilot_summary.py cuda:2 math >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 pilot summary (4060 cells) rc $? -> scratch/gate_s223/pilot.json" >> $L;;
esac
