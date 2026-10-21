#!/bin/bash
# §223 Note 223-4 (b) (Theory 10-21): the per-term grad-norm logging (fa4e87ed) leaves the update unchanged. OLD = boxes/codi_ps.py at 9562591e
# (git show into an untracked boxes/_codi_ps_9562591e_tmp.py, removed after), NEW = HEAD. The PS cell, the same card, seed 1, eff_bs 8 / micro 8,
# 60 steps (d stages 1,2,3; CurGate bin 10 / cap 20), log_every 1 (NEW computes the grad norms at every step), no ckpt, under scratch/s222_det_wrap.py
# (as at G-D) so the comparison can be bitwise. PASS iff every per-step loss / ans_ce / num_ce / num_acc / recode_ce / recode_val_ce / gn_preclip / lr
# is identical AND trainable.pt's sha256 is identical.   bash scratch/s223_gn_bitwise.sh <cuda dev>  -> logs/s223_gn_bitwise.log
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DV=${1:-cuda:2}; T=scratch/_tmp/s223_gnbit; rm -rf "$T"; mkdir -p "$T"; L=logs/s223_gn_bitwise.log; : > "$L"
OLD=boxes/_codi_ps_9562591e_tmp.py; git show 9562591e:boxes/codi_ps.py > $OLD || { echo "git show failed" >> "$L"; exit 1; }
args="--recode --lam 1 --bottleneck --device $DV --eff_bs 8 --micro 8 --steps 60 --d_stages 1,2,3 --cg_bin 10 --cg_cap 20 --ckpt_every 0 --log_every 1"
echo "$(date '+%F %T') OLD 9562591e" >> "$L"; python -B scratch/s222_det_wrap.py $OLD $args --out $T/old >> "$L" 2>&1; echo "old rc $?" >> "$L"
echo "$(date '+%F %T') NEW HEAD $(git rev-parse --short HEAD)" >> "$L"; python -B scratch/s222_det_wrap.py boxes/codi_ps.py $args --out $T/new >> "$L" 2>&1; echo "new rc $?" >> "$L"
rm -f $OLD
python - "$T" >> "$L" 2>&1 <<'EOF'
import hashlib, json, sys
T = sys.argv[1]; K = ("loss", "ans_ce", "num_ce", "num_acc", "recode_ce", "recode_val_ce", "gn_preclip", "lr", "d_max", "stage")
rd = lambda p: {r["step"]: r for r in map(json.loads, open(p))}
a, b = rd(T + "/old/log.jsonl"), rd(T + "/new/log.jsonl")
diff = [(s, k) for s in sorted(a) for k in K if a[s].get(k) != b.get(s, {}).get(k)]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
so, sn = sha(T + "/old/trainable.pt"), sha(T + "/new/trainable.pt")
gn = [b[s].get("gn_ans_shared") is not None and b[s].get("gn_rec_shared") is not None for s in sorted(b)]
print("steps", len(a), len(b), "| field diffs", diff[:5], "| trainable sha old", so[:16], "new", sn[:16], "| NEW logged gn at every step:", all(gn))
print("GN-BITWISE", "PASS" if (not diff and so == sn and len(a) == len(b) == 60 and all(gn)) else "FAIL")
EOF
echo "$(date '+%F %T') done" >> "$L"
