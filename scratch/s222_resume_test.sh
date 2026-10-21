#!/bin/bash
# Note 222-1 item 1 (user's off-host rule): tested resume of the §222 group engine. Same card, smoke sizes (pool 3000, held 64, one group of 3 probes):
# REF = an uninterrupted run to 400 steps (+ extension); KILL = the same run stopped (exit 9) after step 330, i.e. after the step-300 checkpoint and 30
# steps past it; RESUME = --resume from ckpt_last.pt (step 300). PASS iff the per-step per-probe CE (repr) of RESUME's steps 301.. equal REF's bitwise
# and the group output files' arms / probe blocks are identical. Usage: bash scratch/s222_resume_test.sh cert|a1 <cuda dev>   -> logs/s222_resume_<kind>.log
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
K=$1; DV=${2:-cuda:0}; T=scratch/_tmp/s222_resume_$K; rm -rf "$T"; mkdir -p "$T/ref" "$T/res"; L=logs/s222_resume_$K.log; : > "$L"
if [ "$K" = cert ]; then
  base="python -B scratch/s222_cert.py --mode cert --model untrained --device $DV --pool_n 3000 --hold_n 64 --per_D 3 --steps 400 --eval_every 50 --group 3 --groups 1 --ckpt_every 100"
else
  base="python -B scratch/s222_a1.py --run scratch/gate_s218/g_l1c_model --parts probe --ts 0,1 --device $DV --pool_n 3000 --hold_n 64 --per_D 3 --steps 400 --eval_every 50 --group 3 --groups 1 --ckpt_every 100"
fi
echo "$(date '+%F %T') REF" >> "$L"; $base --out $T/ref --ckpt_dir $T/ref --loss_log $T/ref/loss.jsonl >> "$L" 2>&1; echo "ref rc $?" >> "$L"
echo "$(date '+%F %T') KILL at 330" >> "$L"; $base --out $T/res --ckpt_dir $T/res --loss_log $T/res/loss_kill.jsonl --stop_at 330 >> "$L" 2>&1; echo "kill rc $? (9 expected)" >> "$L"
echo "$(date '+%F %T') RESUME" >> "$L"; $base --out $T/res --ckpt_dir $T/res --loss_log $T/res/loss_resume.jsonl --resume >> "$L" 2>&1; echo "resume rc $?" >> "$L"
python - "$T" "$K" >> "$L" 2>&1 <<'EOF'
import json, sys, glob
T, K = sys.argv[1], sys.argv[2]
rd = lambda p: [json.loads(l) for l in open(p)]
ref = {r["step"]: r["loss"] for r in rd(T + "/ref/loss.jsonl")}; kil = {r["step"]: r["loss"] for r in rd(T + "/res/loss_kill.jsonl")}; res = {r["step"]: r["loss"] for r in rd(T + "/res/loss_resume.jsonl")}
pre = all(kil[s] == ref[s] for s in kil); post = sorted(res); same = all(res[s] == ref.get(s) for s in post)
print("kill-run steps", min(kil), "-", max(kil), "== ref:", pre)
print("resume steps", post[0], "-", post[-1], "(%d)" % len(post), "== ref bitwise:", same, "; ref steps", len(ref))
gf = lambda d: sorted(glob.glob(d + ("/cert_untrained.g1.json" if K == "cert" else "/eval_lvb_s222.g1.json")))
a, b = json.load(open(gf(T + "/ref")[0])), json.load(open(gf(T + "/res")[0])); blk = ("arms",) if K == "cert" else ("probe", "plant_s")
eq = all(a[k] == b[k] for k in blk); print("group file blocks", blk, "identical:", eq)
print("RESUME-TEST", K, "PASS" if (pre and same and post[0] == 301 and eq and len(res) + 300 == len(ref)) else "FAIL")
EOF
echo "$(date '+%F %T') done" >> "$L"
