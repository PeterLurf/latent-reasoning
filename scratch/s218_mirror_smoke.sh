#!/bin/bash
# §218 Note 218-15 launch condition: mirror -> resume smoke (local, 4060 cuda:0). A fake host /workspace (scratch/_tmp/mirsmoke/ws) trains lslva 300 steps
# with a ckpt at 300 (+ ckpt_fingerprint.json + log.jsonl); ckpt_mirror.py --local_host --companions copies the set; the mirror copy is placed in a
# fresh run dir (as a relaunch host would receive it) and resumed for 1 step -> resume_check.json (restored == saved fingerprint). ckpts deleted after.
cd /d/Latent; export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
W=scratch/_tmp/mirsmoke; rm -rf $W scratch/ckpt_mirror/lvb_mirsmoke; R=$W/ws/Latent/boxes/runs_fsq/lvb_mirsmoke; mkdir -p $R
G=scratch/gate_s218/mirror_smoke; mkdir -p $G
COMMON="--arch lslva --device cuda:0 --cur 4 --steps 400 --bs 8 --micro 1 --cal_n 16 --log_every 50 --per_n 2 --eval_bs 4"
python -B -m boxes.langvb $COMMON --ckpt_every 300 --stop_at 300 --out $R > $G/train.log 2>&1; rc1=$?
python -B scratch/ckpt_mirror.py 999:lvb_mirsmoke --once --local_host "$(pwd)/$W/ws" --companions ckpt_fingerprint.json,log.jsonl --optional draws_log.json > $G/mirror.log 2>&1; rc2=$?
X=$W/restore/lvb_mirsmoke; mkdir -p $X; cp scratch/ckpt_mirror/lvb_mirsmoke/* $X/
python -B -m boxes.langvb $COMMON --ckpt_every 0 --stop_at 301 --resume $X/ckpt_last.pt --out $X > $G/resume.log 2>&1; rc3=$?
cp $X/resume_check.json $G/ 2>/dev/null; cp scratch/ckpt_mirror/lvb_mirsmoke/meta.json $G/mirror_meta.json
echo "rc train $rc1 mirror $rc2 resume $rc3" > $G/rc.txt
find $W scratch/ckpt_mirror/lvb_mirsmoke -name "*.pt" -delete
echo "$(date '+%F %T') mirror_smoke rc $rc1 $rc2 $rc3" >> logs/s218_gates.out
