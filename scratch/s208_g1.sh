#!/usr/bin/env bash
# §208 gate G1 (QONCE draft §12.4 / AB draft §11.4): pre / post-edit references on the 3070 (cuda:1, bf16), A's flags at smoke size:
#   plain (all new flags off) and --aux_nopass alone (§204's NOPASS path, string target): 200 steps, bs 8, K 10, sched_scale 0.05 (fade / gamma / aux
#   all active), log every 20 + the end-of-run eval; then A's eval-only re-run at eval_bs 64 in a copy dir.   usage: s208_g1.sh <tag>
cd /d/Latent; tag=$1; O=scratch/_tmp/s208ref
T="--device cuda:1 --train_n 2000 --test_n 64 --valid_n 32 --fit_n 16 --extras_n 16 --steps 200 --bs 8 --valid_every 100 --log_every 20 --ckpt_every 0 --eval_bs 16 --sched_scale 0.05"
A="--arm stepaux --think_k 10 --aux_fade 24000,24000 --cache_lower --fast_add --chunk_delta 64 --lr 1e-4 --seed 1"
python -m boxes.gsm8k_stepaux $A $T --out $O/${tag}_plain > $O/${tag}_plain.log 2>&1; echo "${tag}_plain rc $?"
python -m boxes.gsm8k_stepaux $A $T --aux_nopass --out $O/${tag}_nopass > $O/${tag}_nopass.log 2>&1; echo "${tag}_nopass rc $?"
mkdir -p $O/${tag}_Aeval && cp boxes/runs_fsq/gsm_lat_aug_s1/{model.pt,args.json,log.jsonl} $O/${tag}_Aeval/
python -m boxes.gsm8k_stepaux --arm stepaux --think_k 10 --cache_lower --fast_add --chunk_delta 64 --bs 64 --seed 1 --eval_only --device cuda:1 --out $O/${tag}_Aeval > $O/${tag}_Aeval.log 2>&1; echo "${tag}_Aeval rc $?"
python - "$tag" <<'PY'
import sys, torch
t = sys.argv[1]
for a in ("plain", "nopass"):
    sd = torch.load("scratch/_tmp/s208ref/%s_%s/model.pt" % (t, a), map_location="cpu", weights_only=False)
    torch.save({"keys": sorted(sd.keys()), "sum": {k: float(v.double().sum()) for k, v in sd.items()}}, "scratch/_tmp/s208ref/%s_%s/sd_sig.pt" % (t, a))
PY
rm -f $O/${tag}_*/model.pt $O/${tag}_*/model_validbest.pt
echo G1_DONE $tag
