#!/usr/bin/env bash
# §206 smokes on the 3070 (fp32): J1 50-step train with every diagnostic (norms every 20), W 3-step (INIT-IDENTITY vs J1's s200_init), then gate 7:
# eval-only of the J1 smoke model with W's flags and with J1's flags (shared eval fields must be bitwise equal).
cd /d/Latent; O=scratch/_tmp/s206smk
T="--device cuda:1 --fp32 --train_n 64 --test_n 12 --valid_n 8 --fit_n 4 --extras_n 4 --bs 4 --log_every 10 --valid_every 25 --ckpt_every 0 --max_new 24 --eval_bs 4 --cot_eval_bs 4 --sched_scale 0.01"
W="--arm stepaux --think_k 2 --aux_fade 12000,12000 --cache_lower --fast_add --chunk_delta 64 --init_from boxes/runs_fsq/gsm_txt_aug_s1/model.pt --cot_valid"
J="--joint_cot 2 --kd_beta 1 --kd_layers 7-12 --kd_teacher drop --kd_norm teacher_std --kd_loss l1 --jc_norms_every 20"
python -m boxes.gsm8k_stepaux $W $J $T --steps 50 --out $O/j1 > $O/j1.log 2>&1; echo "j1 rc $?"
python -m boxes.gsm8k_stepaux $W $T --steps 3 --out $O/w3 > $O/w3.log 2>&1; echo "w3 rc $?"
python -m boxes.gsm8k_stepaux $W $T --steps 50 --eval_only --eval_out ev_W.json --out $O/j1 > $O/evW.log 2>&1; echo "evW rc $?"
python -m boxes.gsm8k_stepaux $W $J $T --steps 50 --eval_only --eval_out ev_J.json --out $O/j1 > $O/evJ.log 2>&1; echo "evJ rc $?"
echo SMK_DONE
