#!/usr/bin/env bash
# §216 s1 round-2 labelling + 3070-only gates (SD2_v1 §2.1 / §2.2 / §7), on the 3070 (cuda:1) after scratch/s214_post_stepbyt.sh's "§214 stepbyt done" line
# (one job per card). (1) GATE-2 / GATE-1' items (B1 --stage gates) -> (2) their scoring (B2, eval_gate.json) -> (3) the pool (B1 --stage pool; exits 3 if a
# window has no passing labeller) -> (4) G-S16 EVMEM: part deep's deepest chunk (the 16 longest n-24 items, K' 28, fp32, bs 16) on FT-SELF's weights, peak only,
# output to scratch/_tmp (no accuracy printed or read; G-S18) -> (5) G-S19 HIST24-STORE: part hist24 with E4 on FT-SELF from the tree, output to scratch/_tmp,
# per-n means == the committed s215_ftself_s1/eval_hist24.json. (4) / (5) use the §216 tree copy until it is merged into boxes/.
#   nohup bash scratch/s216_s1_pool_chain.sh >> logs/s216_s1_pool.out 2>&1 &
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:1}; TREE=${TREE:-D:/Latent/scratch/_tmp/sd216_tree}; O=D:/Latent/scratch/_tmp/sd216_3070; mkdir -p $O
say() { echo "$(date '+%F %T') $*"; }
until grep -q "§214 stepbyt done" logs/s214_post_landing.out 2>/dev/null; do sleep 120; done
say "§214 chain done -> §216 s1 labelling on $DEV"
python scratch/s216_build_pool.py --lineage s1 --round 2 --stage gates --device $DEV > logs/s216_s1_gates.log 2>&1; rc=$?; say "B1 gates rc $rc $(tail -n 1 logs/s216_s1_gates.log | cut -c1-200)"
[ $rc = 0 ] || exit 1
python scratch/s216_gate2.py --lineage s1 --round 2 --device $DEV > logs/s216_s1_gate2.log 2>&1; rc=$?; say "B2 rc $rc $(tail -n 1 logs/s216_s1_gate2.log | cut -c1-300)"
[ $rc = 0 ] || exit 1
python scratch/s216_build_pool.py --lineage s1 --round 2 --stage pool --device $DEV > logs/s216_s1_pool.log 2>&1; rc=$?; say "B1 pool rc $rc $(tail -n 1 logs/s216_s1_pool.log | cut -c1-200)"
(cd $TREE && python - > $O/gs16.log 2>&1 <<EOF
import sys, json, os, torch; os.chdir("D:/Latent"); sys.path.insert(0, "$TREE")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
import boxes.vbank_eval as VE, boxes.vbank as VB, boxes.nested170 as N
m, _ = VE.load("boxes/runs_fsq/s215_ftself_s1", "$DEV"); res = {}
deep = sorted([it for it in N.test_set(1000, n_max=24) if it["n"] == 24], key=lambda it: -len(it["syms"]))
for bs in (16, 8, 32):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats("$DEV"); its = deep[:bs]
    try:
        with torch.no_grad():
            x, pad, lens = VB.batch_x(its, "$DEV"); rows = torch.arange(len(its), device="$DEV"); r_ = m.rollout(x, pad, 28, coda_rows={k: rows for k in range(29)})
            for k in range(29):
                m.answer_logits(r_["coda"][k][1], lens)
            del r_
        res[str(bs)] = round(torch.cuda.max_memory_allocated("$DEV") / 2 ** 30, 3)
    except torch.OutOfMemoryError:
        res[str(bs)] = "OOM"
out = {"peak_gib_by_bs": res, "T_max": max(len(it["syms"]) + 2 for it in deep[:16]), "card": torch.cuda.get_device_name("$DEV"), "rule": "bs 16 peak <= 7.0 GiB else bs 8",
       "pass_bs16": isinstance(res["16"], float) and res["16"] <= 7.0, "note": "no answer read (G-S18); output scratch only"}
json.dump(out, open("D:/Latent/scratch/gate_s216/gs16.json", "w", newline="\n"), indent=1); print("G-S16", out)
EOF
); say "G-S16 rc $? $(tail -n 1 $O/gs16.log | cut -c1-200)"
(cd $TREE && python -m boxes.vbank_diag --run D:/Latent/boxes/runs_fsq/s215_ftself_s1 --part hist24 --device $DEV --eval_bs 32 --out_dir $O > $O/hist24.log 2>&1); rc=$?
python -c "
import json, numpy as np; h = json.load(open('$O/eval_hist24.json')); c = json.load(open('boxes/runs_fsq/s215_ftself_s1/eval_hist24.json'))
arms = [a for a in c if isinstance(c[a], dict) and 'acc_n2' in c[a]]; same = all(h[a][k] == c[a][k] for a in arms for k in ('acc_n2', 'acc_k24'))
it = all(float(np.mean(np.array(h['items'][a][n]['ans_n2']) == np.array(h['items_y'][n]))) == h[a]['acc_n2'][n] and float(np.mean(np.array(h['items'][a][n]['ans_k24']) == np.array(h['items_y'][n]))) == h[a]['acc_k24'][n] for a in arms for n in h['items'][a])
o = {'per_n_means_bitwise_vs_committed': same, 'items_means_equal_per_n': it, 'arms': arms, 'card': h.get('card'), 'pass': same and it}
json.dump(o, open('scratch/gate_s216/gs19.json', 'w', newline='\n'), indent=1); print('G-S19', o)" 2>&1 | sed "s/^/$(date '+%F %T') /"
say "§216 s1 labelling + G-S16 / G-S19 done"
