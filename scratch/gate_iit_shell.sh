#!/usr/bin/env bash
# IIT-SERIAL shell gates (2070 fp32): G-PLAIN (no --iit: log.jsonl bitwise vs the pre-edit nested170) and G3 (--iit: gamma = 0 at steps 1-2 ->
# iit_calls 0 and records bitwise equal to the flag-off run; step 3 gamma > 0 -> one call).
cd /d/Latent; export OMP_NUM_THREADS=3
git show HEAD:boxes/nested170.py > boxes/_nested170_pre_tmp.py
A="--arm inject --think_k 4 --aux_step 1.0 --aux_fade 3000,3000 --steps 3000 --time_steps 3 --bs 8 --n_max_train 4 --fast_add --add_norm rms --per_n 10 --serial_pool 10 --log_every 1 --sched_scale 0.001 --device cuda:2"
O=scratch/_tmp/iit_gate
python -m boxes._nested170_pre_tmp $A --out $O/pre > $O/pre.log 2>&1; echo "pre rc $?"
python -m boxes.nested170 $A --out $O/post > $O/post.log 2>&1; echo "post rc $?"
python -m boxes.nested170 $A --iit 1.0,0.5,16 --out $O/iit > $O/iit.log 2>&1; echo "iit rc $?"
rm -f boxes/_nested170_pre_tmp.py
python - <<'PY'
import json
L = lambda d: [{k: v for k, v in json.loads(l).items() if k not in ("sec",)} for l in open("scratch/_tmp/iit_gate/%s/log.jsonl" % d) if l.strip()]
pre, post, iit = L("pre"), L("post"), L("iit")
strip = lambda r: {k: v for k, v in r.items() if not k.startswith(("iit_", "siit_"))}
print("G-PLAIN post == pre:", post == pre, len(pre))
g0 = [r for r in iit if r.get("gamma", 1) == 0 and "loss" in r]; print("G3 gamma0 records:", [(r["step"], r.get("iit_calls")) for r in iit if "loss" in r])
print("G3 flag-on == flag-off through gamma 0:", [strip(r) for r in iit if "loss" in r and r.get("gamma") == 0] == [r for r in post if "loss" in r and r.get("gamma") == 0])
PY
