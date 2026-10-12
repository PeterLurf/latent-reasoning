#!/bin/bash
# compare gen_deploy_120's bash auto_ship with scratch/auto_ship.py on the given queue files (Audit: identical lists before the swap)
cd /d/Latent; source /tmp/as_fn.sh; bad=0
for q in "$@"; do
  t0=$(date +%s); b=$(auto_ship "$q"); t1=$(date +%s); p=$(python scratch/auto_ship.py "$q")
  if [ "$(echo $b | tr ' ' '\n' | sort -u)" = "$(echo $p | tr ' ' '\n' | sort -u)" ]; then echo "SAME $q ($(echo $b | wc -w) files, bash $((t1-t0)) s)"; else echo "DIFF $q"; diff <(echo $b | tr ' ' '\n' | sort -u) <(echo $p | tr ' ' '\n' | sort -u); bad=1; fi
done; echo "ALL_SAME=$((1-bad))"
