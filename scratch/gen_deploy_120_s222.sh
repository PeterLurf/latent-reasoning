#!/usr/bin/env bash
# §222 copy of scratch/gen_deploy_120_s219.sh (Note 222-1): the ONLY changes - the syntax gate covers the §222 drain files, and the armed watcher is
# scratch/auto_drain_gen8_s222.sh with prefix s222 (-> scratch/drain_gen_partial_s222.sh, §222 done markers). The §222 filler calls
# bash scratch/gen_deploy_120_s222.sh "<offer> <n>". (§219 copy header: gen_deploy_120.sh for §120 fleets.)
# §120 fleet: bash scratch/gen_deploy_120.sh "<offer> <n>" -> gen-b<n> on scratch/gen_q_b<n>.txt (fsq120), auto-drain gen5.
cd "$(dirname "$0")/.."
for f_ in "$0" scratch/vdeploy_toy.sh scratch/gen_vast_job.sh scratch/auto_drain_gen8_s222.sh scratch/drain_gen_partial_s222.sh scratch/s222_done_marker.sh; do   # Audit 2026-10-06: syntax gate
  bash -n "$f_" 2>/dev/null || { echo "$(date +%T) DEPLOY_ABORT bash -n failed: $f_" >> logs/gen_deploy.log; echo "DEPLOY_ABORT bash -n $f_"; exit 3; }
done
auto_ship() {   # Audit 2026-10-07 10:49 (s138 shipped without scratch/restart138.py + hostcheck.py): the queue's own modules ship automatically -
  # hostcheck always; scratch.X / boxes.X queue modules as scratch/X.py / boxes/X.py; boxes.Y named inside a shipped scratch driver too
  local q=$1 m f a="scratch/hostcheck.py"
  for m in $(grep -v "^#" "$q" | awk 'NF >= 2 {print $2}' | sort -u); do f=$(echo "$m" | tr . /).py; [ -f "$f" ] && a="$a $f"
    case "$f" in scratch/*|boxes/*) for b in $(grep -o -E "boxes[./][a-z0-9_]+" "$f" | sed "s|[./]|/|; s|$|.py|" | sort -u); do [ -f "$b" ] && a="$a $b"; done;; esac
  done
  for f in $(grep -v "^#" "$q" | tr " " "\n" | grep -E "^(boxes|bench|scratch)/" | sort -u); do [ -f "$f" ] && a="$a $f"; done   # file arguments (s142 cache, gate.pt)
  grep -q "^# HOSTPROBE" "$q" && a="$a boxes/nested170.py"                # the on-host speed probe (gen_vast_job.sh) runs §170 INJECT K8
  for t in $(grep -v "^#" "$q" | grep -o -E -- "--task [0-9]x[0-9]" | awk '{print $2}' | sort -u); do   # multiplication bench files
    for f in bench/data/mult_${t}_test.txt bench/data/mult_train/${t}_train.txt; do [ -f "$f" ] && a="$a $f"; done
  done
  for pass_ in 1 2 3; do                                                  # boxes / scratch modules imported by shipped files, to a fixed point
    for f in $(echo $a | xargs -n 1 | grep -E "^(boxes|scratch)/.*[.]py$"); do
      for b in $(grep -o -E "boxes[./][a-z0-9_]+" "$f" | sed "s|[./]|/|; s|$|.py|" | sort -u); do [ -f "$b" ] && a="$a $b"; done
    done
  done; echo $a | xargs -n 1 | sort -u | xargs; }
one() { o=$1; n=$2; np_=$(wc -l < scratch/pods.txt)
  out=$(EXTRA_FILES="${SHIP_GPT2:+prosqa/ckpt/gpt2_pretrained.bin} boxes/gen97.py boxes/mk108.py boxes/fsq120.py $(python scratch/auto_ship.py scratch/gen_q_b$n.txt) ${EXTRA_120:-}" bash scratch/vdeploy_toy.sh $o gen-b$n scratch/gen_q_b$n.txt gen_b$n scratch/gen_vast_job.sh 2>&1)
  echo "$(date +%T) gen host b$n offer $o: $(echo "$out" | grep -E 'INSTANCE|NVIDIA|STARTED|BAD HOST|FAILED|STUCK' | tr '\n' ' ')" >> logs/gen_deploy.log
  echo "$out" | grep -qE 'INSTANCE|FAILED|BAD HOST|STARTED' || echo "$(date +%T) DEPLOY_SILENT_FAIL offer $o: $(echo "$out" | tail -n 3 | tr '\n' ' ')" >> logs/gen_deploy.log   # Audit: a silent vdeploy exit is logged, never blank
  if echo "$out" | grep -q STARTED; then read -r _ id hh pp <<<"$(echo "$out" | grep '^INSTANCE')"
    nohup bash scratch/auto_drain_gen8_s222.sh "$hh $pp gen_b$n $id s222" > logs/auto_drain_gen_b$n.out 2>&1 &
  else id=$(echo "$out" | grep '^INSTANCE' | awk '{print $2}')
    for x_ in $(tail -n +$((np_ + 1)) scratch/pods.txt | grep " label gen-b$n, " | awk '{print $1}'); do   # Audit (b157 leaks): every instance THIS attempt created
      case " $id " in *" $x_ "*) ;; *) id="$id $x_"; echo "$(date +%T) gen host b$n offer $o: no INSTANCE line; destroying created $x_ (pods.txt)" >> logs/gen_deploy.log;; esac
    done
    for x_ in $id; do bash scratch/vdestroy.sh $x_ >> logs/gen_deploy.log 2>&1; done; fi; }
for x in "$@"; do set -- $x; one $1 $2 & sleep 3; done; wait
echo "$(date +%T) gen B DEPLOY_DONE" >> logs/gen_deploy.log
