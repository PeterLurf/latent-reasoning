#!/usr/bin/env bash
# §254 session watch (Monitor source; read-only): filler / chain / post-admit events, the b390 drain logs, and every 5 min each admitted cell's host queue log
# (a run line's rc / QUEUE_DONE / an OOM) for the cells in logs/s256_admitted.log with n >= 378 and no realised line. Prints new lines only.
cd "$(dirname "$0")/.." || exit 2
PF=scratch/_tmp/s256_watch.pid                                                     # an expired Monitor leaves the previous instance running: stop it by its exact pid
if [ -f $PF ]; then op=$(cat $PF); if ps -ef | awk -v p="$op" '$2 == p' | grep -q "s256_watch.sh"; then
  for c in $(ps -ef | awk -v p="$op" '$3 == p {print $2}'); do for g in $(ps -ef | awk -v p="$c" '$3 == p {print $2}'); do kill "$g" 2>/dev/null; done; kill "$c" 2>/dev/null; done; kill "$op" 2>/dev/null; fi; fi
echo $$ > $PF
( tail -n0 -F logs/fill_s256.out logs/gen_deploy.log logs/auto_drain_gen_b390.out 2>/dev/null \
  | grep --line-buffered -E "fill_s256.*(ADMIT|NOT admitted|did not start|refusing|REFUSED|HOLD|CAP:|VANISHED)|chain:|s256_post.*(POSTCHECK|FAIL|BAD|MISMATCH)|Traceback|s256.*(ALERT|STOP)|ALL_DRAINED|DRAIN FAILED|NEW FAILURE|STALLED|UNREACHABLE" ) &
seen=""
while true; do
  while read -r n run hp; do
    h=${hp%:*}; p=${hp#*:}
    l=$(timeout 40 ssh -o ConnectTimeout=15 -o BatchMode=yes -i ~/.ssh/runpod_ed25519 -p "$p" "root@$h" "grep -E 'evaluated boxes/runs_fsq/$run rc|QUEUE_DONE|OutOfMemory' /workspace/Latent/logs/queue_gen_b$n.out | tail -n 2" 2>/dev/null | tr '\n' ' ')
    if [ -n "$l" ] && ! printf '%s' "$seen" | grep -qF "b$n:$l"; then echo "HOST b$n $run: $l"; seen="$seen|b$n:$l"; fi
  done < <(awk '/^3[0-9][0-9]:/ {split($1, a, ":"); if (a[1] >= 390) print a[1], a[2], $2}' logs/s256_admitted.log)
  sleep 300
done
