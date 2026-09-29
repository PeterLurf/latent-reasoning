#!/usr/bin/env bash
# Poll scratch/pods.txt every 5 min until some pod has logs/EVAL_ALL_DONE; fetch all of that pod's runs
# (checksum-verified, scratch/pod_fetch.sh), print "READY <pod-id>" for pods whose fetches all succeeded, then exit
# so the pod can be deleted. Prints "FETCHFAIL <pod-id> <run>" if a fetch fails (pod is then not marked READY).
cd "$(dirname "$0")/.."
while :; do
  found=0
  while read -r id U card price runs; do
    d=$(printf 'test -f /workspace/Latent/logs/EVAL_ALL_DONE && echo "PODDONE_%s"; exit\n' "$id" \
        | timeout 60 ssh -tt -i ~/.ssh/runpod_ed25519 $U@ssh.runpod.io 2>&1 | tr -d '\r' | grep -aoE "PODDONE_$id\$")
    [ -n "$d" ] || continue
    found=1; ok=1
    for r in $runs; do bash scratch/pod_fetch.sh "$U" "$r" </dev/null || { echo "FETCHFAIL $id $r"; ok=0; }; done
    [ $ok = 1 ] && echo "READY $id"
  done < <(grep -v '^#' scratch/pods.txt)
  [ $found = 1 ] && exit 0
  sleep 300
done
