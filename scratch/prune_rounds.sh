#!/usr/bin/env bash
# Audit 2026-10-04: on §112 hosts, delete <run>/rounds/ on-host BEFORE the drain for seeds that never acquired:
# every round's gate < 0.01 AND k1_never_labelled_acc < 0.1; never if any gate >= 0.5 or cumulative 512 >= 0.5.
# Only when no tar is running on the host (never races a drain). Deletions logged to logs/prune_rounds.log.
#   bash scratch/prune_rounds.sh d2 d3 ...   (tags as in scratch/live_hosts.txt: gen-<tag>)
cd "$(dirname "$0")/.."
TAGS="$*"
# ALWAYS=1 (Audit 00:11): delete every finished run's rounds/ (per_round_split in eval.json already records the acquisition)
PY="ALWAYS=${ALWAYS:-0}"'
import json,glob,os,shutil
for e in glob.glob("boxes/runs_s5/s112_*/eval.json"):
    d=os.path.dirname(e); r=os.path.join(d,"rounds")
    if not os.path.isdir(r): continue
    j=json.load(open(e)); gates=[x.get("gate") for x in j.get("rounds",[]) if x.get("gate") is not None]
    keep=any(g>=0.5 for g in gates) or j["cumulative"]["512"]>=0.5
    if ALWAYS or ((not keep) and gates and max(gates)<0.01 and (j.get("k1_never_labelled_acc") or 0)<0.1):
        shutil.rmtree(r); print("PRUNED", d, "max_gate", max(gates) if gates else None, "never_labelled", j.get("k1_never_labelled_acc"), "cum512", j["cumulative"]["512"])'
while :; do
  for t in $TAGS; do
    read -r H P _ < <(grep " gen-$t\$" scratch/live_hosts.txt | tail -1); [ -z "$H" ] && continue
    out=$(printf '%s\n' "$PY" | timeout 60 ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H "cd /workspace/Latent 2>/dev/null && { pgrep -x tar >/dev/null && echo TAR_BUSY || python3 -; }" 2>/dev/null | grep -E "PRUNED|TAR_BUSY")
    echo "$out" | grep PRUNED | sed "s/^/$(date +%T) gen-$t /" >> logs/prune_rounds.log
  done
  sleep 120
done
