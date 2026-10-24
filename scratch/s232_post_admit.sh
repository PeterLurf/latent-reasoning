#!/usr/bin/env bash
# §232 Stage-A post-admit check (eval-only host; started by scratch/s232_fill.sh with nohup): wait (<= 60 min) until every staged dir's shipped model.pt
# is on the host, then sha256 each against the local staged copy (the staged copy was sha-checked against the committed run's model.pt at staging).
# Mismatch / missing -> HOLD: ALERT line (logs/s232_alerts.log + gen_deploy.log), the host's §224-chain drain watcher killed, the instance destroyed,
# the filler stopped (scratch/fill_s232.stop). Read-only ssh otherwise.   bash scratch/s232_post_admit.sh <n> <host> <port> <instance> <staged dir> ...
cd "$(dirname "$0")/.." || exit 2
STOPF=scratch/fill_s232.stop; ALERTS=logs/s232_alerts.log; PALOG=logs/s232_post_admit.log
n=$1; h=$2; p=$3; iid=$4; shift 4; W=/workspace/Latent
log() { echo "$(date +%T) s232_post b$n $*" >> logs/gen_deploy.log; echo "$(date '+%F %T') b$n $*" >> "$PALOG"; }
sq() { timeout 60 ssh -o ConnectTimeout=15 -i "$HOME/.ssh/runpod_ed25519" -p "$p" "root@$h" "$1" 2>/dev/null; }
hold() {
  echo "$(date '+%F %T') ALERT b$n: $* - HOLD (instance $iid destroyed, filler stopped); tell Audit / Theory" >> "$ALERTS"; log "ALERT $* - HOLD"
  for dp in $(ps -ef | grep "auto_drain_gen8_s224.sh $h $p gen_b$n $iid" | grep -v grep | awk '{print $2}'); do kill "$dp"; done
  touch "$STOPF"; log "$(bash scratch/vdestroy.sh "$iid" | tail -1)"; exit 3
}
bad=""
for d in "$@"; do
  rs=""; for i in $(seq 1 12); do rs=$(sq "sha256sum $W/$d/model.pt" | cut -c1-64); [ -n "$rs" ] && break; sleep 300; done
  ls_=$(sha256sum "$d/model.pt" | cut -c1-64)
  if [ "$rs" = "$ls_" ]; then log "$d model.pt sha OK (${ls_:0:12})"; else bad="$bad $d(${rs:0:12}!=${ls_:0:12})"; fi
done
[ -z "$bad" ] || hold "shipped model.pt sha mismatch / missing:$bad"
log "POSTCHECK OK ($# staged dirs)"
