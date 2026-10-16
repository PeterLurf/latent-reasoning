#!/usr/bin/env bash
# fill_generic_px = fill_generic + DPH_MAX (the offer price cap, default 0.30; 2026-10-14: §200 B must land at <= $0.26/h for the $7 cap).
# GPU_NAME (default RTX_3090; 2026-10-15 §209 step 2: A100_SXM4) selects the card class in the offer search.
# Generic one-at-a-time Vast filler (2026-10-06): deploy the given gen-b<n> queues in order as fleet slots free - live instances <= 15
# (cap 16) and the summed $/h + $0.25 < $4. Each n is retried until its deploy STARTED. Waits first for fill135 to finish (no races).
# Offers: RTX 3090, excluded / live IPs skipped, 595.71 first.   EXTRA_120=... nohup bash scratch/fill_generic.sh <tag> <n> [<n> ...] &
cd "$(dirname "$0")/.."
V=C:/Users/peter/.local/vastai/vast.sh; export PYTHONIOENCODING=utf-8; tag=$1; shift
log() { echo "$(date +%T) fill_$tag $*" >> logs/gen_deploy.log; echo "$(date +%T) $*"; }
while ! grep -q "all 4 student pairs deployed" logs/fill135.out 2>/dev/null && [ ! -f scratch/fill135.stop ]; do sleep 120; done
for n in "$@"; do
  while true; do
    [ -f scratch/fill_$tag.stop ] && { log "stopped"; exit 0; }
    read -r cnt burn <<<"$(bash $V show instances --raw 2>/dev/null | python -c "import json,sys;r=json.load(sys.stdin);print(len(r), round(sum((o.get('dph_total') or 0) for o in r), 3))" 2>/dev/null)"
    if [ -n "$cnt" ] && [ "$cnt" -le 15 ] && python -c "import sys;sys.exit(0 if $burn + 0.25 < 4.0 else 1)"; then
      bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
      o=$(bash $V search offers 'gpu_name='"${GPU_NAME:-RTX_3090}"' num_gpus=1 reliability>0.97 dph<='"${DPH_MAX:-0.30}"' disk_space>=40 rentable=true inet_down>=300 cuda_vers>=12.4' -o dph --raw --limit 150 2>/dev/null | python -c "
import json,sys
ex={l.split()[0] for l in open('scratch/vast_exclude_ips.txt') if l.strip() and not l.startswith('#')}   # first token: lines may carry a trailing comment
used={l.split()[0] for l in open('scratch/live_hosts.txt') if l.strip()}
pf=[l.strip() for l in open('scratch/vast_prefer_ips.txt') if l.strip() and not l.startswith('#')] if __import__('os').path.exists('scratch/vast_prefer_ips.txt') else []   # Audit 2026-10-10: known-fast hosts first
rows=[(0 if any(str(x.get('public_ipaddr','')).startswith(q) for q in pf) else 1, 0 if str(x.get('driver_version','')).startswith('595.71') else 1, x['dph_total'], x['id']) for x in json.load(sys.stdin) if (not __import__('os').environ.get('DRIVER_EQ') or str(x.get('driver_version','')) == __import__('os').environ['DRIVER_EQ']) and (__import__('os').environ.get('DRIVER_EQ') or str(x.get('driver_version','')) not in ({l.split()[0] for l in open('scratch/reserved_drivers.txt') if l.strip() and not l.startswith('#')} if __import__('os').path.exists('scratch/reserved_drivers.txt') else set()))and x.get('public_ipaddr') not in ex and 'oid:%s' % x.get('id') not in ex and 'mid:%s' % x.get('machine_id') not in ex and x.get('public_ipaddr') not in used and not str(x.get('driver_version','')).startswith('580')]
print(sorted(rows)[0][3] if rows else '')")
      if [ -n "$o" ]; then
        log "b$n: deploying on offer $o ($cnt live, \$${burn}/h)"; bash scratch/gen_deploy_120.sh "$o $n"
        grep "gen host b$n offer $o:" logs/gen_deploy.log | tail -1 | grep -q STARTED && { log "b$n: STARTED"; bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1; sleep 60; break; }
        log "b$n: did not start, retrying"
      else
        log "b$n: no matching offer (DRIVER_EQ=${DRIVER_EQ:-any}, DPH_MAX=${DPH_MAX:-0.30}, reserved=$(grep -v '^#' scratch/reserved_drivers.txt 2>/dev/null | awk '{print $1}' | tr '\n' ' ')); retry in 10 min"   # Audit 2026-10-14: a silent wait is visible
      fi
    fi
    sleep 600
  done
done
log "done: $*"
