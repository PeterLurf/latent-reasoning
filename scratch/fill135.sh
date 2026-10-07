#!/usr/bin/env bash
# §135 student filler (Theory option (a), Audit 2026-10-06): deploy the students as fleet slots free, one MATCHED PAIR per seed
# (seed S: TSD on gen-b(75+2(S-1)), BASE on gen-b(76+2(S-1))). Every 10 min: if live Vast instances <= 14 (cap 16) and the summed
# $/h + 2 x $0.25 < $4, deploy the next pair (gen_deploy_120: smoke / bash -n gates, gen7 watcher).
# Teacher model (Audit: no repeated home uploads): the FIRST TSD host gets model.pt in its tarball; every later TSD host gets the
# teacher dir host-to-host (`vastai copy C.<a TSD host>:dir C.<new>:dir`; fallback local:dir if none is up). TSD jobs start through
# scratch.wait_then, which waits for model.pt and verifies its sha256 before running fsq120. BASE needs only target_means.pt.
# Stop: touch scratch/fill135.stop.   nohup bash scratch/fill135.sh > logs/fill135.out 2>&1 &
cd "$(dirname "$0")/.."
V=C:/Users/peter/.local/vastai/vast.sh; export PYTHONIOENCODING=utf-8; T=boxes/runs_fsq/s135_teacher_s5; TD=/workspace/Latent/$T
log() { echo "$(date +%T) fill135 $*" >> logs/gen_deploy.log; echo "$(date +%T) $*"; }
while [ ! -s $T/model.pt ]; do [ -f scratch/fill135.stop ] && exit 0; sleep 60; done
SHA=$(sha256sum $T/model.pt | cut -c1-64); log "teacher model.pt local, sha256 $SHA"
A="--family s5 --arm final --micro 2 --micro_from 32 --no_cert --probe127 --ckpt_every 2000 --tsd_teacher $T --tsd_layer 10 --tsd_w 1.0 --tsd_tau 0.1"
for S in 1 2 3 4; do                                                        # TSD queue lines: wait for the teacher model, sha-verified
  nt=$((75 + 2 * (S - 1)))
  printf '%s\n' "# THEORY §135 student TSD s$S (notes 135-1..135-5): §128 S5 recipe, no restarts, --tsd teacher layer 10; model.pt copied host-to-host, sha-verified by scratch.wait_then; RTX 3090; boxes.fsq120" "# P=1" \
    "boxes/runs_fsq/s135_tsd_s5_s$S scratch.wait_then --wait $T/model.pt --sha $SHA --module boxes.fsq120 $A --seed $S --tsd teacher" > scratch/gen_q_b$nt.txt
done
tsd_ids=""; S=1; okT=0; okB=0
started() { grep "gen host b$1 offer $2:" logs/gen_deploy.log | tail -1 | grep -q STARTED; }
while [ $S -le 4 ]; do
  nt=$((75 + 2 * (S - 1))); nb=$((nt + 1)); need=$(( (1 - okT) + (1 - okB) ))
  while true; do
    [ -f scratch/fill135.stop ] && { log "stopped"; exit 0; }
    read -r n burn <<<"$(bash $V show instances --raw 2>/dev/null | python -c "import json,sys;r=json.load(sys.stdin);print(len(r), round(sum((o.get('dph_total') or 0) for o in r), 3))" 2>/dev/null)"
    if [ -n "$n" ] && [ "$n" -le $((16 - need)) ] && python -c "import sys;sys.exit(0 if $burn + 0.25 * $need < 4.0 else 1)"; then break; fi
    sleep 600
  done
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  offers=$(bash $V search offers 'gpu_name=RTX_3090 num_gpus=1 reliability>0.97 dph<=0.30 disk_space>=40 rentable=true inet_down>=300 cuda_vers>=12.4' -o dph --raw --limit 150 2>/dev/null | python -c "
import json,sys
ex={l.strip() for l in open('scratch/vast_exclude_ips.txt') if l.strip() and not l.startswith('#')}
used={l.split()[0] for l in open('scratch/live_hosts.txt') if l.strip()}; seen=set(); rows=[]
for o in json.load(sys.stdin):
    ip=o.get('public_ipaddr'); dv=str(o.get('driver_version',''))
    if ip in ex or ip in used or ip in seen or dv.startswith('580'): continue
    seen.add(ip); rows.append((0 if dv.startswith('595.71') else 1, o['dph_total'], o['id']))
print(' '.join(str(r[2]) for r in sorted(rows)[:2]))")
  set -- $offers
  if [ $# -lt $need ]; then log "seed $S: fewer than $need offers, retry in 10 min"; sleep 600; continue; fi
  oT=""; oB=""; [ $okT = 0 ] && { oT=$1; shift; }; [ $okB = 0 ] && oB=$1
  log "seed $S: deploying${oT:+ TSD b$nt (offer $oT)}${oB:+ BASE b$nb (offer $oB)}; $n live, \$${burn}/h"
  if [ -z "$tsd_ids" ]; then XT="boxes/teacher135.py scratch/wait_then.py $T/model.pt $T/target_means.pt $T/args.json"
  else XT="boxes/teacher135.py scratch/wait_then.py $T/target_means.pt $T/args.json"; fi
  [ -n "$oT" ] && { EXTRA_120="$XT" bash scratch/gen_deploy_120.sh "$oT $nt" & }
  [ -n "$oB" ] && { EXTRA_120="boxes/teacher135.py $T/target_means.pt $T/args.json" bash scratch/gen_deploy_120.sh "$oB $nb" & }
  wait
  [ -n "$oB" ] && started $nb $oB && okB=1
  if [ -n "$oT" ] && started $nt $oT; then
    okT=1; id=$(grep "gen host b$nt offer $oT:" logs/gen_deploy.log | tail -1 | grep -o "INSTANCE [0-9]*" | awk '{print $2}')
    if [ -n "$tsd_ids" ]; then                                              # host-to-host from a live TSD host, else from local
      src=""; for s_ in $tsd_ids; do bash $V show instance $s_ --raw 2>/dev/null | grep -q '"actual_status": *"running"' && { src=$s_; break; }; done
      if [ -n "$src" ]; then log "seed $S: vastai copy C.$src:$TD/ -> C.$id:$TD/"; timeout 1800 bash $V copy "C.$src:$TD/" "C.$id:$TD/" >> logs/fill135.out 2>&1
      else log "seed $S: no TSD source up; copying from local"; timeout 3600 bash $V copy "local:$T/" "C.$id:$TD/" >> logs/fill135.out 2>&1; fi
    fi
    tsd_ids="$tsd_ids $id"
  fi
  bash scratch/rebuild_live_hosts.sh > /dev/null 2>&1
  if [ $okT = 1 ] && [ $okB = 1 ]; then S=$((S + 1)); okT=0; okB=0; sleep 120; else log "seed $S: TSD ok=$okT BASE ok=$okB - retrying the missing arm"; fi
done
log "all 4 student pairs deployed"
