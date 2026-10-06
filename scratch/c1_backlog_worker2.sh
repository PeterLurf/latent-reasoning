#!/usr/bin/env bash
# v2: exports LATENT_MEM_CAP_GB (default 6.0 GiB of PyTorch memory; + CUDA context keeps >= 1.5 GiB of an 8 GB card free) - jobs OOM
# inside their own budget instead of overfilling. Local standing-backlog worker, one per card (user 2026-10-06: "never let the local gpus be idle, always have some small experiments or
# prototypes running" + "only put processes on local cards that wont overfill"). Pulls the next runnable line of a backlog file, one job
# at a time, nothing beside it; when nothing is runnable it polls every 2 min, so lines appended later start automatically.
# Backlog lines:  <measured peak GiB> <done file> <command ... {DEV} ...>   ('#' comments ignored)
#   - <measured peak GiB>: card memory of this job type from a memory check / earlier gated run; lines with peak > total - 1.5 GiB are
#     refused (logged once). Card memory is sampled every 2 s while the job runs; a peak over the limit halts the worker (<tag>.halt).
#   - <done file>: the line is skipped once it exists (non-empty); claims via lock dirs c1/runs/locks/bl_<line hash> (shared backlogs: several cards may pull from one file).
# Optional --after <windows pid>: start only after that process exits (PowerShell check; MSYS kill -0 cannot see Windows pids).
# Stop: touch scratch/<tag>.stop (finishes the current job).   nohup bash scratch/c1_backlog_worker.sh <tag> <torch dev> "<GPU name>" <backlog> [--after pid] &
cd "$(dirname "$0")/.."
export LATENT_MEM_CAP_GB=${LATENT_MEM_CAP_GB:-6.0}                         # v2 (Audit 2026-10-06): hard per-process cap (boxes/memcap.py) on every job
tag=$1; dev=$2; gname=$3; bl=$4; after=""; [ "$5" = "--after" ] && after=$6
alive() { powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" > /dev/null 2>&1; }
if [ -n "$after" ]; then while alive "$after"; do sleep 60; done; fi
idx=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep "$gname" | head -1 | cut -d, -f1)
total=$(nvidia-smi --id=$idx --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' \r'); limit=$((total - 1536))
mkdir -p c1/runs/locks; echo "$(date +%T) $tag backlog worker v2 start (cap ${LATENT_MEM_CAP_GB} GiB): $bl on $dev (nvidia-smi $idx '$gname', limit $limit MiB of $total)" >> logs/c1_local.log
refused=" "; idle_logged=0; halted=0
while [ ! -f scratch/$tag.stop ] && [ ! -f scratch/$tag.halt ]; do
  ran=0
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue;; esac
    [ -f scratch/$tag.stop ] && break
    mem=$(echo "$line" | awk '{print $1}'); done_f=$(echo "$line" | awk '{print $2}'); cmd=$(echo "$line" | cut -d' ' -f3-)
    [ -s "$done_f" ] && continue
    h=$(echo "$line" | md5sum | cut -c1-12)
    memmib=$(python -c "print(int(float('$mem') * 1024))" 2>/dev/null || echo 999999)
    if [ "$memmib" -gt "$limit" ]; then
      case "$refused" in *" $h "*) ;; *) echo "$(date +%T) $tag REFUSED (measured $mem GiB > limit): $cmd" >> logs/c1_local.log; refused="$refused$h ";; esac; continue
    fi
    mkdir c1/runs/locks/bl_$h 2>/dev/null || continue
    c=${cmd//\{DEV\}/$dev}; n=$(echo "$done_f" | tr '/' '_' | tail -c 80)
    echo "$(date +%T) $tag start: $c" >> logs/c1_local.log
    OMP_NUM_THREADS=3 bash -c "$c" > logs/c1_bl_${tag}_$n.log 2>&1 &
    pid=$!; peak=0
    while kill -0 $pid 2>/dev/null; do                                    # own child (MSYS pid): kill -0 is valid here
      u=$(nvidia-smi --id=$idx --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'); [ "$u" -gt "$peak" ] 2>/dev/null && peak=$u; sleep 2
    done
    wait $pid; rc=$?
    echo "$(date +%T) $tag done rc $rc out $([ -s "$done_f" ] && echo yes || echo NO) peak_card_mem $peak MiB (measured $mem GiB) : $done_f" >> logs/c1_local.log
    if [ "$peak" -gt "$limit" ]; then echo "$(date +%T) $tag HALT: peak $peak MiB > limit $limit" >> logs/c1_local.log; touch scratch/$tag.halt; halted=1; break; fi
    ran=1; idle_logged=0; break                                           # re-read the backlog after every job (it may have grown)
  done < "$bl"
  [ "$halted" = 1 ] && break                                              # v2: a halt ends the worker (v1 resumed if the halt file was removed during its idle sleep)
  if [ $ran = 0 ]; then [ $idle_logged = 0 ] && { echo "$(date +%T) $tag IDLE: backlog empty (polling)" >> logs/c1_local.log; idle_logged=1; }; sleep 120; fi
done
echo "$(date +%T) $tag backlog worker exit ($([ -f scratch/$tag.halt ] && echo halt || echo stop))" >> logs/c1_local.log
