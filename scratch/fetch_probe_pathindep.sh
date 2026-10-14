#!/usr/bin/env bash
# Theory / Audit 2026-10-13: path_indep trajectory probe on a RUNNING cloud cell without touching its card - read-only: wait until the host's
# ckpt_last.pt mtime is stable for 30 s (never copy a half-written file), stream it + args.json down (tar to stdout), run scratch/probe_pathindep.py on
# a local card, delete the local checkpoint copy. Output: boxes/runs_fsq/<run>/pathindep_<step>.json (+ the run's args.json if absent locally).
# The host's log.jsonl comes down with it (to the temp dir only, never over the local run log) for the probe's 1k training-loss bin; PROBE_SEEN=yes|no
# records whether that run's training loss at this step had been looked at before the probe; PROBE_STAGE=nmax:K|auto adds the stage-matched probe.
#   [PROBE_SEEN=no] [PROBE_STAGE=...] bash scratch/fetch_probe_pathindep.sh <host> <port> <run name> [local device, default cuda:1]
set -u
cd "$(dirname "$0")/.."
H=$1; P=$2; R=$3; DEV=${4:-cuda:1}; SP=${PROBE_TMP:-/c/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/probe_ck}
SSH="ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
for i in $(seq 1 40); do
  a=$(timeout 40 $SSH "stat -c %Y /workspace/Latent/boxes/runs_fsq/$R/ckpt_last.pt" 2>/dev/null | tail -n 1); sleep 30
  b=$(timeout 40 $SSH "stat -c %Y /workspace/Latent/boxes/runs_fsq/$R/ckpt_last.pt" 2>/dev/null | tail -n 1)
  [ -n "$a" ] && [ "$a" = "$b" ] && break
done
[ -n "${a:-}" ] && [ "$a" = "${b:-}" ] || { echo "$(date +%T) $R: ckpt_last not stable / missing"; exit 1; }
rm -rf "$SP"; mkdir -p "$SP" boxes/runs_fsq/$R
timeout 3000 $SSH "cd /workspace/Latent/boxes/runs_fsq/$R && tar -cf - ckpt_last.pt args.json log.jsonl" 2>/dev/null | tar -xf - -C "$SP" || { echo "$(date +%T) $R: fetch failed"; rm -rf "$SP"; exit 1; }
[ -s boxes/runs_fsq/$R/args.json ] || cp "$SP/args.json" boxes/runs_fsq/$R/args.json
python scratch/probe_pathindep.py boxes/runs_fsq/$R "$SP/ckpt_last.pt" $DEV --log "$SP/log.jsonl" --loss_seen "${PROBE_SEEN:-unrecorded}" 2>&1 | grep -E "PATHINDEP|Error|Traceback"
[ -n "${PROBE_STAGE:-}" ] && python scratch/probe_pathindep.py boxes/runs_fsq/$R "$SP/ckpt_last.pt" $DEV --no_aux --stage "$PROBE_STAGE" --log "$SP/log.jsonl" --loss_seen "${PROBE_SEEN:-unrecorded}" 2>&1 | grep -E "PATHINDEP|Error|Traceback"
rm -rf "$SP"; echo "$(date +%T) $R: probe done, local ckpt copy deleted"
