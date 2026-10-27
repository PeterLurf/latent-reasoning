#!/usr/bin/env bash
# Upload stall watchdog for scratch/vdeploy_toy.sh (Audit 10-27, after the §237 b354 stall: 79 MB of 1.4 GB in 54 min at ~25 KB/s, billed until killed).
# Sourced, not run. upload_watch <local file> <port> <host> <remote path>: scp in the background; every UP_POLL_S (60) s the remote size is read through the deploy's
# $S ssh prefix. Kill + return 4 (prints UPLOAD_STALLED) when < UP_MIN_B (20,000,000) bytes arrived over a UP_WIN_S (600) s window, or when UP_MAX_S (2700) s have
# passed since UP_T0 (the first try's start, set by the caller) without completion. A failed size probe (empty) is 'unknown': that poll skips the window
# check and keeps w0 / b0 (Audit 10-27). Otherwise the scp's own exit status. Works under set -u.
#   test: bash scratch/test_upload_watch.sh
upload_watch() {
  local f=$1 p=$2 h=$3 r=$4 pid w0 b0 b now
  scp -q -i ~/.ssh/runpod_ed25519 -P "$p" "$f" "root@$h:$r" & pid=$!
  w0=$(date +%s); b0=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${UP_POLL_S:-60}"
    kill -0 "$pid" 2>/dev/null || break
    now=$(date +%s); b=$($S "stat -c %s $r 2>/dev/null" 2>/dev/null | tr -dc 0-9)   # empty = the probe failed: unknown (no window check, no reset)
    if [ $(( now - UP_T0 )) -ge "${UP_MAX_S:-2700}" ]; then
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; echo "UPLOAD_STALLED (no completion within ${UP_MAX_S:-2700} s of the first try; ${b:-unknown} bytes on the host)"; return 4
    fi
    if [ -n "$b" ] && [ $(( now - w0 )) -ge "${UP_WIN_S:-600}" ]; then   # Audit 10-27: a failed probe must not read as 0 bytes (false stall)
      if [ $(( b - b0 )) -lt "${UP_MIN_B:-20000000}" ]; then
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; echo "UPLOAD_STALLED ($(( b - b0 )) bytes in $(( now - w0 )) s < ${UP_MIN_B:-20000000}; $b bytes on the host)"; return 4
      fi
      w0=$now; b0=$b
    fi
  done
  wait "$pid"
}
