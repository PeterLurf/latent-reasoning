#!/usr/bin/env bash
# Tests the ACTUAL upload block of scratch/vdeploy_toy.sh (the lines from LSUM= through UPLOAD_DONE, extracted at test time) under set -u, with the stub scp of
# scratch/test_upload_watch.sh (one process, local "remote") and S = a local shell: (a) normal -> UPLOAD_DONE, sha ok; (b) stalled -> "FAILED upload (stalled)",
# exit 3, the local tarball removed; (c) over UP_MAX_S -> FAILED, exit 3. Short windows via the UP_* env vars.
#   bash scratch/test_vdeploy_upload.sh
set -u
cd "$(dirname "$0")/.." || exit 2
T=scratch/_tmp/vdeploy_upload_test; rm -rf "$T"; mkdir -p "$T/bin" "$T/remote"
awk '/^LSUM=\$\(sha256sum/{f=1} f{print} /^echo "UPLOAD_DONE/{exit}' scratch/vdeploy_toy.sh > "$T/block.sh"
grep -q "upload_watch /tmp/up_" "$T/block.sh" && grep -q "UPLOAD_DONE" "$T/block.sh" || { echo "block not extracted"; exit 1; }
cat > "$T/bin/scp" <<'STUB'
#!/usr/bin/env bash
src="${@: -2:1}"; dst="${@: -1}"; dst=${dst#*:}; dst="$TESTREMOTE$dst"; mkdir -p "$(dirname "$dst")"
exec python -c "
import sys, time
src, dst, rate = sys.argv[1], sys.argv[2], float(sys.argv[3]); data = open(src, 'rb').read(); f = open(dst, 'wb')
for i in range(0, len(data), 10000):
    f.write(data[i:i + 10000]); f.flush(); time.sleep(10000 / rate)
f.close()" "$src" "$dst" "$RATE"
STUB
chmod +x "$T/bin/scp"; export PATH="$PWD/$T/bin:$PATH"; export TESTREMOTE="$PWD/$T/remote"
cat > "$T/sh.sh" <<'SH'
#!/usr/bin/env bash
# the deploy's $S stand-in: run the remote command with /tmp paths mapped under $TESTREMOTE
exec bash -c "${1//\/tmp\//$TESTREMOTE/tmp/}"
SH
chmod +x "$T/sh.sh"; ok=1
run_case() {   # name rate bytes expect_rc expect_text
  LBL="test_$1"; head -c "$3" /dev/urandom > "/tmp/up_$LBL.tgz"; rm -rf "$TESTREMOTE/tmp"
  out=$(RATE=$2 UP_POLL_S=1 UP_WIN_S=4 UP_MIN_B=200000 UP_MAX_S=12 LBL=$LBL P=0 H=host S="$PWD/$T/sh.sh" bash -u -c "source $T/block.sh" scratch/vdeploy_toy.sh 2>&1); rc=$?   # $0 = the real script path: the block sources upload_watch.sh itself
  left=$( [ -f "/tmp/up_$LBL.tgz" ] && echo kept || echo removed )
  echo "$1: rc $rc tarball $left out [$(echo "$out" | tr '\n' ' ' | cut -c1-160)]"
  [ "$rc" = "$4" ] && echo "$out" | grep -q "$5" && [ "$left" = removed ] && ! echo "$out" | grep -q "No such file" || { echo "  FAIL $1"; ok=0; }
  rm -f "/tmp/up_$LBL.tgz"
}
run_case normal 2000000 1000000 0 "UPLOAD_DONE"
run_case stalled 20000 1000000 3 "FAILED upload (stalled)"
run_case slowmax 100000 3000000 3 "FAILED upload (stalled)"
rm -rf "$T"; [ "$ok" = 1 ] && echo "VDEPLOY UPLOAD TEST PASS" || { echo "VDEPLOY UPLOAD TEST FAIL"; exit 1; }
