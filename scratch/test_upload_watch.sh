#!/usr/bin/env bash
# Stub test of scratch/upload_watch.sh under set -u (Audit 10-27): scp is a stub executable that copies the file to a LOCAL "remote" path at a controlled rate,
# S reads the local size; short windows (UP_POLL_S 1, UP_WIN_S 4, UP_MIN_B 200000, UP_MAX_S 12). Cases:
#   fast      2 MB at ~1 MB/s                         -> rc 0, file complete
#   stalled   2 MB at ~20 KB/s (< 200 KB per 4 s)      -> rc 4, UPLOAD_STALLED (window), scp killed
#   slowmax   4 MB at ~100 KB/s (>= 200 KB / 4 s, needs ~40 s > UP_MAX_S 12) -> rc 4, UPLOAD_STALLED (max)
#   scpfail   scp exits 1 at once                      -> rc 1 (the caller's sha check / retry path)
# A killed upload must stop: the dst file size is unchanged 3 s after upload_watch returns. The stub is an executable named scp (one process, like the real one).
set -u
cd "$(dirname "$0")/.." || exit 2
. scratch/upload_watch.sh
T=scratch/_tmp/upload_watch_test; rm -rf "$T"; mkdir -p "$T"; head -c 4000000 /dev/urandom > "$T/src.bin"; head -c 2000000 "$T/src.bin" > "$T/src2.bin"
mkdir -p "$T/bin"; cat > "$T/bin/scp" <<'STUB'
#!/usr/bin/env bash
# stub scp: a single process (exec python) like the real one; last arg root@host:path -> local path, copied in 10 KB chunks at $RATE bytes/s
src="${@: -2:1}"; dst="${@: -1}"; dst=${dst#*:}; [ "$RATE" = fail ] && exit 1
exec python -c "
import sys, time
src, dst, rate = sys.argv[1], sys.argv[2], float(sys.argv[3]); data = open(src, 'rb').read(); f = open(dst, 'wb')
for i in range(0, len(data), 10000):
    f.write(data[i:i + 10000]); f.flush(); time.sleep(10000 / rate)
f.close()" "$src" "$dst" "$RATE"
STUB
chmod +x "$T/bin/scp"; export PATH="$PWD/$T/bin:$PATH"; export RATE=1000000
S="bash -c"; UP_POLL_S=1; UP_WIN_S=4; UP_MIN_B=200000; UP_MAX_S=12; ok=1
case_() {   # name file rate expect_rc expect_text
  export RATE=$3; UP_T0=$(date +%s); out=$(upload_watch "$2" 0 host "$T/dst_$1.bin"); rc=$?; sz=$(stat -c %s "$T/dst_$1.bin" 2>/dev/null || echo 0)
  sleep 3; sz2=$(stat -c %s "$T/dst_$1.bin" 2>/dev/null || echo 0); left=$(( sz2 != sz ))   # killed = the file stops growing
  echo "$1: rc $rc size $sz left_procs $left out [$out]"
  [ "$rc" = "$4" ] && { [ -z "$5" ] || echo "$out" | grep -q "$5"; } && [ "$left" = 0 ] || { echo "  FAIL $1"; ok=0; }
}
case_ fast "$T/src2.bin" 1000000 0 ""
case_ stalled "$T/src2.bin" 20000 4 "bytes in"
case_ slowmax "$T/src.bin" 100000 4 "no completion within"
case_ scpfail "$T/src2.bin" fail 1 ""
[ "$(stat -c %s "$T/dst_fast.bin")" = 2000000 ] || { echo "  FAIL fast size"; ok=0; }
rm -rf "$T"; [ "$ok" = 1 ] && echo "UPLOAD_WATCH TEST PASS" || { echo "UPLOAD_WATCH TEST FAIL"; exit 1; }
