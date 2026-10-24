#!/bin/bash
c=$(cat scratch/_tmp/s232_guard_smoke/n 2>/dev/null || echo 0); c=$((c + 1)); echo $c > scratch/_tmp/s232_guard_smoke/n
if [ $c -le 2 ]; then echo "STILL_LIVE $1"; exit 1; fi
echo "DESTROYED_VERIFIED $1"; exit 0
