#!/usr/bin/env bash
# §76(B)/(B-std) feeder (local cards): every 60 s, queue both students for each CERTIFIED S4 teacher in
# theory/results/c1_tab_local/ (ge3 -> p76_ge3_*, std -> p76_std_*), pilot first (the first certified teacher of seed
# 2051, else the first certified teacher at all; others wait until the pilot's two eval.json exist), and start a
# py_local_worker on each card that is free. Stop: touch scratch/.stop_feed76. Log: logs/feed76.log.
cd "$(dirname "$0")/.."
Q=scratch/c1_q_76_local.txt; [ -f $Q ] || echo "# §76(B) local queue (written by scratch/feed76.sh)" > $Q
while [ ! -f scratch/.stop_feed76 ]; do
  python - <<'PYEOF' >> logs/feed76.log 2>&1
import glob, json, os, re, time
Q = "scratch/c1_q_76_local.txt"; have = open(Q).read()
cert = []
for f in sorted(glob.glob("theory/results/c1_tab_local/S4_*_s*.json")):
    m = re.search(r"S4_(ge3|std)_s(\d+)\.json$", f)
    if not m:
        continue
    try:
        t = next(iter(json.load(open(f)).values()))
    except Exception:
        continue                                                  # still being written
    if t.get("certified"):
        cert.append((m.group(1), int(m.group(2)), f))
if cert:
    pil = next((c for c in cert if c[1] == 2051), cert[0])
    pil_done = all(os.path.exists(f"c1/runs/p76_{pil[0]}_{s}_s{pil[1]}/eval.json") for s in ("pfstd", "ptstd"))
    for cur, seed, f in cert:
        if (cur, seed, f) != pil and not pil_done:
            continue
        for s in ("pfstd", "ptstd"):
            out = f"c1/runs/p76_{cur}_{s}_s{seed}"
            if out + " " not in have:
                open(Q, "a", newline="\n").write(f"{out} c1.amortise --teacher {f} --student {s}\n")
                print(time.strftime("%H:%M:%S"), "queued", out, flush=True)
PYEOF
  for d in cuda:0 cuda:1 cuda:2; do
    free=1
    [ -f scratch/.w76_$d ] && kill -0 "$(cat scratch/.w76_$d)" 2>/dev/null && free=0
    [ $d = cuda:2 ] && ! grep -q "p75_pfstd_s2033 rc" logs/c1_local.log && free=0        # descriptive §75 run still there
    if [ $free = 1 ]; then
      todo=$(grep -v '^#' $Q | while read -r out rest; do n=$(basename $out); [ -s $out/eval.json ] || [ -d c1/runs/locks/$n ] || echo $n; done | head -n 1)
      if [ -n "$todo" ]; then
        C1_GROUP=S4 nohup bash scratch/py_local_worker.sh $d $Q > /dev/null 2>&1 & echo $! > scratch/.w76_$d
        echo "$(date +%T) worker $d started (next $todo)" >> logs/feed76.log
      fi
    fi
  done
  sleep 60
done
echo "$(date +%T) FEED_STOPPED" >> logs/feed76.log
