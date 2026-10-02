#!/usr/bin/env bash
# §76(B)/(B-std) + 76-3 feeder (local cards): every 60 s, queue both students, held-out and full (76-3) arms, for each
# CERTIFIED teacher: S4 from theory/results/c1_tab_local/ (queue scratch/c1_q_76_local.txt), A5 from
# theory/results/c1_tab_local/ (76-5; queue scratch/c1_q_76_A5.txt, row minibatch --bs 512 for 8 GB cards). Per group, pilot
# first: the first certified teacher of seed 2051 (else the first certified teacher at all); the others wait until the
# pilot's two held-out-arm eval.json exist. Starts a py_local_worker (C1_GROUP set per queue) on each free card, S4 first.
# Stop: touch scratch/.stop_feed76. Log: logs/feed76.log.
cd "$(dirname "$0")/.."
for G in S4 A5; do Q=scratch/c1_q_76_local.txt; [ $G = A5 ] && Q=scratch/c1_q_76_A5.txt
  [ -f $Q ] || echo "# §76(B) $G local queue (written by scratch/feed76.sh)" > $Q; done
unclaimed() { grep -v '^#' $1 | while read -r out rest; do n=$(basename $out); [ -s $out/eval.json ] || [ -d c1/runs/locks/$n ] || echo $n; done | head -n 1; }
while [ ! -f scratch/.stop_feed76 ]; do
  python - <<'PYEOF' >> logs/feed76.log 2>&1
import glob, json, os, re, time
for G, src, Q, extra0, pre in (("S4", "theory/results/c1_tab_local", "scratch/c1_q_76_local.txt", "", "p76"),
                               ("A5", "theory/results/c1_tab_local", "scratch/c1_q_76_A5.txt", " --bs 512", "p76a5")):
    have = open(Q).read(); cert = []
    for f in sorted(glob.glob(f"{src}/{G}_*_s*.json")):
        m = re.search(rf"{G}_(ge3|std)_s(\d+)\.json$", f)
        if not m:
            continue
        try:
            t = next(iter(json.load(open(f)).values()))
        except Exception:
            continue                                                  # still being written
        if t.get("certified"):
            cert.append((m.group(1), int(m.group(2)), f.replace("\\", "/")))
    if not cert:
        continue
    pil = next((c for c in cert if c[1] == 2051), cert[0])
    pil_done = all(os.path.exists(f"c1/runs/{pre}_{pil[0]}_{s}_s{pil[1]}/eval.json") for s in ("pfstd", "ptstd"))
    for cur, seed, f in cert:
        if (cur, seed, f) != pil and not pil_done:
            continue
        for arm, extra in (("", ""), ("full", " --holdout 0")):          # amendment 76-3: descriptive no-holdout arm
            for s in ("pfstd", "ptstd"):
                out = f"c1/runs/{pre}_{cur}{arm}_{s}_s{seed}"
                if out + " " not in have:
                    open(Q, "a", newline="\n").write(f"{out} c1.amortise --teacher {f} --student {s}{extra0}{extra}\n")
                    print(time.strftime("%H:%M:%S"), "queued", out, flush=True)
PYEOF
  for d in cuda:0 cuda:1 cuda:2; do
    free=1
    [ -f scratch/.w76_$d ] && kill -0 "$(cat scratch/.w76_$d)" 2>/dev/null && free=0
    [ -f scratch/.hold76_$d ] && free=0                                                  # manual hold (e.g. other work)
    # hard guard (2026-10-02 11:25 stacking incident): never start if any amortise job already runs on this card
    if [ $free = 1 ] && [ "$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*c1.amortise*--device $d *' }).Count")" != "0" ]; then free=0; fi
    if [ $free = 1 ]; then
      for G in S4 A5; do Q=scratch/c1_q_76_local.txt; [ $G = A5 ] && Q=scratch/c1_q_76_A5.txt
        todo=$(unclaimed $Q)
        if [ -n "$todo" ]; then
          C1_GROUP=$G nohup bash scratch/py_local_worker.sh $d $Q > /dev/null 2>&1 & echo $! > scratch/.w76_$d
          echo "$(date +%T) worker $d $G started (next $todo)" >> logs/feed76.log; break
        fi
      done
    fi
  done
  sleep 60
done
echo "$(date +%T) FEED_STOPPED" >> logs/feed76.log
