#!/usr/bin/env bash
# v6 item 4 fixture: dummy processes with the real command-line shapes for cell b999 (family s999) and a decoy cell b998, in a fake root; void_cell.sh s999 999 2
# must kill all five b999 processes, leave the b998 decoys alive, rotate the logs, void / retag the ledger and keep the queue.
set -u
cd "$(dirname "$0")/../.." || exit 2
ROOTD=$(pwd); R=$ROOTD/scratch/_tmp/voidfx; rm -rf "$R"; mkdir -p "$R/scratch" "$R/logs"
for s in s999_post_admit.sh auto_drain_gen10_s267.sh; do printf '#!/usr/bin/env bash\nsleep 300\n' > "$R/scratch/$s"; done
for s in cost_guard.py wall_guard_v6.py ckpt_mirror.py; do printf 'import time\ntime.sleep(300)\n' > "$R/scratch/$s"; done
cp scratch/void_cell.sh "$R/scratch/"
cd "$R"
for n in 999 998; do
  nohup bash scratch/s999_post_admit.sh $n run h p 1 > /dev/null 2>&1 &
  nohup python scratch/cost_guard.py s999_b$n 0.5 100 x > /dev/null 2>&1 &
  nohup python scratch/wall_guard_v6.py s999_wall_b$n 0.5 1 h p 0.2 0 gen_b$n 2 0.4 > /dev/null 2>&1 &
  nohup python scratch/ckpt_mirror.py "$n:run" --every_h 0.3 > /dev/null 2>&1 &
  nohup bash scratch/auto_drain_gen10_s267.sh 1.2.3.4 22 gen_b$n 77 s267 > /dev/null 2>&1 &
  for f in s999_post_b$n.out s999_guard_b$n.out s999_wall_b$n.out ckpt_mirror_b$n.out auto_drain_gen_b$n.out; do echo x > logs/$f; done
done
printf '%s\n' "# ledger" "999:run 1.2.3.4:22 77 \$0.2/h sps 1.3 steps 2000 0.4" "realised:b999 # landed 0.21" "998:run 1.2.3.4:22 78 \$0.2/h sps 1.3 steps 2000 0.4" > logs/s999_admitted.log
echo "q" > scratch/gen_q_b999.txt
sleep 3
cnt() { ps -ef | grep -E "$1" | grep -v -E 'grep|void_cell' | wc -l; }
before999=$(cnt " (s999_([a-z_]+_)?b999|gen_b999|999:run|scratch/s999_post_admit\.sh 999)"); before998=$(cnt " (s999_([a-z_]+_)?b998|gen_b998|998:run|scratch/s999_post_admit\.sh 998)")
out=$(bash scratch/void_cell.sh s999 999 2); rc=$?
after999=$(cnt " (s999_([a-z_]+_)?b999|gen_b999|999:run|scratch/s999_post_admit\.sh 999)"); after998=$(cnt " (s999_([a-z_]+_)?b998|gen_b998|998:run|scratch/s999_post_admit\.sh 998)")
ok=1
[ "$before999" = 5 ] && [ "$before998" = 5 ] || { echo "FIXTURE setup: expected 5 + 5 dummies, saw $before999 + $before998"; ok=0; }
[ $rc = 0 ] && [ "$after999" = 0 ] || { echo "FAIL: b999 processes left ($after999), rc $rc"; ok=0; }
[ "$after998" = 5 ] || { echo "FAIL: decoy b998 hit ($after998 left of 5)"; ok=0; }
[ -e logs/s999_wall_b999.out.attempt2 ] && [ ! -e logs/s999_wall_b999.out ] && [ -e logs/s999_wall_b998.out ] || { echo "FAIL: log rotation"; ok=0; }
grep -q "^# VOID attempt2 999:run" logs/s999_admitted.log && grep -q "^realised:b999:a2 " logs/s999_admitted.log && grep -q "^998:run" logs/s999_admitted.log || { echo "FAIL: ledger"; ok=0; }
[ -e scratch/gen_q_b999_attempt2.txt ] || { echo "FAIL: queue copy"; ok=0; }
for p_ in $(ps -ef | grep -E " (s999_([a-z_]+_)?b998|gen_b998|998:run|scratch/s999_post_admit\.sh 998)" | grep -v grep | awk '{print $2}'); do kill $p_ 2>/dev/null; done
echo "$out" | tail -3
[ $ok = 1 ] && echo "FIXTURE void_cell PASS" || echo "FIXTURE void_cell FAIL"
