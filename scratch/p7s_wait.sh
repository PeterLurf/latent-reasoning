#!/usr/bin/env bash
# Wake on the next event worth acting on (2026-09-29): P eval done, 4060 O queue done, Q pod finished/failed,
# vast3 T9 queue done/failed, or a new local training failure. Prints the event and exits.
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=20 -i $HOME/.ssh/runpod_ed25519"
fails0=$(cat logs/queue_*.out 2>/dev/null | grep -ac "rc [1-9]")
for i in $(seq 1 200); do
  grep -q P_EVAL_DONE logs/p_eval.out && [ ! -f results/.p_scored ] && { echo "EVENT P_EVAL_DONE"; exit 0; }
  grep -q QUEUE_DONE logs/queue_o_4060.out && [ ! -f results/.o_scored ] && { echo "EVENT O_4060_DONE"; exit 0; }
  f=$(cat logs/queue_*.out 2>/dev/null | grep -ac "rc [1-9]"); [ "$f" -gt "$fails0" ] && { echo "EVENT LOCAL_RC_FAIL"; exit 0; }
  for x in "96.20.36.104 40044 q0" "142.198.191.106 40409 q1"; do set -- $x
    [ -f results/.q_$3_fetched ] && continue
    s=$(timeout 40 $S -p $2 root@$1 "grep -aE 'ANALYSIS_DONE|NO_FINAL|Error|FAIL ' /workspace/logs/prosqa.out | tail -2" 2>/dev/null)
    [ -n "$s" ] && { echo "EVENT Q $3: $s"; exit 0; }
  done
  [ ! -f results/.t9v3_fetched ] && { s=$(timeout 40 $S -p 1743 root@174.0.242.177 "grep -aE 'QUEUE_DONE|rc [1-9]' /workspace/Latent/logs/queue_t9_vast3.out" 2>/dev/null); [ -n "$s" ] && { echo "EVENT T9_VAST3: $s"; exit 0; }; }
  sleep 300
done
echo WAIT_TIMEOUT
