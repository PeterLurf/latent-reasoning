#!/bin/bash
# §223 Note 223-10 pilot P3a ($0, local; §223's last iteration): DIRECT only on the RTX 4060 (cuda:0, the registered kernel), the frozen DIRECT flags +
# --lr 3e-4 --cg_thr 0.45 --cg_cap 6000 --steps 48000 (the LR schedule) --stop_at 12000, full-state ckpt every 2,000 (--resume restarts it), --auto_micro
# --mem_budget 7.0 -> scratch/_tmp/s223_p3a/direct. Early stop at the advance into stage 2 (d <= 3): once stages.jsonl holds an advance "to": 3, the
# run is stopped (by pid) after the next 100-step log record lands, so the advance's bins / entry are logged. Then scratch/s223_pilot_summary.py --p3a
# -> scratch/gate_s223/pilot3a.json (stages, entry_step, num_acc / num_ce 1k bins, gn_preclip rows with the first window flagged, non-finite rows).
#   nohup bash scratch/s223_p3a.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -q "def soft_decoder" boxes/codi_ps.py || { echo "$(date '+%F %T') s223 P3a REFUSED: boxes/codi_ps.py not at the P2 build"; exit 2; }
DV=cuda:0; O=scratch/_tmp/s223_p3a/direct; L=logs/s222_chain.out; RES=""
if [ -e "$O/ckpt_last.pt" ]; then RES="--resume"; else [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"; fi
mkdir -p "$O"; echo "$(date '+%F %T') s223 P3a direct start ($DV, kernel registered${RES:+, resume})" >> $L
python -B boxes/codi_ps.py --direct --lr 3e-4 --cg_thr 0.45 --cg_cap 6000 --steps 48000 --stop_at 12000 --device $DV --out $O --auto_micro --mem_budget 7.0 $RES >> logs/s223_p3a_direct.log 2>&1 &
PID=$!; stopped=""
while kill -0 $PID 2>/dev/null; do
  if [ -z "$stopped" ] && grep -q '"to": 3' "$O/stages.jsonl" 2>/dev/null; then
    adv=$(python -c "import json; print([json.loads(l) for l in open('$O/stages.jsonl') if json.loads(l)['to'] == 3][0]['step'])")
    last=$(tail -n 1 "$O/log.jsonl" 2>/dev/null | python -c "import json, sys; print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo 0)
    if [ "$last" -ge "$adv" ]; then
      wpid=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 's223_p3a/direct' }).ProcessId" | tr -d '\r' | head -1)
      echo "$(date '+%F %T') s223 P3a EARLY STOP: advance into stage 2 (d 3) at step $adv; log at $last; stopping pid $wpid" >> $L
      [ -n "$wpid" ] && powershell -NoProfile -Command "Stop-Process -Id $wpid -Force"; stopped=1
    fi
  fi
  sleep 30
done
wait $PID 2>/dev/null; echo "$(date '+%F %T') s223 P3a direct ended (${stopped:+early stop at the stage-2 advance}${stopped:-rc / stop_at 12000})" >> $L
printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "early_stop_stage2": %s}\n' "$DV" "$([ -n "$stopped" ] && echo true || echo false)" > $O/cell_meta.json
python -B scratch/s223_pilot_summary.py cuda:0 registered --p3a >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 P3a summary rc $? -> scratch/gate_s223/pilot3a.json" >> $L
