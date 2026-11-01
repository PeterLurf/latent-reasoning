"""s258 filler template = scratch/s257_fill.sh (sed s257 -> s258) + Audit's 11-01 template fixes (each anchor asserted once):
  (1) REFUSE before a cell's deploy unless WALLW's weight count == the queue's non-comment line count;
  (2) at every ADMIT the family guard is (re)started with EVERY cell: admitted ones from the ledger (not '#'-voided, at their admit projection) + this filler's
      unadmitted QS cells (at their estimate); a running sNNNfam guard is stopped by its exact pid (ps fields $7 / $8), its log kept as .part<HHMMSS>;
  (3) 120 s after starting the cell / wall / family guards, each .out must hold a "start:" line and no Traceback, else an ALERT line + HOLD (exit 3; Audit decides).
    python -B scratch/s258_build/mk_s258_fill.py   -> scratch/s258_fill.sh (+ s258_post_admit.sh / s258_chain.sh by sed)"""
import re
import subprocess

src = open("scratch/s257_fill.sh", encoding="utf-8").read().replace("s257", "s258")
src = re.sub(r"(?m)^# §256 DISTRACT runs filler.*$", "# s258 filler template (= s257_fill.sh + Audit 11-01 guard fixes: WALLW count refusal, family guard over every cell, guard-health check after ADMIT).", src)
E = [
    ('''  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }\n''',
     '''  [ -f scratch/gen_q_b$n.txt ] || { log "b$n: scratch/gen_q_b$n.txt missing (still .hold?) - not starting"; exit 2; }
  nlq=$(grep -vc '^#' "scratch/gen_q_b$n.txt"); nw=$(echo "$WALLW" | tr ',' '\\n' | grep -c .)   # Audit 11-01 (1): one wall-guard weight per queue line
  [ "$nlq" = "$nw" ] || { log "REFUSED b$n: WALLW has $nw weights, the queue $nlq lines"; exit 2; }
'''),
    ('''      if ! ps -ef | grep "cost_guard.py s258fam " | grep -v grep > /dev/null; then
        fcap=$(python -c "print(round($CAPF - $(outside), 4))")
        GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py s258fam "$fcap" "$STEPS" $FAMC > "$FAMGUARD_LOG" 2>&1 &
        log "guard s258fam (\\$$fcap = \\$$CAPF - outside \\$$(outside), GUARD_STOP=1;$FAMC) started"
      fi
''',
     '''      # Audit 11-01 (2): the family guard over EVERY cell (admitted: the ledger at their projection; unadmitted QS cells: their estimate), restarted at each ADMIT
      for fp in $(ps -ef | awk '$7 == "scratch/cost_guard.py" && $8 == "s258fam" {print $2}'); do kill "$fp"; done
      [ -f "$FAMGUARD_LOG" ] && mv "$FAMGUARD_LOG" "${FAMGUARD_LOG%.out}.part$(date +%H%M%S).out"
      ALLC=$(awk -v S="$STEPS" '!/^#/ && /^[0-9]+:/ {split($1, a, ":"); print a[1] ":" a[2] ":" $NF ":" S}' "$ADMLOG" | tr '\\n' ' ')
      for c_ in $FAMC; do grep -q "^${c_%%:*}:" "$ADMLOG" || ALLC="$ALLC $c_"; done
      fcap=$(python -c "print(round($CAPF - $(outside), 4))")
      GUARD_STOP=1 GUARD_MARKS=$GM nohup python scratch/cost_guard.py s258fam "$fcap" "$STEPS" $ALLC > "$FAMGUARD_LOG" 2>&1 &
      log "guard s258fam (\\$$fcap = \\$$CAPF - outside \\$$(outside), GUARD_STOP=1; $ALLC) (re)started"
'''),
    ('''      sleep 420; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break''',
     '''      sleep 120                                                             # Audit 11-01 (3): every guard alive 120 s after ADMIT
      for gf in "logs/s258_guard_b$n.out" "logs/s258_wall_b$n.out" "$FAMGUARD_LOG"; do
        { grep -q "start:" "$gf" && ! grep -q "Traceback" "$gf"; } || { log "ALERT guard health: $gf has no start: line or a Traceback - HOLD; tell Audit"; exit 3; }
      done
      log "b$n: guard health OK (cell / wall / family started, no traceback)"
      sleep 300; mv "scratch/gen_q_b$n.txt.norequeue" "scratch/gen_q_b$n.txt"; famcheck "after ADMIT (b$n)"; break'''),
]
for o, n in E:
    assert src.count(o) == 1, ("s258 template anchor", src.count(o), o[:70]); src = src.replace(o, n)
open("scratch/s258_fill.sh", "w", encoding="utf-8", newline="\n").write(src)
for f in ("post_admit", "chain", "watch"):
    open("scratch/s258_%s.sh" % f, "w", encoding="utf-8", newline="\n").write(open("scratch/s257_%s.sh" % f, encoding="utf-8").read().replace("s257", "s258"))
print(subprocess.run(["bash", "-n", "scratch/s258_fill.sh"]).returncode == 0 and "s258 template written, bash -n OK")
