# §224 done markers (FH_v1 §6 Stage 1; sourced by scratch/drain_gen_partial_s224.sh locally AND sent verbatim to the host). A §224 cloud run dir is one of:
#   <run>        a trained boxes.vbank cell: eval.json + model.pt, then boxes.vbank_eval's eval_vb.json (acc, stall) and boxes.vbank_diag's eval_khorizon.json
#   <run>_re     an eval-only dir (s224_ftctrl_s1_re: the shipped s215_ftctrl_s1 model): eval_vb.json + eval_khorizon.json (model.pt is not fetched back)
#   <run>_gtime  G-TIME (--time_steps): timing.json;   <run>_mem  K7 (--mem_check): mem_check.json
#   <run>_eqA / <run>_eqB  G-FH-EQ (--stop_at 1200): log.jsonl holding the step-1200 record
# s267_mark <run dir> -> RUN (a marker present) | BADM (no marker, a resume checkpoint ckpt_last.pt present) | BAD (neither)
s267_mark() {
  case "$1" in *_re) [ -s "$1/eval_khorizon.json" ] && [ -s "$1/eval_vb.json" ] && echo RUN || echo BAD; return ;; esac   # the shipped model.pt is no marker
  case "$1" in *_gates) ls "$1"/*.json > /dev/null 2>&1 && echo RUN || echo BAD; return ;; esac   # 11-04: per-host gate dirs (gate_s271p / gate_s274): any JSON record; b418's was left on the host
  case "$1" in *_c1) [ -s "$1/eval_c1hold.json" ] && [ -s "$1/eval_vb.json" ] && echo RUN || echo BAD; return ;; esac   # §232 Stage A staged eval-only dirs
  if [ -s "$1/eval_khorizon.json" ] || [ -s "$1/model.pt" ] || [ -s "$1/timing.json" ] || [ -s "$1/mem_check.json" ]; then echo RUN
  elif case "$1" in *_eqA|*_eqB) true;; *) false;; esac && [ -s "$1/log.jsonl" ] && grep -q '"step": 1200,' "$1/log.jsonl"; then echo RUN   # gen10: G-FH-EQ dirs only
  elif [ -s "$1/ckpt_last.pt" ]; then echo BADM
  else echo BAD; fi
}
# s267_need <run dir> -> the files a fetched RUN must hold locally (a trained cell whose evals failed still lands: its evals re-run later)
s267_need() {
  case "$1" in
    *_re) echo "eval_vb.json eval_khorizon.json" ;;
    *_c1) echo "eval_vb.json eval_d3hist.json eval_c1hold.json" ;;   # §232 Stage A
    *_gtime) echo "timing.json args.json" ;;
    *_mem) echo "mem_check.json" ;;
    *_gates) echo "" ;;   # 11-04: gate outputs differ by gate (GATE-FAIL has no gates_pass.json); s267_mark already required a JSON   # §228 (10-23): inject_ow --mem_check exits before args.json (b341 FETCH_INCOMPLETE)
    *_eqA|*_eqB) echo "log.jsonl args.json" ;;
    *) echo "eval.json model.pt log.jsonl args.json" ;;
  esac
}
# s267_check_fetched <run dir> -> OK, or MISSING <files>
s267_check_fetched() {
  local m="" f
  for f in $(s267_need "$1"); do [ -s "$1/$f" ] || m="$m $f"; done
  [ -z "$m" ] && echo OK || echo "MISSING$m"
}
# s267_resume_ckpt <run dir> -> the full-state checkpoint a --resume continues from (ckpt_last.pt), else nothing
s267_resume_ckpt() {
  [ -s "$1/ckpt_last.pt" ] && echo "$1/ckpt_last.pt"
}

# gen10 (Audit 11-02) s267_nofetch <n>: resolve the queue's '# NOFETCH <path>' lines (scratch/gen_q_b<n>.txt*, run on the HOST from /workspace/Latent) into
#   NOFETCH <path>                      safe to delete on the host before the fetch, or
#   NOFETCH_REFUSED <path> <reasons>    kept and fetched as usual: (a) train_model = a model.pt in a dir whose queue line trains (--mode train / a boxes.inject_aw line);
#                                       (a) init_or_model = the path is another line's --init / --model / --init_model / --init_trunk value; (b) no_eval_output = the dir
#                                       holds no eval.json / eval_*.json yet.
s267_nofetch() {
  local q p d b why
  q=$(ls scratch/gen_q_b$1.txt* 2>/dev/null | head -1); [ -n "$q" ] || return 0
  for p in $(grep -h '^# NOFETCH ' "$q" | awk '{print $3}' | sort -u); do
    d=$(dirname "$p"); b=$(basename "$p"); why=""
    if [ "$b" = model.pt ] && grep -v '^#' "$q" | awk -v d="$d" '$1 == d' | grep -q -E -- '--mode train|^[^ ]+ boxes\.inject_aw '; then why="train_model"; fi
    if grep -v '^#' "$q" | grep -q -E -- "(--init|--model|--init_model|--init_trunk) $p([= ]|\$)"; then why="${why:+$why,}init_or_model"; fi
    { [ -e "$d/eval.json" ] || compgen -G "$d/eval_*.json" > /dev/null; } || why="${why:+$why,}no_eval_output"   # either alone counts
    if [ -z "$why" ]; then echo "NOFETCH $p"; else echo "NOFETCH_REFUSED $p $why"; fi
  done
}
