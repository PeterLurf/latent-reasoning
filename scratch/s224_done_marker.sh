# §224 done markers (FH_v1 §6 Stage 1; sourced by scratch/drain_gen_partial_s224.sh locally AND sent verbatim to the host). A §224 cloud run dir is one of:
#   <run>        a trained boxes.vbank cell: eval.json + model.pt, then boxes.vbank_eval's eval_vb.json (acc, stall) and boxes.vbank_diag's eval_khorizon.json
#   <run>_re     an eval-only dir (s224_ftctrl_s1_re: the shipped s215_ftctrl_s1 model): eval_vb.json + eval_khorizon.json (model.pt is not fetched back)
#   <run>_gtime  G-TIME (--time_steps): timing.json;   <run>_mem  K7 (--mem_check): mem_check.json
#   <run>_eqA / <run>_eqB  G-FH-EQ (--stop_at 1200): log.jsonl holding the step-1200 record
# s224_mark <run dir> -> RUN (a marker present) | BADM (no marker, a resume checkpoint ckpt_last.pt present) | BAD (neither)
s224_mark() {
  case "$1" in *_re) [ -s "$1/eval_khorizon.json" ] && [ -s "$1/eval_vb.json" ] && echo RUN || echo BAD; return ;; esac   # the shipped model.pt is no marker
  if [ -s "$1/eval_khorizon.json" ] || [ -s "$1/model.pt" ] || [ -s "$1/timing.json" ] || [ -s "$1/mem_check.json" ]; then echo RUN
  elif [ -s "$1/log.jsonl" ] && grep -q '"step": 1200,' "$1/log.jsonl"; then echo RUN
  elif [ -s "$1/ckpt_last.pt" ]; then echo BADM
  else echo BAD; fi
}
# s224_need <run dir> -> the files a fetched RUN must hold locally (a trained cell whose evals failed still lands: its evals re-run later)
s224_need() {
  case "$1" in
    *_re) echo "eval_vb.json eval_khorizon.json" ;;
    *_gtime) echo "timing.json args.json" ;;
    *_mem) echo "mem_check.json" ;;   # §228 (10-23): inject_ow --mem_check exits before args.json (b341 FETCH_INCOMPLETE)
    *_eqA|*_eqB) echo "log.jsonl args.json" ;;
    *) echo "eval.json model.pt log.jsonl args.json" ;;
  esac
}
# s224_check_fetched <run dir> -> OK, or MISSING <files>
s224_check_fetched() {
  local m="" f
  for f in $(s224_need "$1"); do [ -s "$1/$f" ] || m="$m $f"; done
  [ -z "$m" ] && echo OK || echo "MISSING$m"
}
# s224_resume_ckpt <run dir> -> the full-state checkpoint a --resume continues from (ckpt_last.pt), else nothing
s224_resume_ckpt() {
  [ -s "$1/ckpt_last.pt" ] && echo "$1/ckpt_last.pt"
}
