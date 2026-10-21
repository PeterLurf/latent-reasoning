# §223 done markers (PS2_v1 §8; sourced by scratch/drain_gen_partial_s223.sh locally AND sent verbatim to the host). A §223 cloud run dir holds:
#   train_done.json (+ trainable.pt)   a finished boxes.codi_ps cell (written after trainable.pt); the same dir also gets eval_ps.json (boxes.ps_eval)
#   timing.json                        G-TIME (--time_steps; with --auto_micro also mem_check.json)
# s223_mark <run dir> -> RUN (a marker present) | BADM (no marker, a resume checkpoint ckpt_last.pt present) | BAD (neither)
s223_mark() {
  if [ -s "$1/train_done.json" ] || [ -s "$1/timing.json" ]; then echo RUN
  elif [ -s "$1/ckpt_last.pt" ]; then echo BADM
  else echo BAD; fi
}
# s223_need <run dir> -> the files a fetched RUN must hold locally
s223_need() {
  if [ -s "$1/train_done.json" ]; then echo "train_done.json trainable.pt log.jsonl args.json eval_ps.json"
  else echo "timing.json args.json"; fi
}
# s223_check_fetched <run dir> -> OK, or MISSING <files>
s223_check_fetched() {
  local m="" f
  for f in $(s223_need "$1"); do [ -s "$1/$f" ] || m="$m $f"; done
  [ -z "$m" ] && echo OK || echo "MISSING$m"
}
# s223_resume_ckpt <run dir> -> the full-state checkpoint a --resume continues from (ckpt_last.pt), else nothing
s223_resume_ckpt() {
  [ -s "$1/ckpt_last.pt" ] && echo "$1/ckpt_last.pt"
}
