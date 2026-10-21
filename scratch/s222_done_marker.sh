# §222 done markers (Note 222-1; sourced by scratch/drain_gen_partial_s222.sh locally AND sent verbatim to the host, so one definition decides the host
# status and the local post-fetch check). A §222 cloud run dir (boxes/runs_fsq/s222_*) holds one of:
#   cert_<model>.g<i>.json / eval_lvb_s222.g<i>.json   a finished group (s222_cert.py / s222_a1.py; written once, at the group's end)
#   eval_lvb_s222_pre.json                              A1's K2 guard + DIAG-ALIGN (s222_a1.py --parts k2guard,diag)
#   timing.json                                         G-TIME (--time_steps)
#   eval_lvb.json                                       the ACC cross-card check (Note 222-1 item 6: langvb_eval --parts acc on a COPY of A1's run dir)
#   envelope.json                                       Note 222-3 (b): the registered-kernel same-state envelope + resume pair (b320)
# s222_mark <run dir> -> RUN (a marker present) | BADM (no marker, a resume checkpoint ckpt_last.pt present) | BAD (neither)
s222_mark() {
  local f
  for f in "$1"/cert_*.g*.json "$1"/eval_lvb_s222.g*.json "$1"/eval_lvb_s222_pre.json "$1"/timing.json "$1"/eval_lvb.json "$1"/envelope.json; do [ -s "$f" ] && { echo RUN; return; }; done
  if [ -s "$1/ckpt_last.pt" ]; then echo BADM; else echo BAD; fi
}
# s222_need <run dir> -> the files a fetched RUN must hold locally
s222_need() {
  local f
  for f in "$1"/cert_*.g*.json "$1"/eval_lvb_s222.g*.json; do [ -s "$f" ] && { echo "$(basename "$f") log.jsonl ckpt_fingerprint.json"; return; }; done
  [ -s "$1/eval_lvb_s222_pre.json" ] && { echo "eval_lvb_s222_pre.json"; return; }
  [ -s "$1/timing.json" ] && { echo "timing.json"; return; }
  [ -s "$1/envelope.json" ] && { echo "envelope.json"; return; }
  echo "eval_lvb.json"
}
# s222_check_fetched <run dir> -> OK, or MISSING <files>
s222_check_fetched() {
  local m="" f
  for f in $(s222_need "$1"); do [ -s "$1/$f" ] || m="$m $f"; done
  [ -z "$m" ] && echo OK || echo "MISSING$m"
}
# s222_resume_ckpt <run dir> -> the full-state checkpoint a --resume continues from (ckpt_last.pt), else nothing
s222_resume_ckpt() {
  [ -s "$1/ckpt_last.pt" ] && echo "$1/ckpt_last.pt"
}
