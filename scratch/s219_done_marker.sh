# §219 done markers (Note 219-3 item 1; sourced by scratch/drain_gen_partial_s219.sh locally AND sent verbatim to the host, so one definition decides
# both the host status and the local post-fetch verification). A §219 run never writes eval*.json (SEAL-BREACH, TS_v1 §5.2); its markers are:
#   train_done.json     a finished textself_ft cell (written after model.pt)
#   mem_check_self.json G-T9 (--mem_check_self)
#   timing.json         G-T10 (--time_steps)
# s219_mark <run dir> -> RUN (a marker present) | BADM (no marker, model.pt present) | BAD (neither)
s219_mark() {
  if [ -s "$1/train_done.json" ] || [ -s "$1/mem_check_self.json" ] || [ -s "$1/timing.json" ]; then echo RUN
  elif [ -s "$1/model.pt" ]; then echo BADM
  else echo BAD; fi
}
# s219_need <run dir> -> the files a fetched RUN must hold locally (model.pt checked separately: it comes in pass 2)
s219_need() {
  if [ -s "$1/train_done.json" ]; then echo "log.jsonl args.json draws_log.json init_from.json train_done.json"
  elif [ -s "$1/mem_check_self.json" ]; then echo "args.json mem_check_self.json"
  else echo "args.json timing.json"; fi
}
# s219_check_fetched <run dir> -> OK, or MISSING <files> (every s219_need file present and non-empty, and no eval*.json at all)
s219_check_fetched() {
  local m="" f
  for f in $(s219_need "$1"); do [ -s "$1/$f" ] || m="$m $f"; done
  for f in "$1"/eval*.json; do [ -e "$f" ] && m="$m $(basename "$f")(eval-file)"; done
  [ -z "$m" ] && echo OK || echo "MISSING$m"
}
# s219_seal_rename <run dir> -> renames any eval*.json to *.void_unsealed (kept and disclosed, TS_v1 §5.2) and prints SEAL_BREACH <file> per file
s219_seal_rename() {
  local f
  for f in "$1"/eval*.json; do [ -e "$f" ] && mv "$f" "$f.void_unsealed" && echo "SEAL_BREACH $f"; done
  return 0
}
