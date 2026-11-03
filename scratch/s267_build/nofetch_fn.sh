
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
