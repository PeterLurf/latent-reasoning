#!/usr/bin/env bash
# gen10 fixture: drain_gen_partial_s267.sh against a LOCAL fake host (ssh -> bash -c, scp -> cp, /workspace/Latent -> the fixture host root, vdestroy stubbed).
# Expect: runA (a TRAIN line) RUN, fetched in full; runA_x (an extract line, eval present) RUN with its model.pt NOFETCH'd = deleted on the "host", not fetched, need list
# satisfied; pil (stop_at; log step 1200 + ckpt) BADM -> pil.void_partial; Audit (a) / (b) refusals ALERTed and NOT deleted: runA/model.pt (train_model),
# x_init/model.pt (init_or_model: another line's --init), runC_x/model.pt (no_eval_output); DRAINED.   bash scratch/s267_build/fixture_gen10.sh
set -u
cd "$(dirname "$0")/../.." || exit 2
R=$(pwd)/scratch/_tmp/g10fx; rm -rf "$R"; mkdir -p "$R/host/logs" "$R/host/scratch" "$R/host/boxes/runs_fsq" "$R/local/scratch" "$R/local/logs" "$R/local/boxes/runs_fsq"
H="$R/host"; B=$H/boxes/runs_fsq
mkdir -p $B/runA $B/runA_x $B/pil $B/x_init $B/runC_x
for d in runA runA_x; do echo '{"acc": 1}' > $B/$d/eval.json; echo '{}' > $B/$d/args.json; echo '{"step": 100}' > $B/$d/log.jsonl; head -c 2048 /dev/urandom > $B/$d/model.pt; done
printf '{"step": 1200, "loss": 1}\n' > $B/pil/log.jsonl; echo '{}' > $B/pil/args.json; head -c 1024 /dev/urandom > $B/pil/ckpt_last.pt
head -c 512 /dev/urandom > $B/x_init/model.pt; head -c 512 /dev/urandom > $B/runC_x/model.pt
printf '%s\n' "00:00:01 start boxes/runs_fsq/runA" "00:00:02 evaluated boxes/runs_fsq/runA rc 0" "00:00:03 evaluated boxes/runs_fsq/runA_x rc 0" \
  "00:00:04 evaluated boxes/runs_fsq/pil rc 9 no-eval" "00:00:05 QUEUE_DONE" > $H/logs/queue_gen_b999.out
for d in runA runA_x pil; do echo "log $d" > $H/logs/gen_$d.log; done
printf '%s\n' "# THEORY fixture" "# NOFETCH boxes/runs_fsq/runA_x/model.pt" "# NOFETCH boxes/runs_fsq/runA/model.pt" "# NOFETCH boxes/runs_fsq/x_init/model.pt" \
  "# NOFETCH boxes/runs_fsq/runC_x/model.pt" "boxes/runs_fsq/runA scratch.s257_guarded --mode train --init boxes/runs_fsq/x_init/model.pt=abcd1234 -- --steps 10" \
  "boxes/runs_fsq/runA_x scratch.s258s2_ws6k --run boxes/runs_fsq/runA --step 6" > $H/scratch/gen_q_b999.txt
cp scratch/s267_done_marker.sh "$R/local/scratch/"; printf '#!/usr/bin/env bash\necho "DESTROYED_VERIFIED $1"\n' > "$R/local/scratch/vdestroy.sh"
sed -e "s|^S=\"ssh .*root@\$H\"|S=\"bash -c\"|" -e "s|/workspace/Latent|$H|g" \
    -e "s|scp -q -o ServerAliveInterval=15 -i ~/.ssh/runpod_ed25519 -P \$P root@\$H:/tmp/f_\$n.tgz|cp /tmp/f_\$n.tgz|" scratch/drain_gen_partial_s267.sh > "$R/local/scratch/drain_gen_partial_s267.sh"
grep -q 'S="bash -c"' "$R/local/scratch/drain_gen_partial_s267.sh" || { echo "FIXTURE: S not replaced"; exit 2; }
out=$(cd "$R/local" && bash scratch/drain_gen_partial_s267.sh fakehost 1 gen_b999 999 2>&1); echo "$out" | tail -6
L="$R/local/boxes/runs_fsq"; ok=1
[ -s $L/runA/model.pt ] && [ -s $L/runA/eval.json ] || { echo "FAIL runA not fetched in full"; ok=0; }
[ -s $L/runA_x/eval.json ] && [ ! -e $L/runA_x/model.pt ] || { echo "FAIL runA_x: eval missing or NOFETCH model.pt fetched"; ok=0; }
[ ! -e $B/runA_x/model.pt ] || { echo "FAIL NOFETCH model.pt not deleted on the host"; ok=0; }
[ -d $L/pil.void_partial ] && [ ! -d $L/pil ] || { echo "FAIL pil not BADM (void_partial)"; ok=0; }
echo "$out" | grep -q "^DRAINED 999" || { echo "FAIL no DRAINED"; ok=0; }
echo "$out" | grep -q "ALERT NOFETCH_REFUSED boxes/runs_fsq/runA/model.pt train_model" || { echo "FAIL (a) train model.pt not refused"; ok=0; }
echo "$out" | grep -q "ALERT NOFETCH_REFUSED boxes/runs_fsq/x_init/model.pt .*init_or_model" || { echo "FAIL (a) --init path not refused"; ok=0; }
echo "$out" | grep -q "ALERT NOFETCH_REFUSED boxes/runs_fsq/runC_x/model.pt .*no_eval_output" || { echo "FAIL (b) no-eval dir not refused"; ok=0; }
[ -e $B/runA/model.pt ] && [ -e $B/x_init/model.pt ] && [ -e $B/runC_x/model.pt ] || { echo "FAIL a refused path was deleted on the host"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE gen10 PASS" || echo "FIXTURE gen10 FAIL"
