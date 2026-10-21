#!/usr/bin/env bash
# local unit test (no ssh) of scratch/s219_done_marker.sh and of drain_gen_partial_s219.sh's host status script (its REM string, evaluated from the
# drain file itself, run on a fake /workspace/Latent tree).  bash scratch/_tmp/s219_fix/test_drain_marker.sh
set -u
cd /d/Latent; . scratch/s219_done_marker.sh; F=scratch/_tmp/s219_fix/fakehost; rm -rf "$F"; mkdir -p "$F/logs"; cd "$F"; fails=0
ok() { if [ "$2" = "$3" ]; then echo "ok   $1: $2"; else echo "FAIL $1: got '$2' want '$3'"; fails=$((fails + 1)); fi; }
R=boxes/runs_fsq
mk() { mkdir -p "$R/$1"; shift_=$1; shift; for f in "$@"; do echo '{"x":1}' > "$R/$shift_/$f"; done; }
mk s219_t1_s1 model.pt log.jsonl args.json args_diff.json draws_log.json init_from.json train_done.json
mk s219_t1_s1_mem args.json args_diff.json init_from.json mem_check_self.json
mk s219_t1_s1_gtime args.json args_diff.json init_from.json log.jsonl timing.json
mk s219_t1_s1_crash model.pt log.jsonl args.json draws_log.json
mk s219_t1_s1_dead args.json log.jsonl
mk s219_t1_s1_breach model.pt log.jsonl args.json draws_log.json init_from.json train_done.json eval.json eval_2k.json
mk s219_t1_s1_noinit model.pt log.jsonl args.json draws_log.json train_done.json
: > "$R/s219_t1_s1_empty_marker/train_done.json" 2>/dev/null || { mkdir -p "$R/s219_t1_s1_empty_marker"; : > "$R/s219_t1_s1_empty_marker/train_done.json"; }
ok mark_cell "$(s219_mark $R/s219_t1_s1)" RUN; ok mark_mem "$(s219_mark $R/s219_t1_s1_mem)" RUN; ok mark_gtime "$(s219_mark $R/s219_t1_s1_gtime)" RUN
ok mark_crash "$(s219_mark $R/s219_t1_s1_crash)" BADM; ok mark_dead "$(s219_mark $R/s219_t1_s1_dead)" BAD; ok mark_empty_marker "$(s219_mark $R/s219_t1_s1_empty_marker)" BAD
ok check_cell "$(s219_check_fetched $R/s219_t1_s1)" OK; ok check_mem "$(s219_check_fetched $R/s219_t1_s1_mem)" OK; ok check_gtime "$(s219_check_fetched $R/s219_t1_s1_gtime)" OK
ok check_noinit "$(s219_check_fetched $R/s219_t1_s1_noinit)" "MISSING init_from.json"
ok check_breach "$(s219_check_fetched $R/s219_t1_s1_breach)" "MISSING eval.json(eval-file) eval_2k.json(eval-file)"
# the drain's own host status script, run on this fake tree
TAG=gen_b999; for j in s219_t1_s1 s219_t1_s1_mem s219_t1_s1_gtime s219_t1_s1_crash s219_t1_s1_dead s219_t1_s1_breach; do echo "12:00:00 evaluated $R/$j rc 0 no-eval" >> logs/queue_$TAG.out; done
echo "QUEUE_DONE" >> logs/queue_$TAG.out
pushd /d/Latent >/dev/null; eval "$(sed -n '/^REM=/,/END_OK"$/p' scratch/drain_gen_partial_s219.sh)"; popd >/dev/null
st=$(bash -c "${REM//\/workspace\/Latent/$PWD}" 2>&1)
echo "--- status script output:"; echo "$st"
ok status_end "$(echo "$st" | tail -n 1)" END_OK; ok status_qd "$(echo "$st" | sed -n 's/^QD=//p')" 1
ok status_runs "$(echo "$st" | sed -n 's/^RUN //p' | xargs)" "$R/s219_t1_s1 $R/s219_t1_s1_mem $R/s219_t1_s1_gtime $R/s219_t1_s1_breach"
ok status_bad "$(echo "$st" | sed -n 's/^BADM\{0,1\} //p' | xargs)" "$R/s219_t1_s1_crash $R/s219_t1_s1_dead"
ok status_seal "$(echo "$st" | sed -n 's/^SEAL_BREACH //p' | xargs)" "$R/s219_t1_s1_breach/eval.json $R/s219_t1_s1_breach/eval_2k.json"
ok breach_renamed "$(ls $R/s219_t1_s1_breach | grep -c void_unsealed) $(ls $R/s219_t1_s1_breach | grep -c '^eval.*json$')" "2 0"
ok breach_after_rename "$(s219_check_fetched $R/s219_t1_s1_breach)" OK
# a not-finished queue: no rename happens
TAG=gen_b998; mk s219_t1_s1_live train_done.json eval.json; echo "12:00:00 evaluated $R/s219_t1_s1_live rc 0" > logs/queue_$TAG.out
pushd /d/Latent >/dev/null; eval "$(sed -n '/^REM=/,/END_OK"$/p' scratch/drain_gen_partial_s219.sh)"; popd >/dev/null; st=$(bash -c "${REM//\/workspace\/Latent/$PWD}" 2>&1)
ok notdone_qd "$(echo "$st" | sed -n 's/^QD=//p')" 0; ok notdone_no_rename "$(ls $R/s219_t1_s1_live | grep -c void_unsealed)" 0
# the drain never names eval.json as a marker
ok drain_no_eval_marker "$(grep -v '^#' /d/Latent/scratch/drain_gen_partial_s219.sh | grep -c 'eval\.json')" 0
cd /d/Latent; rm -rf "$F"; echo "FAILS $fails"; [ $fails = 0 ]
