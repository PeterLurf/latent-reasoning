#!/usr/bin/env bash
# Audit CODE_REVIEW MED-2 fixture: scratch/s271_guarded.py --mode evalonly on a broken model file (the eval fails at load) with --long:
# expect rc != 0, log.jsonl {"status": "FAILED", "eval_rc" != 0, "long_skipped": true}, an explicit eval.json {"FAILED": true}, and a printed 'RUN_LONG SKIPPED'.
set -u
cd "$(dirname "$0")/../.." || exit 2
R=scratch/_tmp/gfailfx; rm -rf "$R"; mkdir -p "$R"
head -c 4096 /dev/urandom > "$R/broken.pt"; s=$(sha256sum "$R/broken.pt" | cut -c1-16)
out=$(python -B -m scratch.s271_guarded --mode evalonly --model "$R/broken.pt=$s" --args boxes/runs_fsq/bun_s1/args.json --parts acc --long "8:1:eval_long_x.json" \
      --device cpu --out "$R/run" 2>&1); rc=$?
ok=1
[ $rc != 0 ] || { echo "FAIL: rc 0"; ok=0; }
python -c "import json,sys; l=json.loads(open('$R/run/log.jsonl').read()); e=json.load(open('$R/run/eval.json')); sys.exit(0 if l['status'] == 'FAILED' and l['eval_rc'] != 0 and l['long_skipped'] and e.get('FAILED') else 1)" \
  || { echo "FAIL: markers"; cat "$R/run/log.jsonl" "$R/run/eval.json" 2>&1 | head -5; ok=0; }
echo "$out" | grep -q "RUN_LONG SKIPPED" || { echo "FAIL: no loud skip"; ok=0; }
[ -e "$R/run/eval_long_x.json" ] && { echo "FAIL: run_long ran"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE guarded_failed PASS (rc $rc)" || echo "FIXTURE guarded_failed FAIL"
