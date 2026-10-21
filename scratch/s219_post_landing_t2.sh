#!/usr/bin/env bash
# §219 T2 post-landing chain: copy of scratch/s219_post_landing_t1.sh (G-D'd) with T2's pins ONLY: RUN s219_t2_s1, TAG gen_b319, init sha = T1's
# model.pt 65b020b0 (Note 219-12; was V6), landing JSON scratch/gate_s219/t2_landing_check.json, logs s219_t2_post_*.log, stop file
# scratch/s219_post_landing_t2.stop, eval_textself --round 2, G-T19 also requires T1's eval_text.json; mirror 319 (KEEP_MIRROR=1 at launch: kept until
# score_s219, Audit). n_self_steps 5992 stays a descriptive check (the coin stream rng_s is the latent's, shared by both rounds; TS_v1 P4).
#   KEEP_MIRROR=1 nohup bash scratch/s219_post_landing_t2.sh >> logs/s219_post_landing.out 2>&1 &
# (T1 header follows.)
# §219 T1 post-landing chain (TS_v1 at 146a25d0 §2.9 step 4 (+ step 5's "GPU parts in the gaps"), §2.7, §8 retention; G-T6 LOAD, G-T19 BASE FIRST, G-T20 SEAL,
# G-T11 stage boundary; Addendum 2 item 3: once started the 3070 stage runs uninterrupted, no co-tenant). Arm it at T1 launch, beside the §219 drain:
#   nohup bash scratch/s219_post_landing_t1.sh >> logs/s219_post_landing.out 2>&1 &
# Steps (each done-line in logs/s219_post_landing.out; rc captured BEFORE any $(date)):
#   0 wait (10-min polls; stop file scratch/s219_post_landing.stop) for the §219 drain's verified line for gen_b318 in logs/auto_drain.log AND the fetched
#     run dir (train_done.json + model.pt); a drained cell without train_done.json -> LANDING_FAILED, exit 3 (the .void_partial / resume path is a person's);
#   1 landing check (CPU): train_done steps 12000, self_reuse 0, model_sha256 == sha256(model.pt), log.jsonl last step 12000, draws_log 1..12000, init_from
#     sha == V6, args.json present; G-T20 part: no eval*.json / *.void_unsealed in the cell dir; G-T6: strict load + one forward on synthetic ids (rng 21903,
#     never a test item) -> scratch/gate_s219/t1_landing_check.json; any failure -> exit 3;
#   2 retention (TS_v1 §8, "deleted at landing"): stop the 318 ckpt_mirror (exact pid match) and delete scratch/ckpt_mirror/s219_t1_s1 (KEEP_MIRROR=1 skips; Audit 10-21 (d): launch with
#     KEEP_MIRROR=1 - the mirror is kept until score_s219 has run on T1, pending Theory's one-line Note);
#   3 G-T19: s219_base_s1/eval_text.json exists and theory/score_s219.py carries BASE_NOTE (Note 219-5) -> else exit 3 (no T1 output is produced);
#   4 wait until no python process holds the RTX 3070 (nvidia-smi), then T1 eval_text: python -m boxes.text_eval --run <cell> --part text --cot_eval_bs 32
#     --device cuda:1 (the 3070, as scratch/s219_3070_base.sh; LATENT_MEM_CAP_GB 6.0); the meta card must read RTX 3070 / fp32 / tf32 false / bs 32;
#     STAGE_BOUNDARY line: commit eval_text.json (G-T11);
#   5 eval_textself CPU parts (LABEL-AUDIT-T, CONTAMINATED; registered; the round-2 condition reads them) on CPU;
#   6 eval_textself GPU parts (MEMO-GAP-T, LABELLER-KEPT-T; descriptive; after eval_text, V3) on the 3070 (default HELD, Audit 10-21: RUN_GPU_PARTS=1 runs them; Theory's step-4 note first).
# GATE-T2 / GATE-T1' / the T2 pool are NOT started here: they wait for Theory's step-4 note (T1 readings, Delta_1, the round-2 condition; G-T20).
cd /d/Latent || exit 2
RUN=boxes/runs_fsq/s219_t2_s1; TAG=gen_b319; OUT=logs/s219_post_landing.out; STOP=scratch/s219_post_landing_t2.stop; DEV=${DEV:-cuda:1}
V6=65b020b053008f94a9f494d1bf85c5264e5b1a18e800e83cbf357fb278901bf6   # T2 init = T1 model.pt (variable name kept)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 LATENT_MEM_CAP_GB=6.0
say() { echo "$(date '+%F %T') s219_post_t2 $*" >> "$OUT"; }
run() { local tag=$1; shift; "$@" > "logs/s219_t2_post_$tag.log" 2>&1; local rc=$?; say "$tag rc $rc"; return $rc; }
gpu3070_busy() {   # -> the python compute apps on the RTX 3070 (empty = free)
  local u; u=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | awk -F', ' '/3070/ {print $2}' | head -1)
  [ -z "$u" ] && { echo "NO_3070_VISIBLE"; return; }
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null | grep "$u" | grep -i python
}
wait3070() { local w=0 b; while :; do [ -f "$STOP" ] && { say "stopped ($STOP) while waiting for the 3070"; exit 0; }
    b=$(gpu3070_busy); [ -z "$b" ] && return 0
    [ "$b" = NO_3070_VISIBLE ] && { say "ALERT no RTX 3070 visible to nvidia-smi - held"; exit 3; }
    [ $w = 0 ] && say "waiting for the 3070 (busy: $(echo "$b" | tr '\n' ';' | cut -c1-200))"; w=1; sleep 300; done; }
landing_check() {   # <run dir> <out json>: step 1 (train_done / sha / log / draws_log / init_from / seal / G-T6) -> rc 0 iff PASS
  python -B -c "
import glob, hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, 'D:/Latent'); torch.set_num_threads(2)
import boxes.textself as TS
R, OUTJ = sys.argv[1], sys.argv[2]; o = {'run': R}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
td = json.load(open(R + '/train_done.json')); ms = sha(R + '/model.pt')
recs = [json.loads(l) for l in open(R + '/log.jsonl') if l.strip()]; last = max((r['step'] for r in recs if 'sec' in r), default=None)
dl = json.load(open(R + '/draws_log.json')); ini = json.load(open(R + '/init_from.json'))
o['train_done'] = {k: td.get(k) for k in ('steps', 'train_sec', 'n_nonfinite', 'n_self_steps', 'pool_cursor', 'self_reuse', 'model_sha256')}
o['checks'] = {'steps_12000': td.get('steps') == 12000, 'self_reuse_0': td.get('self_reuse') == 0, 'model_sha_matches': td.get('model_sha256') == ms,
               'log_last_step_12000': last == 12000, 'draws_log_1_12000': [d['step'] for d in dl] == list(range(1, 12001)), 'init_from_T1': ini.get('sha256') == '$V6',
               'args_json': os.path.exists(R + '/args.json'),
               'no_eval_files (G-T20)': not [f for f in os.listdir(R) if f.startswith('eval') or f.endswith('.void_unsealed')],
               'no_ckpt_weights_home (Audit 10-21, retention)': not [f for f in os.listdir(R) if f.startswith('ckpt_') and f.endswith('.pt')]}
o['model_sha256'] = ms; o['n_self_steps_expected_5992'] = td.get('n_self_steps') == 5992
g, s2 = TS.load_model(R + '/model.pt', 'cpu'); x = torch.tensor(np.random.default_rng(21903).integers(0, 50257, (2, 40)))
with torch.no_grad():
    h, _ = g(g.embed(x), None); lg = g.logits(h)
o['G_T6'] = {'strict_load': True, 'sha256': s2, 'forward_shape': list(lg.shape), 'finite': bool(torch.isfinite(lg).all()), 'input': 'random ids rng 21903, never a test item'}
o['checks']['G_T6'] = o['G_T6']['finite'] and s2 == ms
o['PASS'] = all(o['checks'].values())
json.dump(o, open(OUTJ, 'w', newline='\n'), indent=1); print(json.dumps(o)); sys.exit(0 if o['PASS'] else 1)
" "$1" "$2"; }
evalmeta() {   # <eval_text.json> -> "OK|BAD card dtype tf32 bs ..." (T5: RTX 3070, fp32, tf32 off, cot_eval_bs 32)
  python -c "import json, sys
try: m = json.load(open(sys.argv[1]))['meta']
except Exception as e: print('BAD unreadable', e); sys.exit()
ok = '3070' in str(m.get('card')) and m.get('dtype') == 'fp32' and m.get('tf32') is False and m.get('cot_eval_bs') == 32
print(('OK' if ok else 'BAD'), m.get('card'), m.get('dtype'), 'tf32', m.get('tf32'), 'bs', m.get('cot_eval_bs'), 'sec', m.get('sec'), 'peak_gib', m.get('peak_gib'))" "$1"; }
[ "${S219_POST_LIB:-0}" = 1 ] && return 0

say "armed (pid $$): waiting for the $TAG drain + $RUN/train_done.json + model.pt"
alerted=0
while :; do
  [ -f "$STOP" ] && { say "stopped ($STOP)"; exit 0; }
  if grep -q " $TAG " logs/auto_drain.log 2>/dev/null; then
    if [ -s "$RUN/train_done.json" ] && [ -s "$RUN/model.pt" ]; then say "LANDED: $(grep " $TAG " logs/auto_drain.log | tail -1 | cut -c1-200)"; break; fi
    say "LANDING_FAILED: $TAG drained but $RUN has no train_done.json / model.pt ($(ls -d $RUN* 2>/dev/null | tr '\n' ' ')) - a person decides (resume from the mirror / .void_partial)"; exit 3
  fi
  if [ $alerted = 0 ] && grep -q "DRAIN FAILED\|STALLED\|UNREACHABLE 3 h" "logs/auto_drain_$TAG.out" 2>/dev/null; then
    say "ALERT the $TAG watcher reported: $(grep "DRAIN FAILED\|STALLED\|UNREACHABLE 3 h" "logs/auto_drain_$TAG.out" | tail -1) - still waiting for a verified drain"; alerted=1
  fi
  sleep 600
done

run landing landing_check "$RUN" scratch/gate_s219/t2_landing_check.json; rc=$?
[ $rc -eq 0 ] || { say "ALERT landing check FAILED (scratch/gate_s219/t2_landing_check.json, logs/s219_t2_post_landing.log) - held"; exit 3; }
say "landing check PASS: $(python -c "import json; d = json.load(open('scratch/gate_s219/t2_landing_check.json')); print('model_sha256', d['model_sha256'][:16], 'self steps', d['train_done']['n_self_steps'], 'train_sec', d['train_done']['train_sec'], 'nonfinite', d['train_done']['n_nonfinite'], '5992:', d['n_self_steps_expected_5992'])")"

if [ "${KEEP_MIRROR:-0}" != 1 ] && [ -d scratch/ckpt_mirror/s219_t2_s1 ]; then   # TS_v1 §8: checkpoints and mirrors are deleted at landing (model.pt verified above)
  for mp in $(ps -ef | awk '$7 ~ /ckpt_mirror\.py$/ && $8 == "319:s219_t2_s1" {print $2}'); do kill "$mp" && say "stopped ckpt_mirror 319 (pid $mp)"; done
  sleep 5; rm -rf scratch/ckpt_mirror/s219_t2_s1; rc=$?; say "RETENTION mirror scratch/ckpt_mirror/s219_t2_s1 deleted rc $rc"
fi

[ -s boxes/runs_fsq/s219_base_s1/eval_text.json ] || { say "ALERT G-T19: s219_base_s1/eval_text.json missing - no T2 output is produced; held"; exit 3; }
[ -s boxes/runs_fsq/s219_t1_s1/eval_text.json ] || { say "ALERT T1 eval_text.json missing - no T2 output is produced; held"; exit 3; }
grep -q '^BASE_NOTE = {' theory/score_s219.py || { say "ALERT G-T19: theory/score_s219.py has no BASE_NOTE (Theory's base note) - no T2 output is produced; held"; exit 3; }
say "G-T19 ok (base eval_text.json + BASE_NOTE $(grep -o '^BASE_NOTE = {[^}]*}' theory/score_s219.py))"
[ -e "$RUN/eval_text.json" ] && { say "ALERT $RUN/eval_text.json already exists - held (rename it *.void_* first)"; exit 3; }

wait3070
run eval_text python -B -m boxes.text_eval --run "$RUN" --part text --cot_eval_bs 32 --device "$DEV"; rc=$?
[ $rc -eq 0 ] && [ -s "$RUN/eval_text.json" ] || { say "ALERT T2 eval_text failed (rc $rc; logs/s219_t2_post_eval_text.log; any .partial.json kept) - held"; exit 3; }
mc=$(evalmeta "$RUN/eval_text.json")
say "T2 eval_text meta: $mc"; echo "$mc" | grep -q "^OK " || { say "ALERT eval_text meta is not RTX 3070 / fp32 / tf32 off / bs 32 (T5: VOID) - held"; exit 3; }
say "STAGE_BOUNDARY (G-T11): commit $RUN/eval_text.json now (off the working card)"

run textself_cpu python -B -m boxes.text_eval --run "$RUN" --part textself --round 2 --parts cpu --device cpu; rc=$?
[ $rc -eq 0 ] || { say "ALERT eval_textself CPU parts failed (rc $rc; logs/s219_t2_post_textself_cpu.log) - held"; exit 3; }
say "eval_textself CPU parts done ($RUN/eval_textself.json: label_audit, contamination); Theory's T2 readings note is next; STAGE_BOUNDARY: commit eval_textself.json"

if [ "${RUN_GPU_PARTS:-0}" = 1 ]; then   # Audit 10-21 (c): default HOLD until Theory's step-4 note (or Theory confirms in writing it changes nothing these parts read)
  wait3070
  run textself_gpu python -B -m boxes.text_eval --run "$RUN" --part textself --round 2 --parts gpu --device "$DEV"; rc=$?
  [ $rc -eq 0 ] || { say "ALERT eval_textself GPU parts failed (rc $rc; logs/s219_t2_post_textself_gpu.log; descriptive) - held"; exit 3; }
  say "eval_textself GPU parts done (memo, labeller_kept; descriptive); STAGE_BOUNDARY: commit eval_textself.json"
fi
say "T2 post-landing chain DONE"
