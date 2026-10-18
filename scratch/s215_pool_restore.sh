#!/usr/bin/env bash
# Restore boxes/runs_fsq/s215_pool/pool_r1.jsonl (deleted 12:4x 10-18 while R1-s2's selfdepth still read it for pool_excl_hits): after R1-s2's remaining parts
# ("§216 R1-s2 remaining parts done"; one 3070 job at a time) rebuild it with the unchanged builder (scratch/s215_build_pool.py, same seeds, D3-T labeller, 3070
# fp32 bs 32) into *_rebuild dirs, require sha256 == the recorded pool_meta sha (c2e226ae...), move the file back, re-run selfdepth --lineage s2 --round 1 on
# R1-s2, then delete the restored pool again (nothing else reads it).
#   nohup bash scratch/s215_pool_restore.sh >> logs/s215_pool_restore.out 2>&1 &
cd "$(dirname "$0")/.."
say() { echo "$(date '+%F %T') $*"; }
until grep -q "§216 R1-s2 remaining parts done" logs/s216_s2_chain.out 2>/dev/null; do sleep 120; done
say "rebuilding s215 pool_r1 on cuda:1"
python scratch/s215_build_pool.py --device cuda:1 --out_suffix _rebuild > logs/s215_pool_rebuild.log 2>&1; rc=$?; say "rebuild rc $rc $(tail -n 1 logs/s215_pool_rebuild.log | cut -c1-120)"
want=$(python -c "import json; print(json.load(open('boxes/runs_fsq/s215_pool/pool_meta.json'))['sha256'])"); got=$(python -c "import sys; sys.path.insert(0,'D:/Latent'); import boxes.selfdepth as SD; print(SD.file_sha256('boxes/runs_fsq/s215_pool_rebuild/pool_r1.jsonl'))")
if [ "$want" != "$got" ]; then say "SHA MISMATCH want $want got $got - NOT restored; tell Theory / Audit"; exit 1; fi
mv boxes/runs_fsq/s215_pool_rebuild/pool_r1.jsonl boxes/runs_fsq/s215_pool/pool_r1.jsonl && say "restored, sha $got"
python -m boxes.vbank_diag --run boxes/runs_fsq/s216_r1_s2 --part selfdepth --lineage s2 --round 1 --device cuda:1 --eval_bs 32 > logs/s216_selfdepth_s216_r1_s2.log 2>&1; rc=$?
say "b314 s216_r1_s2 selfdepth rc $rc $(tail -n 1 logs/s216_selfdepth_s216_r1_s2.log | cut -c1-140)"
[ $rc = 0 ] && [ -s boxes/runs_fsq/s216_r1_s2/eval_selfdepth.json ] && rm -f boxes/runs_fsq/s215_pool/pool_r1.jsonl && say "s215 pool_r1 deleted again"
say "§215 pool restore + R1-s2 selfdepth done"
