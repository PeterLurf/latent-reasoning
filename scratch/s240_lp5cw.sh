#!/usr/bin/env bash
# §240 extra parts for score_df's §238 path (Theory request 10-28): lp5 + cue_wrong on lh_fh_s1 / lh_fh_s2; $0, local 4060 = cuda:0, fp32, eval_bs 250
# (the host parts' bs; the host card was an RTX 3090 - each part file records eval_gpu). The host eval.json summary is kept (run() would rewrite its
# meta with the local card) and only gains the two parts' entries. Writes eval_lp5.json / eval_cue_wrong.json. Unread.
#   nohup bash scratch/s240_lp5cw.sh >> logs/s240_land.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in lh_fh_s1 lh_fh_s2; do
  d=boxes/runs_fsq/$r; cp $d/eval.json scratch/_tmp/s240_evaljson_$r.json
  echo "$(date '+%F %T') start lp5,cue_wrong $r"; python -B -m boxes.inject_aw_eval --run $d --parts lp5,cue_wrong --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc lp5,cue_wrong $r"; [ $rc = 0 ] || exit 3
  python -B - "$d" "scratch/_tmp/s240_evaljson_$r.json" <<'EOF' || exit 4
import json, os, sys
d, keep = sys.argv[1], sys.argv[2]; s = json.load(open(keep))
for p_ in ("lp5", "cue_wrong"):
    s["parts"][p_] = {"file": "eval_%s.json" % p_, "sec": json.load(open(os.path.join(d, "eval_%s.json" % p_))).get("sec"), "local": "4060 fp32 bs 250 (s240_lp5cw.sh)"}
json.dump(s, open(os.path.join(d, "eval.json"), "w", newline="\n"), indent=1); print("eval.json kept + lp5 / cue_wrong entries", d)
EOF
  echo "$(date '+%F %T') LP5CW DONE $r"
done
