"""Audit G-D condition 3 (10-21): after re-running langvb_eval --parts contamination on DIRECT, eval_direct.json minus {contamination, contamination_note,
part_sec.contamination, card_by_part.contamination, part_peak_gib.contamination} must equal the pre-run copy exactly (no panel / direct value moves).
    python -I scratch/s218_direct_contam_check.py <before.json> <after.json>   -> prints SAME / DIFF and the differing keys (names only; no values)"""
import copy
import json
import sys

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))


def strip(d):
    d = copy.deepcopy(d)
    for k in ("contamination", "contamination_note"):
        d.pop(k, None)
    for k in ("part_sec", "card_by_part", "part_peak_gib"):
        if isinstance(d.get(k), dict):
            d[k].pop("contamination", None)
    return d


a, b = strip(A), strip(B)
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("SAME" if not diff else "DIFF", "differing keys:", diff, "| contamination key present after:", "contamination" in B)
sys.exit(0 if not diff and "contamination" in B else 1)
