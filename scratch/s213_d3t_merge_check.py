"""§213 D3-T eval_vb.json after the probe / vswap / stall re-run (Theory / Audit 10-16 ~21:0x): (1) every key of the pre-merge snapshot (acc part, run 1) is
unchanged in the merged file except part_sec (json-serialised, sorted keys); (2) a "provenance" block records run 1 (acc) and run 2 (probe / vswap / stall)
with log names and timestamps. Atomic rewrite with newline="\\n".   python scratch/s213_d3t_merge_check.py"""
import json
import os

R = "D:/Latent/boxes/runs_fsq/s213_d3t_s1/eval_vb.json"; PRE = "D:/Latent/scratch/_tmp/s213_d3t_eval_vb_pre_merge.json"
pre = json.load(open(PRE)); post = json.load(open(R))
ser = lambda v: json.dumps(v, sort_keys=True)  # noqa: E731
diff = [k for k in pre if k != "part_sec" and ser(pre[k]) != ser(post.get(k))]
added = sorted(set(post) - set(pre))
post["provenance"] = {
    "run1": {"parts": ["acc"], "log": "logs/s213_eval_vb_s213_d3t_s1.log", "finished": "2026-10-16 20:54 (acc dumped; probe then failed: CUBLAS_STATUS_ALLOC_FAILED)",
             "part_sec": pre.get("part_sec")},
    "run2": {"parts": ["probe", "vswap", "stall"], "log": "logs/s213_eval_vb2_s213_d3t_s1.log", "cmd": "vbank_eval --parts probe,vswap,stall (same card / fp32 / eval_bs 32)",
             "fix": "a57e450e: torch.cuda.empty_cache() between parts"},
    "merge_check": {"pre_merge_sha256_prefix": "005fc9372afce89c", "keys_changed_except_part_sec": diff, "keys_added": added}}
json.dump(post, open(R + ".tmp", "w", newline="\n")); os.replace(R + ".tmp", R)
print("MERGE_CHECK keys changed (excl. part_sec):", diff, "| added:", added)
