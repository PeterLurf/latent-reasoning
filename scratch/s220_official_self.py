"""Note 220-3 + Audit addendum: official-vs-official self-consistency of the official-semantics (virtual bs 128) arm. Reads the model's eval_gcs.json
official per-item parsed predictions (per_item.official.pred, item order = golds key order; the gate input only - no swap arm is read) and the two shuffled
runs scratch/s220/[zen_codi_gpt2/]official_shuffle_{22030,22031}.json -> gate_official_self.json beside eval_gcs.json:
  official_self = [A1, A2, A3]: A1 = agree(off, off_22030), A2 = agree(off, off_22031), A3 = agree(off_22030, off_22031); agree = the share of items with
  equal parsed predictions (None == None agrees), gate.agree's definition (U10).
    python -B scratch/s220_official_self.py --model codi20|zen"""
import argparse
import json
import os

ROOT = "D:/Latent"
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["codi20", "zen"]); a = ap.parse_args()
run = os.path.join(ROOT, "boxes/runs_fsq/gsm_codi_20ep_s1") if a.model == "codi20" else os.path.join(ROOT, "scratch/s220/zen_codi_gpt2")
sd = os.path.join(ROOT, "scratch/s220") if a.model == "codi20" else os.path.join(ROOT, "scratch/s220/zen_codi_gpt2")
E = json.load(open(os.path.join(run, "eval_gcs.json"))); ids = list(E["golds"].keys()); off = dict(zip(ids, E["per_item"]["official"]["pred"]))
S = {sd_: json.load(open(os.path.join(sd, "official_shuffle_%d.json" % sd_)))["pred"] for sd_ in (22030, 22031)}
assert all(set(v) == set(ids) for v in S.values()), "item sets differ"
ag = lambda x, y: sum(x[i] == y[i] for i in ids) / len(ids)  # noqa: E731
res = {"official_self": [ag(off, S[22030]), ag(off, S[22031]), ag(S[22030], S[22031])], "n": len(ids),
       "def": "A1 = agree(off, off_22030), A2 = agree(off, off_22031), A3 = agree(off_22030, off_22031); agree = equal parsed predictions (None == None agrees), U10",
       "sources": {"eval_gcs": os.path.relpath(os.path.join(run, "eval_gcs.json"), ROOT), "shuffles": [os.path.relpath(os.path.join(sd, "official_shuffle_%d.json" % s_), ROOT) for s_ in (22030, 22031)]}}
json.dump(res, open(os.path.join(run, "gate_official_self.json"), "w", newline="\n"), indent=1); print("OFFICIAL_SELF", res["official_self"])
