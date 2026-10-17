"""§210 (Theory / Audit 10-17, $0): all 1,319 decoded texts of the 0-latent arm (the §209 / s210_zero_latent_texts recipe), per item {id, text, ntok, ok, has_<<},
so the "<<" share can be recomputed from raw; asserts the share and acc equal eval_latent_texts.json's. -> <run>/eval_latent_texts_full.json
    python scratch/s210_latent_texts_full.py gsm_codi_20ep_s1 cuda:1"""
import json, os, sys
sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch
import boxes.codi_port as CP
RUN = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"; m, prj = CP.build(dev); TP = dict(CP.trainable(m, prj))
sd = torch.load("boxes/runs_fsq/%s/trainable.pt" % RUN, map_location=dev, weights_only=False)
for n, p_ in TP.items():
    p_.data.copy_(sd[n])
m.eval(); prj.eval(); data = CP.cp_mod().gsm8k(10_000)
acc, by, rows = CP.evaluate(m, prj, data, dev, 0)
items = [{"id": i, "text": r["txt"], "ntok": r["ntok"], "ok": bool(r["ok"]), "has_<<": "<<" in r["txt"]} for i, r in enumerate(rows)]
share = sum(it["has_<<"] for it in items) / len(items); ref = json.load(open("boxes/runs_fsq/%s/eval_latent_texts.json" % RUN))["0"]
out = {"n_latent": 0, "acc": acc, "share_txt_with_<<": share, "n": len(items), "matches_eval_latent_texts": {"acc": acc == ref["acc"], "share": share == ref["share_txt_with_<<"]}, "items": items}
json.dump(out, open("boxes/runs_fsq/%s/eval_latent_texts_full.json" % RUN, "w", newline="\n"), indent=1)
print("FULL", len(items), acc, share, out["matches_eval_latent_texts"], flush=True)
