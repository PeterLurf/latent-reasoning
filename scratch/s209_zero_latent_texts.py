"""§209 descriptive check (eval-only, $0): decoded texts / lengths of the 0-latent and 6-latent arms (codi_probe recipe) on the 1,319 test items.
-> boxes/runs_fsq/gsm_codi_4ep_s1/eval_latent_texts.json"""
import json, os, sys
sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch
import boxes.codi_port as CP
dev = "cuda:1"; m, prj = CP.build(dev); TP = dict(CP.trainable(m, prj)); sd = torch.load("boxes/runs_fsq/gsm_codi_4ep_s1/trainable.pt", map_location=dev, weights_only=False)
for n, p_ in TP.items():
    p_.data.copy_(sd[n])
m.eval(); prj.eval(); data = CP.cp_mod().gsm8k(10_000); out = {}
for nl in (0, 6):
    acc, by, rows = CP.evaluate(m, prj, data, dev, nl)
    cot = [("<<" in r["txt"]) or ("The answer is" in r["txt"] and r["ntok"] > 12) for r in rows]
    out[str(nl)] = {"acc": acc, "mean_ntok": sum(r["ntok"] for r in rows) / len(rows), "share_txt_with_<<": sum("<<" in r["txt"] for r in rows) / len(rows),
                    "examples": [{"txt": r["txt"], "ntok": r["ntok"], "ok": bool(r["ok"])} for r in rows[:12]]}
    print(nl, acc, out[str(nl)]["mean_ntok"], out[str(nl)]["share_txt_with_<<"], flush=True)
json.dump(out, open("boxes/runs_fsq/gsm_codi_4ep_s1/eval_latent_texts.json", "w", newline="\n"), indent=1)
