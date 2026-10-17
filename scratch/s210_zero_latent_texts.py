"""§210 (the same check as §209, run dir as argv[1], device argv[2]; eval-only, $0): decoded texts / lengths of the 0-latent and 6-latent arms (codi_probe recipe) on the 1,319 test items.
-> <run>/eval_latent_texts.json    python scratch/s210_zero_latent_texts.py gsm_codi_20ep_s1 cuda:1"""
import json, os, sys
sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch
import boxes.codi_port as CP
RUN = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"; m, prj = CP.build(dev); TP = dict(CP.trainable(m, prj)); sd = torch.load("boxes/runs_fsq/%s/trainable.pt" % RUN, map_location=dev, weights_only=False)
for n, p_ in TP.items():
    p_.data.copy_(sd[n])
m.eval(); prj.eval(); data = CP.cp_mod().gsm8k(10_000); out = {}
for nl in (0, 6):
    acc, by, rows = CP.evaluate(m, prj, data, dev, nl)
    cot = [("<<" in r["txt"]) or ("The answer is" in r["txt"] and r["ntok"] > 12) for r in rows]
    out[str(nl)] = {"acc": acc, "mean_ntok": sum(r["ntok"] for r in rows) / len(rows), "share_txt_with_<<": sum("<<" in r["txt"] for r in rows) / len(rows),
                    "examples": [{"txt": r["txt"], "ntok": r["ntok"], "ok": bool(r["ok"])} for r in rows[:12]]}
    print(nl, acc, out[str(nl)]["mean_ntok"], out[str(nl)]["share_txt_with_<<"], flush=True)
json.dump(out, open("boxes/runs_fsq/%s/eval_latent_texts.json" % RUN, "w", newline="\n"), indent=1)
