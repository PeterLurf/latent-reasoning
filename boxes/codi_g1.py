"""§210 G1' (registered de919de5): on the training host, before training, re-score the released zen-E/CODI-gpt2 through boxes/codi_port's eval path (the
checkpoint loaded into the port's LoRA structure -> merged fp32 -> theory/codi_probe.run, bs 16, CODI's GSM8K test json). PASS iff |acc - 0.425| <= 0.010.
A NEW module (imports codi_port read-only; scratch/gate_s209.py's g1 recipe). -> <out>/eval.json {"acc", "n", "pass", "ref", "tol", "card"}
    python -m boxes.codi_g1 --out boxes/runs_fsq/gsm_codi_20ep_s1_g1 [--device cuda:0]"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402

import boxes.codi_port as CP  # noqa: E402

REL = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "codi", "pytorch_model.bin"); REF, TOL = 0.425, 0.010
URL = "https://huggingface.co/zen-E/CODI-gpt2/resolve/main/pytorch_model.bin"; SHA = "fd223b14932b8b66605c2a688e7d7193058ff2b40b2c6f6c1547c227d06d8417"   # = our local copy (checked 10-15)


def fetch():
    """the released checkpoint: fetched on the host from the HF hub (b303's upload failed 3x with it in the bundle), sha256-verified against our copy."""
    import hashlib
    import urllib.request
    if not os.path.exists(REL):
        os.makedirs(os.path.dirname(REL), exist_ok=True); urllib.request.urlretrieve(URL, REL + ".part"); os.replace(REL + ".part", REL)
    h = hashlib.sha256()
    with open(REL, "rb") as f_:
        for b_ in iter(lambda: f_.read(1 << 24), b""):
            h.update(b_)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sha = fetch()
    if sha != SHA:
        json.dump({"pass": False, "error": "released ckpt sha256 mismatch", "sha256": sha}, open(os.path.join(a.out, "eval.json"), "w", newline=chr(10)), indent=1)
        print("G1PRIME sha mismatch", sha, flush=True); return
    m, prj = CP.build(a.device); sd = torch.load(REL, map_location="cpu", weights_only=False); pre = "codi.base_model.model."; mine = dict(m.named_parameters()); n_set = 0
    for k, v in sd.items():
        if k.startswith("prj."):
            continue
        k2 = k[len(pre):].replace(".base_layer.", ".base.").replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        if k2 in mine:
            with torch.no_grad():
                mine[k2].copy_(v.to(mine[k2].dtype)); n_set += 1
    prj.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("prj.")}); m.eval(); prj.eval()
    acc, by, _ = CP.evaluate(m, prj, CP.cp_mod().gsm8k(10_000), a.device, 6, 16)
    res = {"acc": acc, "n": 1319, "ref": REF, "tol": TOL, "pass": bool(abs(acc - REF) <= TOL), "params_loaded": n_set, "by_steps": by,
           "sha256": sha, "card": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu", "transformers": __import__("transformers").__version__}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print("G1PRIME", json.dumps({k: res[k] for k in ("acc", "pass", "card")}), flush=True)


if __name__ == "__main__":
    main()
