"""PROTOCOL8 confirmatory configs: arms x seeds from the Phase 0 Coconut recipe (p8_coconut.yaml).
    python make_p8c_configs.py [--seeds 1,2,3,4] [--anytime_C 0]
Arms:
  coco   standard Coconut, answer mask (as Phase 0)
  at     + anytime: C (patch_p8)
  rt     + random_tight: True (patch_p8)
  cot    text chain of thought (cot: True, 30 epochs), the C3 baseline
Writes p8c_<arm>_s<seed>.yaml."""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1,2,3,4"); ap.add_argument("--anytime_C", type=int, default=0)
a = ap.parse_args()
base = open("p8_coconut.yaml").read()
base = re.sub(r"^#.*\n", "", base, flags=re.M)
for s in map(int, a.seeds.split(",")):
    for arm in ("coco", "at", "rt", "cot"):
        y = base.replace("name: p8-coconut", f"name: p8c-{arm}-s{s}").replace("seed: 0", f"seed: {s}")
        if arm == "at":
            y += f"anytime: {a.anytime_C}\n"
        if arm == "rt":
            y += "random_tight: True\n"
        if arm == "cot":
            y = re.sub(r"^coconut: True$", "coconut: False", y, flags=re.M)
            y = re.sub(r"^cot: False$", "cot: True", y, flags=re.M)
            y = re.sub(r"^num_epochs: 48$", "num_epochs: 30", y, flags=re.M)
        open(f"p8c_{arm}_s{s}.yaml", "w", newline="\n").write(f"# PROTOCOL8 confirmatory, arm {arm}, seed {s}\n" + y)
print("written")
