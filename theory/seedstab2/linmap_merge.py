"""SEEDSTAB2 rev 2: merge linmap_check.log (part 1 was cut by a 1100 s wall-clock limit after 11 rows, so its JSON was never written; part 2
re-ran the 5 missing configurations with the same code and seeds) into linmap_check.json. Each log row: store op map best-acc(seed 0, seed 1)."""
import json, os, re

here = os.path.dirname(os.path.abspath(__file__)); out = {}
for line in open(os.path.join(here, "linmap_check.log")):
    m = re.match(r"(\w+)\s+([+*])\s+(\w+)\s+best train acc \['([\d.]+)', '([\d.]+)'\]", line)
    if m:
        out["%s_%s_%s" % m.group(1, 2, 3)] = [float(m.group(4)), float(m.group(5))]
out["_note"] = ("best training accuracy over Adam steps, seeds 0 / 1; linear = rev-2 map (per-head stop-gradient batch-RMS constants, global cosine, "
                "no bias); perhead = rev-1 map (per-head RMS + bias); rows 1-11 from the first run (cut at 1100 s), rows 12-16 from part 2")
json.dump(out, open(os.path.join(here, "linmap_check.json"), "w", newline="\n"), indent=1)
print(len(out) - 1, "rows")
for k, v in out.items():
    if not k.startswith("_"):
        print("%-22s %s" % (k, v))
