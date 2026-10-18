"""LANG-VB draft $0: size of the anchor-proximity anti-cue subset of p8_na_test by depth (for the A0-1 cue-reliance eval)."""
import collections, json, sys
sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/scripts")
from heur_panel import panel_for, anchors
d = json.load(open(sys.argv[1])); c = collections.defaultdict(collections.Counter)
for x in d:
    if int(x.get("ext", 0)) < 2:
        continue
    at, an = anchors(x); v = panel_for(x, at, an)["proximity"]
    c[len(x["steps"])]["pro" if v == 1.0 else "anti" if v == 0.0 else "tie"] += 1
tot = collections.Counter()
for k in sorted(c):
    print(k, dict(c[k])); tot.update(c[k])
print("total m>=2", dict(tot))
