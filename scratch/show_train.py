import json
import sys

run = sys.argv[1]
for l in open(f"runs/{run}/log.jsonl"):
    x = json.loads(l)
    if "eval" in x:
        e = x["eval"]["delayed"]
        print(x["step"], f"bal {e['bal_acc']:.3f}", "depth", {k: round(v, 2) for k, v in e["by_depth"].items()},
              "upfront", round(x["eval"]["upfront"]["bal_acc"], 3))
    elif "curriculum_advance_to" in x:
        print("   ADVANCE", x)
    elif x["step"] % 1000 == 0:
        print("  ", x["step"], "loss", round(x["loss"], 3), "delayed", round(x["loss_delayed"], 3), "parent", round(x["loss_parent"], 4),
              "children", round(x["loss_children"], 4), "gn", round(x["gn"], 2), "bdepth", x["max_bdepth"])
