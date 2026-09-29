import json
import sys

for r in sys.argv[1:]:
    L = [json.loads(l) for l in open(f"runs/{r}/log.jsonl") if '"sec"' in l]
    c = json.load(open(f"runs/{r}/config.json"))
    s = L[-1]
    rate = s["sec"] / s["step"]
    print(f"{r:20s} {c['device']} step {s['step']:>6} sec {s['sec']:7.0f}  {rate * 1000:.0f} ms/step  "
          f"ETA {(c['steps'] - s['step']) * rate / 60:.0f} min")
