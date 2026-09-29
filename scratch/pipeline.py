"""Run the full frozen analysis pipeline for one trained run."""
import json
import subprocess
import sys

run, dev, tag = sys.argv[1], sys.argv[2], sys.argv[3]
py = [sys.executable, "-m"]
subprocess.run(py + ["latent.competence", run, "--out", f"results/competence_{tag}.json", "--n", "4096", "--device", dev], check=True)
subprocess.run(py + ["latent.study", "discover", run, "--device", dev], check=True)
subprocess.run(py + ["latent.study", "subspace", run, "--device", dev], check=True)
k = json.load(open(f"{run}/discovery/discovery.json"))["k_star"]
P = json.load(open("docs/protocol_frozen.json"))
P["k_star_by_run"][run] = k          # value produced by the frozen site rule
json.dump(P, open("docs/protocol_frozen.json", "w"), indent=2)
print(f"{run}: k* = {k} (frozen rule)", flush=True)
subprocess.run(py + ["latent.study", "confirm", run, "--protocol", "docs/protocol_frozen.json", "--device", dev], check=True)
subprocess.run(py + ["latent.mechanism", run, "--device", dev], check=True)
