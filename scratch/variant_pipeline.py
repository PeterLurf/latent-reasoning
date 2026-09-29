"""Frozen Part II battery for an ablation variant or extra seed (PROTOCOL3). Skips steps whose output exists."""
import os
import subprocess
import sys

run, dev, tag = sys.argv[1], sys.argv[2], sys.argv[3]
full = len(sys.argv) > 4 and sys.argv[4] == "full"          # full = every Part II experiment (seed replication)
py = [sys.executable, "-m"]


def complete(out):
    """An output counts as done only if it exists and parses (a crash mid-write leaves a truncated file)."""
    if not os.path.exists(out):
        return False
    try:
        if out.endswith(".json"):
            import json
            json.load(open(out))
        elif out.endswith(".pt"):
            import torch
            torch.load(out, map_location="cpu", weights_only=False)
        return True
    except Exception as e:  # noqa: BLE001 - any failure to parse means redo
        print("incomplete", out, type(e).__name__, flush=True)
        return False


def sh(args, out):
    if complete(out):
        print("skip", out, flush=True)
        return
    subprocess.run(py + args, check=True)


fu = lambda sub: os.path.join(run, "followup", sub, "summary.json")
sh(["latent.competence", run, "--out", f"results/competence_{tag}.json", "--n", "4096", "--device", dev],
   f"results/competence_{tag}.json")
sh(["latent.competence_matched", run, "--out", f"results/competence_matched_{tag}.json", "--device", dev],
   f"results/competence_matched_{tag}.json")
sh(["latent.study", "discover", run, "--device", dev], os.path.join(run, "discovery", "discovery.json"))
sh(["latent.study", "subspace", run, "--device", dev], os.path.join(run, "discovery", "patterns.pt"))
sh(["latent.mechanism", run, "--device", dev, "--out", f"results/extra_{tag}.json"], f"results/extra_{tag}.json")
exps = ["dose", "swap", "cont", "cache", "attn", "gain", "jac"] if full else ["dose", "cache", "attn"]
sh(["latent.followup_run", "--run", run, "--split", "fresh", "--n", "300", "--device", dev, "--experiments", *exps], fu("fresh"))
if full:
    sh(["latent.followup_run", "--run", run, "--split", "discovery", "--device", dev, "--experiments", "jac"], fu("discovery"))
    sh(["latent.followup_run", "--run", run, "--split", "fresh", "--n", "300", "--device", dev, "--experiments", "pair", "--tag", "pair"],
       fu("fresh_pair"))
sh(["latent.followup_run", "--run", run, "--split", "fresh", "--n", "300", "--device", dev, "--experiments", "amp", "--tag", "amp"],
   fu("fresh_amp"))
sh(["latent.followup_run", "--run", run, "--split", "fresh", "--n", "300", "--device", dev, "--experiments", "cache", "--tag", "cachedepth"],
   fu("fresh_cachedepth"))
for split in ("discovery", "fresh"):
    sh(["latent.followup_run", "--run", run, "--split", split, "--device", dev, "--experiments", "gainfit",
        "--widths", "1", "2", "3", "4", "5", "6", "7", "8", "--n", "300" if split == "fresh" else "200", "--tag", "gainfit"],
       fu(f"{split}_gainfit"))
if True:                            # PROTOCOL3 addenda E-G (P15, P16, budget runs): every recurrent model here
    sh(["latent.budget_sweep", run, "--out", f"results/budget_sweep_{tag}.json", "--device", dev], f"results/budget_sweep_{tag}.json")
    sh(["latent.length_gen", run, "--out", f"results/length_gen_{tag}.json", "--device", dev], f"results/length_gen_{tag}.json")
    sh(["latent.upfront_trace", run, "--delayed", "--out", f"results/delayed_trace_{tag}.json", "--device", dev, "--n", "400"],
       f"results/delayed_trace_{tag}.json")
print("DONE", run, flush=True)
