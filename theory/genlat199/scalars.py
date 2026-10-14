import math, torch, psutil, json, sys
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS); torch.set_num_threads(3)
RUNS = "D:/Latent/boxes/runs_fsq/"
d = json.load(open("trunk_drift.json"))
for run, r in d.items():
    sd = torch.load(RUNS + run + "/model.pt", map_location="cpu", weights_only=False)
    a = json.load(open(RUNS + run + "/args.json"))
    sc = {k: round(float(v), 4) for k, v in sd.items() if k.startswith("mem.") and v.numel() == 1}
    ag = [round(float(x), 3) for x in sd["mem.add_gain"].flatten()] if "mem.add_gain" in sd else None
    print(run, r["group"], "steps", a.get("steps"), "lr", a.get("lr"), "add_norm", a.get("add_norm"), sc, "add_gain", ag)
    del sd
# pure weight-decay trajectory of a parameter with no gradient: b_T = b_0 * prod(1 - lr_t * wd), lr_t = fsq120.lr_at (warmup 100, cosine over steps)
def lr_at(base, step, steps): return base * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
for T in (6000, 8000, 20000, 48000):
    p = 1.0
    for s in range(1, T + 1): p *= 1 - lr_at(1e-4, s, 48000) * 0.01
    b = 5 * p; al = 1 / (1 + math.exp(-b)); a0 = 1 / (1 + math.exp(-5))
    print(f"step {T}: decay factor {p:.5f}, alpha-bias {b:.4f}, alpha/token {al:.5f} (init {a0:.5f}); per-pass log change at 30 tokens {30*(math.log(al)-math.log(a0)):+.4f}")
