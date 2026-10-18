"""SD2 draft check ($0, CPU fp32, 2 threads): peak working-set increase of ONE eval rollout (vbank_eval.part_acc's main rollout: K' = Kmax with a coda
at every K' in the grid) for FT-SELF's model, per (n, Kmax, B, item choice). Used only to scale the measured GPU facts to n 24 (memory ~ B (K'+1) T).
Read-only on the repo (model.pt / args.json loaded; nothing saved).  usage: python evalmem_cpu.py n Kmax B longest|order"""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
sys.path.insert(0, "D:/Latent")
import psutil, torch
torch.set_num_threads(2)
import boxes.nested170 as N, boxes.vbank as VB, boxes.vbank_eval as VE
n, K, B, how = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
P = psutil.Process()
m, a = VE.load("D:/Latent/boxes/runs_fsq/s215_ftself_s1", "cpu")
TE = [it for it in N.test_set(1000, max(n, 16)) if it["n"] == n]
its = sorted(TE, key=lambda it: -len(it["syms"]))[:B] if how == "longest" else TE[:B]
x, pad, lens = VB.batch_x(its, "cpu"); T = x.shape[1]
base = P.memory_info().peak_wset
with torch.no_grad():
    r = m.rollout(x, pad, K, coda_rows={t: torch.arange(B) for t in range(0, K + 1)})
peak = P.memory_info().peak_wset
st = {"V": sum(v.numel() * v.element_size() for v in r["V"]), "kv": sum(k.numel() * k.element_size() + v.numel() * v.element_size() for L in r["kv"].values() for k, v in L),
      "coda": sum(h.numel() * h.element_size() for _, h in r["coda"].values())}
out = {"n": n, "K": K, "B": B, "how": how, "T": T, "delta_peak_gib": (peak - base) / 2 ** 30, "stored_gib": {k: v / 2 ** 30 for k, v in st.items()},
       "per_unit_bytes": (peak - base) / (B * (K + 1) * T)}
print(json.dumps(out))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalmem_cpu.jsonl"), "a", newline="\n") as f:
    f.write(json.dumps(out) + "\n")
