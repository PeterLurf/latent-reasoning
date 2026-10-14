"""§199 note 199-A5 (Theory, pre-stated before the readout): gradient shares of the CSR term vs the task loss on kept checkpoints. At each of the run's
ckpt_model_<step>.pt (4k / 8k / 12k / 16k), on 4 fixed batches of 64 TRAINING items (train_items(rng 19950), n <= n_max_train, test-excluded), K =
think_k, the run's recipe (uncached inject + rms; fade / gamma buffers as saved at that step), fp32 (no autocast), the exact delta loop:
  task = answer CE + aux_step * aux (nested170.loss_items: aux = the item-mean of each item's token-mean aux CE over its passes t <= #steps)
  csr  = w * CSR (nested170.read_parts / csr_lams / csr_terms with the run's --csr w, lam_min, lam_post, tau; h6 from gpt_lower under no_grad, as training)
Each as a SEPARATE backward; the gradient is the mean over the 4 batches (each batch accumulated over 8-item chunks with weights 8/64, exact for the
batch-mean losses; the csr term is a per-item mean too). Reported: ||g||_2 over ALL parameters and over the memory branch (m.mem.*), plus ||W_o||_F;
also the mean over batches of the per-batch norms.
-> <run>/eval_s199_gradshare.json {step: {G_csr_all, G_task_all, G_csr_mem, G_task_mem, wo_norm, ...}}
    python scratch/s199_gradshare.py <run> [device cuda:2] [--chunk 8]"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402

NB, BSZ = 4, 64


def losses(m, a, items, dev, CSR):
    """(task, csr) for a chunk, as nested170.loss_items computes them at lambda = aux_step (fp32)."""
    g = m.g; K = int(a["think_k"]); x, msk = m.think_inputs(items, dev); per = [[] for _ in items]
    S = m.mem.zero_state(len(items), dev)
    with torch.no_grad():
        h6c = DM.gpt_lower(g, g.embed(x), m.mem.layer)
    ra_, rd_ = N.read_parts(m.mem, h6c, S); R_ = [ra_ + rd_]
    for t in range(1, K + 1):
        S = m.write(S, x, msk); ra_, rd_ = N.read_parts(m.mem, h6c, S); R_.append(ra_ + rd_)
        idx = [i for i, it in enumerate(items) if len(N.aux_of(it)) >= t]
        if idx:
            lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev)
            ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
            for r_, i in enumerate(idx):
                per[i].append(ce[r_])
    la = Fn.cross_entropy(m.dec_logits(S, items, dev), torch.tensor(N.NUM_IDS[[it["t"] for it in items]], device=dev))
    aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()
    csr = N.csr_terms(N.csr_lams(R_, msk), [it["n"] for it in items], K, CSR[1], CSR[2], CSR[3])
    return la + float(a["aux_step"]) * aux, CSR[0] * csr


def gnorm(m, sel):
    v = [p.grad.double().pow(2).sum() for n, p in m.named_parameters() if p.grad is not None and sel(n)]
    return float(torch.stack(v).sum().sqrt()) if v else 0.0


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:2"
    ch = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 8; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    a = json.load(open(os.path.join(rd, "args.json"))); assert a.get("csr") and a["arm"] == "inject"; CSR = [float(v) for v in a["csr"].split(",")]
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0
    g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a); m = N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None).to(dev); m.nopass = bool(a.get("aux_nopass"))
    TE = N.test_set(int(a.get("per_n") or 1000)); EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    rng = np.random.default_rng(19950); batches = [N.train_items(rng, BSZ, int(a.get("n_max_train") or 8), EX) for _ in range(NB)]
    out = {}; fp = os.path.join(rd, "eval_s199_gradshare.json")
    for st in (4000, 8000, 12000, 16000):
        ckp = os.path.join(rd, "ckpt_model_%d.pt" % st)
        if not os.path.exists(ckp):
            out[str(st)] = {"missing": True}; _final(rd, out); continue
        ck = torch.load(ckp, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); m.train(); assert int(ck["step"]) == st
        rec = {}
        for term in ("task", "csr"):
            m.zero_grad(set_to_none=True); per_all, per_mem = [], []
            for b in batches:
                before = {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}
                for i in range(0, BSZ, ch):
                    tk, cs = losses(m, a, b[i:i + ch], dev, CSR); ((tk if term == "task" else cs) * (len(b[i:i + ch]) / BSZ) / NB).backward()
                diff_all = sum(float((p.grad.double() - (before[n].double() if before[n] is not None else 0)).pow(2).sum()) for n, p in m.named_parameters() if p.grad is not None)
                diff_mem = sum(float((p.grad.double() - (before[n].double() if before[n] is not None else 0)).pow(2).sum()) for n, p in m.named_parameters() if p.grad is not None and n.startswith("mem."))
                per_all.append(np.sqrt(diff_all) * NB); per_mem.append(np.sqrt(diff_mem) * NB)   # this batch's own gradient norm (undo the 1/NB)
            rec["G_%s_all" % term] = gnorm(m, lambda n: True); rec["G_%s_mem" % term] = gnorm(m, lambda n: n.startswith("mem."))
            rec["G_%s_all_mean_of_batch_norms" % term] = float(np.mean(per_all)); rec["G_%s_mem_mean_of_batch_norms" % term] = float(np.mean(per_mem))
        rec["wo_norm"] = float(m.mem.wo.weight.detach().norm()); rec["gamma"] = float(m.mem.gamma); rec["fade"] = float(m.mem.fade)
        rec["ratio_csr_over_task_all"] = rec["G_csr_all"] / max(rec["G_task_all"], 1e-30); rec["ratio_csr_over_task_mem"] = rec["G_csr_mem"] / max(rec["G_task_mem"], 1e-30)
        out[str(st)] = rec; print(run, st, {k: round(v, 5) for k, v in rec.items()}, flush=True); m.zero_grad(set_to_none=True); del ck
        json.dump({"by_step": out, "csr": CSR, "batches": "4 x 64 train_items(rng 19950), n <= n_max_train, test-excluded", "chunk": ch, "dtype": "float32 (no autocast)",
                   "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                   "def": "note 199-A5: G = ||grad||_2 of the batch-mean-of-4 gradient, separate backward per term (task = CE + aux_step aux; csr = w CSR); mem = m.mem.* params"},
                  open(fp, "w", newline="\n"), indent=1)


def _final(rd, out):                                                        # also record missing steps (the per-step dump only runs for present ones)
    fp = os.path.join(rd, "eval_s199_gradshare.json")
    if os.path.exists(fp):
        j = json.load(open(fp)); j["by_step"].update({k: v for k, v in out.items() if k not in j["by_step"]}); json.dump(j, open(fp, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
