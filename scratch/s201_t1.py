"""THEORY §201 OVERTHINK-EXIT T1 (registered 121eb9ef; plan docs/THEORY_OVERTHINK_2026-10-14.md Part V; reader theory/score_s201.py). $0, eval-only,
fp32 (no autocast), one local card, no co-tenants. One nested model per call (halt173.load_model; DM.CHUNK_DELTA 0 = the exact token loop).
Items: test_set (halt173's: the first 1,000 per n) for n 2..12 and 16; passes t = 0..32 (t = 0: the zero state).
Arms (test only; oracle freeze point f = n + 1): CLEAN; FA: after every write past f, S_add (and den's z block) reset to S_add,f; FD: S_delta reset to
S_delta,f; FB: the whole state reset to S_f (Prop 7: = halting at f).
Per item and pass (CLEAN; also --train: 2,000 training-distribution items n 2..8, train_items(rng 19900) test-excluded, n = 1 draws skipped), dumped to
<dump dir>/<run>_{test,train}.npz (float32): ans_logits over NUM_IDS (11); aux_done bit + prob (2 greedy aux tokens: a number token then EOT, as
halt173.auxdone_fields; prob = the number-token mass x p(EOT | the greedy number token)); per delta head beta_mean / alpha_mean (token means over real
tokens), sum_log_alpha, retention rho (||S_prev P|| / ||S_prev||, P = prod_tok alpha (I - beta k k^T), by replaying the delta recurrence with v = 0
from the previous delta state, fp32); rel_inc_add_read / rel_inc_delta_read (the decoder's first-query read, halt173.read_vec's additive / delta
parts, ||r_t - r_{t-1}|| / ||r_{t-1}||); theta (the angle between the additive read r_t and r_f, test only); dinc (||S_delta,t - S_delta,t-1||).
FA / FD / FB: answers per pass, beta_mean (head mean) and dinc (T1-c).
-> <run>/eval_s201_freeze.json {arms: {ARM: {acc: {"16"|"24"|"32": {pool_3_8, pool_9_10}}, acc_by_n_t: {n: [t 0..32]}}}, gates: {fb_identity,
   clean_vs_s173_maxdiff}, mechanism (T1-c tables), card, dtype, memcheck}
    python scratch/s201_t1.py <run> [device cuda:2] [--train] [--bs 250] [--n_items 1000] [--dump_dir D]"""
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes.deltas5 import fwd_read  # noqa: E402

TMAX = 32; NS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16); ARMS = ("CLEAN", "FA", "FD", "FB")
REC = {"on": False, "d": None}; _orig = DM.write_seq


def write_seq_rec(mem, h6, hw, S, mask=None):
    Sn = _orig(mem, h6, hw, S, mask)
    if REC["on"]:                                                           # emp_gates.py's recorder: recompute u -> k / alpha / beta (the original computes S)
        HA, HD = mem.h_add, mem.h_del; B, T, D = h6.shape; a6 = mem.ln(h6.float()); aw = mem.ln_w(hw.float()); gm = mem.gamma
        u = (1 - gm) * a6 + gm * aw; _, k, _ = mem.qkv(u).split(D, -1); k = k.view(B, T, HA + HD, DM.DH); g_ = mem.gate(u)
        REC["d"] = {"al": torch.sigmoid(g_[..., :HD]), "be": 2 * torch.sigmoid(g_[..., HD:]), "kd": Fn.normalize(k[:, :, HA:], dim=-1), "mask": mask}
    return Sn


DM.write_seq = write_seq_rec


def retention(Sd_prev, d):
    """||S_prev P|| / ||S_prev|| per item and head: the delta recurrence replayed with v = 0 from S_prev (masked tokens leave it unchanged)."""
    al, be, kd, msk = d["al"], d["be"], d["kd"], d["mask"]; Sr = Sd_prev.float().clone()
    for t in range(kd.shape[1]):
        k_ = kd[:, t]; Sk = torch.einsum("bhvk,bhk->bhv", Sr, k_); Sn = al[:, t][..., None, None] * (Sr - be[:, t][..., None, None] * Sk[..., :, None] * k_[..., None, :])
        Sr = torch.where(msk[:, t][:, None, None, None], Sn, Sr)
    pn = Sd_prev.float().flatten(2).norm(dim=2); return torch.where(pn > 0, Sr.flatten(2).norm(dim=2) / pn.clamp_min(1e-30), torch.full_like(pn, float("nan")))


def run_batch(m, ch, dev, arm, f, rec):
    """one batch (fixed n for the freeze arms) -> dict of per-pass arrays (b, TMAX + 1, ...)."""
    mem = m.mem; HA, HD = mem.h_add, mem.h_del; B = len(ch); x, msk = m.think_inputs(ch, dev); lut = set(int(v) for v in N.NUM_IDS)
    rpos = torch.tensor([len(N.ids(it["syms"])) + 1 for it in ch], device=dev); den = getattr(mem, "add_norm", None) == "den"
    out = {k: [] for k in ("logits", "aux_bit", "aux_prob", "beta", "alpha", "sla", "ret", "ri_add", "ri_del", "dinc", "ra")}
    S = mem.zero_state(B, dev); Sf = None; rprev = None
    with H.ReadTap() as tap:
        for t in range(TMAX + 1):
            if t:
                REC["on"] = rec or arm != "CLEAN"; Sp = S
                try:
                    S = m.write(S, x, msk)
                finally:
                    REC["on"] = False
                if arm != "CLEAN" and t > f:                                   # the freeze: reset after the write
                    S = S.clone()
                    if arm in ("FA", "FB"):
                        S[:, :HA] = Sf[:, :HA]
                        if den:
                            S[:, HA + HD:] = Sf[:, HA + HD:]
                    if arm in ("FD", "FB"):
                        S[:, HA:HA + HD] = Sf[:, HA:HA + HD]
                d = REC["d"]; REC["d"] = None
                if d is not None:
                    mf = d["mask"].float()[..., None]; L_ = mf.sum(1)
                    out["beta"].append(((d["be"] * mf).sum(1) / L_)); out["alpha"].append(((d["al"] * mf).sum(1) / L_)); out["sla"].append((torch.log(d["al"]) * mf).sum(1))
                    out["ret"].append(retention(Sp[:, HA:HA + HD], d) if (rec and arm == "CLEAN") else torch.full((B, HD), float("nan"), device=dev))
                out["dinc"].append((S[:, HA:HA + HD] - Sp[:, HA:HA + HD]).float().flatten(1).norm(dim=1))
                if t == f:
                    Sf = S.clone()
            else:
                nanh = torch.full((B, HD), float("nan"), device=dev)
                for k_ in ("beta", "alpha", "sla", "ret"):
                    out[k_].append(nanh)
                out["dinc"].append(torch.full((B,), float("nan"), device=dev))
            tap.arm(); lg = m.dec_logits(S, ch, dev); tap.on = False; out["logits"].append(lg[:, N.NUM_IDS].float())
            xq = tap.x[torch.arange(B, device=dev), rpos]; r = H.read_vec(mem, xq, S); ra, rd = r[:, :HA * DM.DH], r[:, HA * DM.DH:]
            out["ra"].append(ra)
            if rprev is None:
                out["ri_add"].append(torch.full((B,), float("nan"), device=dev)); out["ri_del"].append(torch.full((B,), float("nan"), device=dev))
            else:
                out["ri_add"].append((ra - rprev[0]).norm(dim=1) / rprev[0].norm(dim=1).clamp_min(1e-30)); out["ri_del"].append((rd - rprev[1]).norm(dim=1) / rprev[1].norm(dim=1).clamp_min(1e-30))
            rprev = (ra, rd)
            if rec and arm == "CLEAN" and t:                                   # AUX-DONE: 2 greedy aux tokens (halt173.auxdone_fields' rule) + its probability
                pre = m.aux_prompt(t); xa = torch.tensor([pre] * B, device=dev)
                p1 = torch.softmax(m.g.logits(fwd_read(m.g, m.g.embed(xa), mem, S)[:, -1]).float(), -1); t1 = p1.argmax(-1)
                p2 = torch.softmax(m.g.logits(fwd_read(m.g, m.g.embed(torch.cat([xa, t1[:, None]], 1)), mem, S)[:, -1]).float(), -1)
                isnum = torch.tensor([int(v) in lut for v in t1.tolist()], device=dev)
                out["aux_bit"].append(isnum & (p2.argmax(-1) == N.EOT)); out["aux_prob"].append(p1[:, N.NUM_IDS].sum(-1) * p2[:, N.EOT])
            else:
                out["aux_bit"].append(torch.zeros(B, dtype=torch.bool, device=dev)); out["aux_prob"].append(torch.full((B,), float("nan"), device=dev))
    res = {k: torch.stack(v, 1).cpu() for k, v in out.items() if v}
    if f is not None:                                                         # theta_t = angle(r_add,t , r_add,f)
        ra = res.pop("ra"); rf = ra[:, f]; res["theta"] = torch.acos(Fn.cosine_similarity(ra, rf[:, None], dim=2).clamp(-1, 1))
    else:
        res.pop("ra")
    return res


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:2"
    opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d   # noqa: E731
    bs, n_items = opt("--bs", 250), opt("--n_items", 1000); dump = opt("--dump_dir", "C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/s201_dump")
    os.makedirs(dump, exist_ok=True); rd = os.path.join(ROOT, "boxes", "runs_fsq", run); t0 = time.time()
    kind, m, a = H.load_model(rd, dev); assert kind == "nested"; DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; m.eval()
    TE = N.test_set(int(a.get("per_n", 1000))); sets = {n: [it for it in TE if it["n"] == n][:n_items] for n in NS}
    torch.cuda.reset_peak_memory_stats(dev); run_batch(m, sets[16][:bs], dev, "CLEAN", 17, True); peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    assert peak < 7.0, ("memory check", peak); print(run, "memcheck n16 bs", bs, "peak %.2f GiB" % peak, flush=True)
    accs = {A: {} for A in ARMS}; dumps = {}; fb_ok = True; mech = {A: {} for A in ARMS}
    for n in NS:
        f = n + 1; tgt = torch.tensor([it["t"] for it in sets[n]])
        for A in ARMS:
            R = [run_batch(m, sets[n][i:i + bs], dev, A, f, A == "CLEAN") for i in range(0, len(sets[n]), bs)]; R = {k: torch.cat([r[k] for r in R]) for k in R[0]}
            ans = R["logits"].argmax(-1); accs[A][str(n)] = [float(v) for v in (ans == tgt[:, None]).float().mean(0)]
            mech[A][str(n)] = {"beta_mean_by_t": [float(v) for v in R["beta"].float().mean(2).nanmean(0)], "dinc_mean_by_t": [float(v) for v in R["dinc"].float().nanmean(0)]}
            if A == "CLEAN":
                dumps[n] = R; clean_logits = R["logits"]
            if A == "FB":
                fb_ok = fb_ok and all(torch.equal(R["logits"][:, k], clean_logits[:, f]) for k in range(f, TMAX + 1))
        print(run, "n", n, {A: round(accs[A][str(n)][24], 3) for A in ARMS}, "%.0fs" % (time.time() - t0), flush=True)
    pool = lambda A, k, ns: float(np.mean([accs[A][str(n)][k] for n in ns]))   # noqa: E731
    s173 = json.load(open(os.path.join(rd, "eval_s173_halt.json"))) if os.path.exists(os.path.join(rd, "eval_s173_halt.json")) else None
    dif = None
    if s173 is not None:
        bt = s173.get("acc_by_t") or s173.get("halt", {}).get("acc_by_t"); dif = max(abs(bt[str(n)][t] - accs["CLEAN"][str(n)][t]) for n in H.NEST for t in range(len(bt[str(n)])) if str(n) in bt)
    out = {"arms": {A: {"acc": {k: {"pool_3_8": pool(A, int(k), range(3, 9)), "pool_9_10": pool(A, int(k), (9, 10))} for k in ("16", "24", "32")}, "acc_by_n_t": accs[A]} for A in ARMS},
           "gates": {"fb_identity": bool(fb_ok), "clean_vs_s173_maxdiff": dif, "s173_dtype": (s173 or {}).get("eval_dtype")},
           "mechanism": mech, "card": torch.cuda.get_device_name(dev), "dtype": "float32 (no autocast)", "memcheck_peak_gib_n16": round(peak, 3), "n_items_per_n": n_items, "tmax": TMAX,
           "def": "§201 T1 (121eb9ef): oracle f = n + 1; FA / FD / FB reset after every write past f; pool_3_8 = mean of per-n accuracies n 3..8; acc_by_n_t index t = 0..32"}
    json.dump(out, open(os.path.join(rd, "eval_s201_freeze.json"), "w", newline="\n"), indent=1)
    np.savez_compressed(os.path.join(dump, run + "_test.npz"), n=np.concatenate([np.full(len(sets[n]), n) for n in NS]), t=np.concatenate([[it["t"] for it in sets[n]] for n in NS]),
                        **{k: torch.cat([dumps[n][k] for n in NS]).numpy().astype(np.float32 if dumps[NS[0]][k].dtype.is_floating_point else dumps[NS[0]][k].numpy().dtype) for k in dumps[NS[0]]})
    print(run, "T1 test done", {A: out["arms"][A]["acc"]["24"] for A in ARMS}, out["gates"], "%.0fs" % (time.time() - t0), flush=True)
    if "--train" in sys.argv:
        EX = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}; tr = [it for it in N.train_items(np.random.default_rng(19900), 4000, 8, EX) if it["n"] >= 2][:2000]
        R = [run_batch(m, tr[i:i + bs], dev, "CLEAN", None, True) for i in range(0, len(tr), bs)]; R = {k: torch.cat([r[k] for r in R]) for k in R[0]}
        np.savez_compressed(os.path.join(dump, run + "_train.npz"), n=np.array([it["n"] for it in tr]), t=np.array([it["t"] for it in tr]),
                            **{k: v.numpy().astype(np.float32 if v.dtype.is_floating_point else v.numpy().dtype) for k, v in R.items()})
        print(run, "T1 train dump done", len(tr), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
