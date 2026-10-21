"""§222 PROBE-S certification, S1 (docs/research_probe2_2026-10-21/PS_v1.md, frozen at 7174fc3b; §2 m rule, §3 CERT + E1's streams). Scratch only, no
boxes edits: the registered read path (langvb_eval.Probe / pr.scores, frozen-decoder sA - sB), recipe (Probe(m, 1e-4), bs 64, 10,000 steps, held-out
1,000 every 500, the 20k extension rule: extend once by 10k iff curve[-1] - curve[-5] > 0.02) and the lockstep groups of part_probe (218-10: probes in a
group share one rollout per batch; every group replays the batch stream; each probe has its own params and its own generator instances, so the grouping
is equivalent by construction). Only the PLANTS change: PLANT-lin-S (antisymmetric) and PLANT-xor-S (content-equivariant xor, the label is the target
row T). fp32, tf32 off, the RTX 4060 (cuda:0).
  pool / held-out  PS.Stream 22700 (+ np rng 22700 for D in 3..8; held 1,000, pool 200,000 - part_probe's layout at seed 22700)
  batches 22701; plant_dirs coefficients 22750 (plant_dirs_seeded = langvb_eval.plant_dirs with the hard-coded 22550 as a parameter; m_rule checks the
  copy is bitwise equal at 22550); lin z 22751 / 22752; xor coin 22753 / 22754; xor signs 22755 / 22756 (s1 then s3, two sequential draws; s2 = s1,
  s4 = -s3); NOPLANT coin 22757 (held-out reads 22754's coins); shuffled: each training batch's targets permuted by np rng 22758 (its plant draws:
  fresh 22753 / 22755 instances; held-out on the true targets, 22754 / 22756). Held-out generators restart at every read.
  m rule (PS §2; Theory 10-21: certification pool and dirs): untrained model, K2 split pool[2000:22000] train / pool[22000:24000] test, at t 0 and 1,
  m in {1.5, 2, 3}: K2-anti-lin = logreg on the standardised [r1, r2] (L2 tuned on a 16k / 4k split), lin z 22571 / 22572; K2-cexor = the rule oracle
  (per-row sign product about the pooled unplanted train option-row mean, pick the +1 row, ties 0.5), coin / signs 22611 / 22612 train, 22613 / 22614 test.
    python -B scratch/s222_cert.py --mode m_rule --device cuda:0 --pool_n 25000
    python -B scratch/s222_cert.py --mode cert --model untrained|ref --device cuda:0 [--resume]
  -> scratch/gate_s222/m_rule.json, cert_untrained.json / cert_ref.json (+ cert_<model>.partial.json after every group; the final file only at the end)."""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)   # D:/Latent locally, /workspace/Latent on a host
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.langvb_eval as E  # noqa: E402

OUT = "scratch/gate_s222"
REF_RUN = "scratch/gate_s218/g_l1c_model"
STREAMS = {"lin_z": [22751, 22752], "xor_coin": [22753, 22754], "xor_signs": [22755, 22756], "noplant_coin": 22757, "shuffle_perm": 22758}
MS = ("1.5", "2", "3")
KINDS = ("cexor", "lin", "noplant", "probe", "shuffled")                   # sorted, as part_probe orders keys (t, kind)


def sha_file(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def dump(name, o):
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, name + ".json"); tmp = p + ".tmp"
    json.dump(o, open(tmp, "w", newline="\n"), indent=1); os.replace(tmp, p); print("WROTE", p, flush=True)


def build_pool(pool_n, hold_n=1000, seed=22700):
    excl = E.PS.exclusion_sigs(); s = E.PS.Stream(seed, excl); rng = np.random.default_rng(seed); Dseq = rng.integers(3, 9, size=hold_n + pool_n)
    held = [E.ev(s.draw(int(D))) for D in Dseq[:hold_n]]; pool = [E.ev(s.draw(int(D))) for D in Dseq[hold_n:]]
    return held, pool


def plant_dirs_seeded(m, pool, t_list, dev, chunk, seed, n=2000):
    """= langvb_eval.plant_dirs with the coefficient generator's seed as a parameter (PS §3: 22750 for certification, 22550 for A1)."""
    out = {}; rng = np.random.default_rng(seed); tm = max(t_list)
    rows = {t: [] for t in t_list}
    for ch in E.chunks(range(min(n, len(pool))), chunk * 4):
        b, S = E.stored(m, [pool[j] for j in ch], tm, dev, chunk); sp = b[4]
        for t in t_list:
            rows[t].append(S[t][sp].double().cpu())
        del S
    for t in t_list:
        X = torch.cat(rows[t], 0); mu = X.mean(0); sd = X.std(0).clamp_min(1e-6); Xc = X - mu; C = Xc.T @ Xc / (len(X) - 1)
        ev_, Ev = torch.linalg.eigh(C); ev_, Ev = ev_.flip(0), Ev.flip(1)
        c1 = torch.tensor(rng.standard_normal(21)); c2 = torch.tensor(rng.standard_normal(21)); B_ = Ev[:, 39:60]
        u = B_ @ c1; u = u / u.norm(); u2 = B_ @ c2; u2 = u2 - (u2 @ u) * u; u2 = u2 / u2.norm(); eps = float(ev_[49].clamp_min(0).sqrt())
        vn = float(X.norm(dim=1).mean())
        out[t] = {"u": u.float().to(dev), "u2": u2.float().to(dev), "eps": eps, "eps_rel": eps / vn, "vnorm": vn, "eig": [float(v) for v in ev_[:60]],
                  "mean": mu.float().to(dev), "std": sd.float().to(dev)}
    return out


def zdraw(r_, n, dev):                                                      # = part_probe.zdraw (two arrays; PLANT-lin-S reads the first)
    return torch.tensor(r_.integers(0, 2, size=n), device=dev), torch.tensor(r_.integers(0, 2, size=n), device=dev)


def cexor_draw(rc, rs, n, dev):
    """coin T (0 = o1 is the target row) from rc; signs from rs: s1 then s3 (two sequential draws), s2 = s1, s4 = -s3 -> (T (n,), signs (n, 4))."""
    T = rc.integers(0, 2, size=n); s1 = rs.choice([-1, 1], size=n); s3 = rs.choice([-1, 1], size=n)
    return torch.tensor(T, device=dev), torch.tensor(np.stack([s1, s1, s3, -s3], 1), device=dev, dtype=torch.float32)


def slots(its, dev):
    return (torch.tensor([it["o1_slot"] for it in its], device=dev), torch.tensor([it["o2_slot"] for it in its], device=dev))


def plant_lin_s(Vt, its, z, d):
    """PLANT-lin-S: o1 row += eps (2z - 1) u, o2 row -= the same."""
    V = Vt.clone(); ar = torch.arange(V.shape[0], device=V.device); j1, j2 = slots(its, V.device)
    add = (d["eps"] * (2 * z - 1).float()[:, None] * d["u"][None]).to(V.dtype); V[ar, j1] = V[ar, j1] + add; V[ar, j2] = V[ar, j2] - add
    return V


def plant_xor_s(Vt, its, d, T, sg):
    """PLANT-xor-S: target row += eps (s1 u + s2 u2), the other option row += eps (s3 u + s4 u2)."""
    V = Vt.clone(); ar = torch.arange(V.shape[0], device=V.device); j1, j2 = slots(its, V.device)
    tgt = torch.where(T == 0, j1, j2); oth = torch.where(T == 0, j2, j1); u, u2 = d["u"][None], d["u2"][None]
    V[ar, tgt] = V[ar, tgt] + (d["eps"] * (sg[:, 0:1] * u + sg[:, 1:2] * u2)).to(V.dtype)
    V[ar, oth] = V[ar, oth] + (d["eps"] * (sg[:, 2:3] * u + sg[:, 3:4] * u2)).to(V.dtype)
    return V


def scaled(d, mult):
    d2 = dict(d); d2["eps"] = d["eps"] * mult; return d2


# ------------------------------------------------------------------ m rule (PS §2)
def logreg(Xtr, ytr, Xte, yte, l2, iters=300):
    w = torch.zeros(Xtr.shape[1], dtype=torch.float64, requires_grad=True); c = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad(); L = F.binary_cross_entropy_with_logits(Xtr @ w + c, ytr) + l2 * (w @ w); L.backward(); return L
    opt.step(clo)
    with torch.no_grad():
        return float(((Xte @ w + c > 0).double() == yte).double().mean())


def rows_opt(m, its, ts, dev, chunk, bs):
    """unplanted option rows -> {t: (N, 2, D) float64 cpu} (row 0 = o1_slot, 1 = o2_slot), one rollout to max(ts)."""
    out = {t: [] for t in ts}
    for ch in E.chunks(range(len(its)), bs * 4):
        sub = [its[j] for j in ch]; b, S = E.stored(m, sub, max(ts), dev, chunk); ar = torch.arange(len(sub), device=dev); j1, j2 = slots(sub, dev)
        for t in ts:
            out[t].append(torch.stack([S[t][ar, j1], S[t][ar, j2]], 1).double().cpu())
        del S
    return {t: torch.cat(v, 0) for t, v in out.items()}


def fit_l2(Xtr, ytr, Xte, yte, k):
    grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]; val = {str(l2): logreg(Xtr[:k], ytr[:k], Xtr[k:], ytr[k:], l2) for l2 in grid}
    best = max(grid, key=lambda l2: (val[str(l2)], l2)); return logreg(Xtr, ytr, Xte, yte, best), best, val


def k2_anti_lin(Rtr, Rte, d, e):
    ztr = torch.tensor(np.random.default_rng(22571).integers(0, 2, len(Rtr)), dtype=torch.float64); zte = torch.tensor(np.random.default_rng(22572).integers(0, 2, len(Rte)), dtype=torch.float64)
    u = d["u"].double().cpu(); sgn = torch.tensor([1.0, -1.0], dtype=torch.float64)
    Ptr = Rtr + e * (2 * ztr - 1)[:, None, None] * sgn[None, :, None] * u[None, None]; Pte = Rte + e * (2 * zte - 1)[:, None, None] * sgn[None, :, None] * u[None, None]
    Xtr, Xte = Ptr.reshape(len(Rtr), -1), Pte.reshape(len(Rte), -1); mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8); Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    acc, l2, val = fit_l2(Xtr, ztr, Xte, zte, int(0.8 * len(Xtr)))
    thr = float(((Rtr[:, 0] - Rtr[:, 1]) @ u).mean()); dp = float(((((Pte[:, 0] - Pte[:, 1]) @ u) > thr).double() == zte).double().mean())
    return {"anti_lin_logreg": acc, "anti_lin_l2": l2, "anti_lin_val_by_l2": val, "anti_lin_diffproj_acc": dp}


def k2_cexor(Rtr, Rte, d, e):
    u = d["u"].double().cpu(); u2 = d["u2"].double().cpu(); mu = Rtr.reshape(-1, Rtr.shape[-1]).mean(0); out = {}
    for nm, R, sd in (("train", Rtr, (22611, 22612)), ("test", Rte, (22613, 22614))):
        T, sg = cexor_draw(np.random.default_rng(sd[0]), np.random.default_rng(sd[1]), len(R), "cpu"); T = T.long(); sg = sg.double()
        P = R.clone(); ar = torch.arange(len(R)); P[ar, T] += e * (sg[:, 0:1] * u + sg[:, 1:2] * u2); P[ar, 1 - T] += e * (sg[:, 2:3] * u + sg[:, 3:4] * u2)
        C = P - mu; out[nm] = (C @ u, C @ u2, (T == 0).double())
    pu, pu2, y = out["test"]; sp = torch.sign(pu) * torch.sign(pu2)
    pick = torch.where(sp[:, 0] > sp[:, 1], torch.ones_like(y), torch.where(sp[:, 0] < sp[:, 1], torch.zeros_like(y), torch.full_like(y, 0.5)))
    rule = float(torch.where(pick == 0.5, torch.full_like(y, 0.5), (pick == y).double()).mean())

    def feats(o):
        a, b, _ = o; f = torch.stack([a, b, a * b], -1); return f[:, 0] - f[:, 1]
    Ftr, Fte = feats(out["train"]), feats(out["test"]); mu_f, sd_f = Ftr.mean(0), Ftr.std(0).clamp_min(1e-8); Ftr, Fte = (Ftr - mu_f) / sd_f, (Fte - mu_f) / sd_f
    acc, l2, _ = fit_l2(Ftr, out["train"][2], Fte, y, int(0.8 * len(Ftr)))
    return {"cexor_rule": rule, "cexor_rule_ties": int((pick == 0.5).sum()), "cexor_logreg": acc, "cexor_logreg_l2": l2}


def mode_m_rule(a, dev):
    t0 = time.time(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # the pool prefix does not depend on pool_n (np Generator.integers fills sequentially); checked here so a short pool_n is the cert pool's prefix
    pre = np.random.default_rng(22700).integers(3, 9, size=1000 + 200000)[:1000 + a.pool_n]; assert (pre == np.random.default_rng(22700).integers(3, 9, size=1000 + a.pool_n)).all()
    m, info = E.untrained(dev, 16); held, pool = build_pool(a.pool_n); n1, n2 = 2000 + a.k2_ntr, 2000 + a.k2_ntr + a.k2_nte; assert len(pool) >= n2
    dirs = plant_dirs_seeded(m, pool, [0, 1, 2], dev, a.chunk, 22750)
    chk = plant_dirs_seeded(m, pool, [0, 1, 2], dev, a.chunk, 22550); ref = E.plant_dirs(m, pool, [0, 1, 2], dev, a.chunk)
    copy_eq = all(torch.equal(chk[t][k], ref[t][k]) for t in (0, 1, 2) for k in ("u", "u2", "mean", "std")) and all(chk[t]["eps"] == ref[t]["eps"] for t in (0, 1, 2))
    print("SETUP", round(time.time() - t0), "s; plant_dirs copy bitwise == langvb_eval.plant_dirs at 22550:", copy_eq, flush=True); assert copy_eq
    tr, te = pool[2000:n1], pool[n1:n2]; Rtr, Rte = rows_opt(m, tr, (0, 1), dev, a.chunk, a.bs), rows_opt(m, te, (0, 1), dev, a.chunk, a.bs)
    k2 = {}
    for t in (0, 1):
        k2["t%d" % t] = {}
        for x in MS:
            e = dirs[t]["eps"] * float(x); r = {"eps": e, **k2_anti_lin(Rtr[t], Rte[t], dirs[t], e), **k2_cexor(Rtr[t], Rte[t], dirs[t], e)}
            k2["t%d" % t][x] = r; print("K2 t", t, "m", x, "anti_lin", round(r["anti_lin_logreg"], 4), "cexor_rule", round(r["cexor_rule"], 4), round(time.time() - t0), "s", flush=True)
    mm = next((x for x in MS if all(k2[t][x]["anti_lin_logreg"] >= 0.985 and k2[t][x]["cexor_rule"] >= 0.985 for t in ("t0", "t1"))), None)
    meta = {"section": 222, "registered": "PS_v1.md at 7174fc3b, section 2", "model": "langvb_eval.untrained(cal_n 16)", "untrained_cal": info,
            "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": "fp32", "tf32": False,
            "commit": os.popen("git rev-parse --short HEAD").read().strip(), "script_sha256": sha_file(__file__), "pool_stream": 22700, "pool_n_built": len(pool),
            "dirs_seed": 22750, "plant_dirs_copy_bitwise_at_22550": copy_eq, "k2_split": "pool[2000:%d] train / pool[%d:%d] test" % (n1, n1, n2),
            "k2_streams": {"lin_z": [22571, 22572], "cexor_coin": [22611, 22613], "cexor_signs": [22612, 22614]},
            "eps": {str(t): dirs[t]["eps"] for t in (0, 1, 2)}, "eps_rel": {str(t): dirs[t]["eps_rel"] for t in (0, 1, 2)}, "sec": round(time.time() - t0)}
    dump("m_rule", {"meta": meta, "k2": k2, "m": mm or "3", "flag": None if mm else "ceiling below 0.985", "rule": "PS_v1 section 2: smallest m in 1.5 / 2 / 3 with anti_lin_logreg >= 0.985 and cexor_rule >= 0.985 at t0 and t1, else 3 flagged"})


# ------------------------------------------------------------------ certification (PS §3)
class Arm:
    """one probe: kind in KINDS, pass t; its own Probe and its own generator instances (train streams continue across the extension)."""
    GENS = ("rz", "rc", "rs", "rn", "rp")

    def __init__(self, m, kind, t, dev):
        self.kind, self.t, self.dev = kind, t, dev; self.pr = E.Probe(m, 1e-4)
        self.rz = np.random.default_rng(STREAMS["lin_z"][0]); self.rc = np.random.default_rng(STREAMS["xor_coin"][0])
        self.rs = np.random.default_rng(STREAMS["xor_signs"][0]); self.rn = np.random.default_rng(STREAMS["noplant_coin"])
        self.rp = np.random.default_rng(STREAMS["shuffle_perm"])

    @property
    def key(self):
        return "probe0" if self.kind == "probe" else "%s_t%d" % (self.kind, self.t)

    def draw_train(self, n):
        """-> (labels y (n,) [index of the true option: 0 = A = o1 / gold], plant args)."""
        dv = self.dev
        if self.kind == "lin":
            z1, _ = zdraw(self.rz, n, dv); return (1 - z1).long(), (z1,)
        if self.kind in ("cexor", "shuffled"):
            T, sg = cexor_draw(self.rc, self.rs, n, dv); y = T.long()
            if self.kind == "shuffled":
                y = y[torch.tensor(self.rp.permutation(n), device=dv)]
            return y, (T, sg)
        if self.kind == "noplant":
            T = torch.tensor(self.rn.integers(0, 2, size=n), device=dv); return T.long(), ()
        return torch.zeros(n, dtype=torch.long, device=dv), ()

    def draw_held(self, n):
        """held-out / test generators restarted at every read; true targets."""
        dv = self.dev
        if self.kind == "lin":
            z1, _ = zdraw(np.random.default_rng(STREAMS["lin_z"][1]), n, dv); return (1 - z1).long(), (z1,)
        if self.kind in ("cexor", "shuffled"):
            T, sg = cexor_draw(np.random.default_rng(STREAMS["xor_coin"][1]), np.random.default_rng(STREAMS["xor_signs"][1]), n, dv); return T.long(), (T, sg)
        if self.kind == "noplant":
            T = torch.tensor(np.random.default_rng(STREAMS["xor_coin"][1]).integers(0, 2, size=n), device=dv); return T.long(), ()
        return torch.zeros(n, dtype=torch.long, device=dv), ()

    def scores(self, m, its, b, S, args, d):
        x, pad, lens, si, sp, dR, dP = b; Vt = S[self.t]
        if self.kind == "probe":
            return self.pr.scores(m, self.t, S, Vt, sp, dR, dP, lens)
        d1 = torch.tensor([E.LV.dec_ids(it["o1"]) for it in its], device=self.dev); d2 = torch.tensor([E.LV.dec_ids(it["o2"]) for it in its], device=self.dev)
        V = plant_lin_s(Vt, its, args[0], d) if self.kind == "lin" else (plant_xor_s(Vt, its, d, *args) if self.kind in ("cexor", "shuffled") else Vt)
        return self.pr.scores(m, self.t, S, V, sp, d1, d2, lens)


def sl(args, q0, q1):
    return tuple(x[q0:q1] for x in args)


# ------------------------------------------------------------------ group engine (Note 222-1: one group per host, full-state off-host checkpoints)
def provenance():
    """host / driver / image / torch / code (Note 222-1 item 4)."""
    import platform
    import socket
    mods = sorted({os.path.relpath(f).replace(os.sep, "/") for f in (E.__file__, E.PS.__file__, E.LV.__file__, E.VB.__file__)})
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "nvidia_smi": os.popen("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").read().strip(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0], "image": os.environ.get("LATENT_IMAGE", ""),
            "code_sha256": {f: sha_file(f) for f in mods}, "cert_script_sha256": sha_file(__file__)}


def arm_state(x):
    return {"P": {k: v.detach().cpu().clone() for k, v in x.pr.P.items()}, "beta": x.pr.beta.detach().cpu().clone(), "gam": x.pr.gam.detach().cpu().clone(),
            "opt": x.pr.opt.state_dict(), "steps": x.pr.steps, "curve": [list(c) for c in x.pr.curve], "gens": {g: getattr(x, g).bit_generator.state for g in x.GENS}}


def arm_load(x, s_):
    with torch.no_grad():
        for k, v in s_["P"].items():
            x.pr.P[k].copy_(v)
        x.pr.beta.copy_(s_["beta"]); x.pr.gam.copy_(s_["gam"])
    x.pr.opt.load_state_dict(s_["opt"]); x.pr.steps = s_["steps"]; x.pr.curve = [tuple(c) for c in s_["curve"]]
    for g, st in s_["gens"].items():
        getattr(x, g).bit_generator.state = st


def wjson_plain(p, o):
    tmp = p + ".tmp"; json.dump(o, open(tmp, "w", newline="\n"), indent=1); os.replace(tmp, p)


class Engine:
    """trains one lockstep group (part_probe's recipe: shared rollout per batch; extension rule; held-out every eval_every) with a FULL-STATE checkpoint
    (every probe's params + AdamW state + generator states + curve, the batch generator, ext keys, seen, torch RNG) to <ckpt_dir>/ckpt_last.pt every
    ckpt_every steps (+ ckpt_fingerprint.json {step} after it, for ckpt_mirror's companion check); --resume continues each probe exactly."""
    def __init__(self, m, pool, held, dev, a, dirs, batch_seed, tag, show_vals, script_sha):
        self.m, self.pool, self.held, self.dev, self.a, self.dirs, self.batch_seed, self.tag, self.show, self.sha = m, pool, held, dev, a, dirs, batch_seed, tag, show_vals, script_sha
        self.seen = set(); self.t0 = time.time()

    def evaluate(self, A, its, Kmax):
        a, m, dev = self.a, self.m, self.dev; ok = {x.key: [] for x in A}; lab = {x.key: x.draw_held(len(its)) for x in A}
        for ch in E.chunks(range(len(its)), a.bs):
            sub = [its[j] for j in ch]; b, S = E.stored(m, sub, Kmax, dev, a.chunk); ix = torch.tensor(ch, device=dev)
            with torch.no_grad():
                for x in A:
                    y, args = lab[x.key]; sA, sB = x.scores(m, sub, b, S, tuple(v[ix] for v in args), self.dirs.get(x.t))
                    ok[x.key] += ((sA > sB).long() == (1 - y[ix])).long().tolist()
            del S
        return ok

    def save(self, gi, A, rb, ext_keys, st):
        d = self.a.ckpt_dir; os.makedirs(d, exist_ok=True); p = os.path.join(d, "ckpt_last.pt"); tmp = p + ".tmp"
        ck = {"step": st, "group": gi, "keys": [x.key for x in A], "arms": {x.key: arm_state(x) for x in A}, "rb": rb.bit_generator.state, "ext_keys": ext_keys,
              "seen": sorted(self.seen), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(self.dev) if str(self.dev).startswith("cuda") else None,
              "script_sha256": self.sha, "tag": self.tag}
        torch.save(ck, tmp); os.replace(tmp, p)
        wjson_plain(os.path.join(d, "ckpt_fingerprint.json"), {"step": st, "group": gi, "keys": ck["keys"], "ext_keys": ext_keys, "tag": self.tag})
        print("CKPT", self.tag, "group", gi, "step", st, flush=True)

    def train_until(self, gi, T, A, target, rb, ext_keys):
        a, m, dev = self.a, self.m, self.dev; Kmax = max(x.t for x in T)
        while T[0].pr.steps < target:
            ix = rb.integers(0, len(self.pool), a.pbs); self.seen.update(int(j) for j in ix); sub = [self.pool[j] for j in ix]
            b, S = E.stored(m, sub, Kmax, dev, a.chunk); L = {}
            for x in T:
                y, args = x.draw_train(a.pbs); x.pr.opt.zero_grad(set_to_none=True); lt = 0.0
                for q0 in range(0, a.pbs, a.pmicro):
                    q1 = min(a.pbs, q0 + a.pmicro); bq = tuple(t_[q0:q1] for t_ in b); Sq = [s_[q0:q1] for s_ in S]
                    sA, sB = x.scores(m, sub[q0:q1], bq, Sq, sl(args, q0, q1), self.dirs.get(x.t))
                    lq = F.cross_entropy(torch.stack([sA, sB], 1), y[q0:q1]) * ((q1 - q0) / a.pbs); lq.backward()
                    if a.loss_log:
                        lt += float(lq.detach())
                x.pr.opt.step(); x.pr.steps += 1; L[x.key] = lt
            del S
            st = T[0].pr.steps
            if st % 100 == 0:                                               # progress for cost_guard / the drain's activity check: step + sec only (no loss)
                open(os.path.join(a.out, "log.jsonl"), "a", newline="\n").write(json.dumps({"step": st, "loss": None, "sec": int(time.time() - self.t0), "group": gi, "n_training": len(T)}) + "\n")
            if a.loss_log:
                open(a.loss_log, "a", newline="\n").write(json.dumps({"g": gi, "step": st, "loss": {k: repr(v) for k, v in L.items()}}) + "\n")
            if st % a.eval_every == 0:
                ok = self.evaluate(T, self.held, Kmax)
                for x in T:
                    x.pr.curve.append((st, float(np.mean(ok[x.key]))))
                print(self.tag, "step", st, {x.key: round(x.pr.curve[-1][1], 3) for x in T} if self.show else "%d probes" % len(T), round(time.time() - self.t0), "s", flush=True)
            if a.ckpt_dir and a.ckpt_every and st % a.ckpt_every == 0:
                self.save(gi, A, rb, ext_keys, st)
            if a.stop_at and st >= a.stop_at:
                print("STOP_AT", st, "(resume-test kill)", flush=True); sys.exit(9)

    def run_group(self, gi, gk, make_arm):
        a = self.a; A = [make_arm(k, t) for k, t in gk]; rb = np.random.default_rng(self.batch_seed); ext_keys = None
        p = os.path.join(a.ckpt_dir, "ckpt_last.pt") if a.ckpt_dir else ""
        if a.resume and p and os.path.exists(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            assert ck["group"] == gi and ck["keys"] == [x.key for x in A] and ck["script_sha256"] == self.sha and ck["tag"] == self.tag, "checkpoint does not match this group / script"
            for x in A:
                arm_load(x, ck["arms"][x.key])
            rb.bit_generator.state = ck["rb"]; ext_keys = ck["ext_keys"]; self.seen.update(ck["seen"]); torch.set_rng_state(ck["torch_rng"])
            if ck["cuda_rng"] is not None and str(self.dev).startswith("cuda"):
                torch.cuda.set_rng_state(ck["cuda_rng"], self.dev)
            print("RESUME", self.tag, "group", gi, "from step", ck["step"], "ext", ext_keys if self.show else (None if ext_keys is None else len(ext_keys)), flush=True)
        print(self.tag, "group", gi, [x.key for x in A], flush=True)
        if ext_keys is None:
            self.train_until(gi, A, A, a.steps, rb, None)
            ext_keys = [x.key for x in A if len(x.pr.curve) >= 5 and x.pr.curve[-1][1] - x.pr.curve[-5][1] > 0.02]   # curve every 500: [-5] = 2k steps earlier
        X = [x for x in A if x.key in ext_keys]
        if X:
            print(self.tag, "extend to %d:" % (2 * a.steps), ext_keys if self.show else "%d probes" % len(X), flush=True)
            self.train_until(gi, X, A, 2 * a.steps, rb, ext_keys)
        return A, ext_keys


def time_group(eng, gi, gk, make_arm, n):
    """G-TIME (Note 222-1): n training steps of this group's real shape (no eval, no checkpoint, no output beyond timing) -> s/step and peak memory."""
    A = [make_arm(k, t) for k, t in gk]; rb = np.random.default_rng(12345); a = eng.a; Kmax = max(x.t for x in A); cu = str(eng.dev).startswith("cuda")
    if cu:
        torch.cuda.reset_peak_memory_stats(eng.dev); torch.cuda.synchronize(eng.dev)
    ts = []
    for i in range(n):
        t1 = time.time(); ix = rb.integers(0, len(eng.pool), a.pbs); sub = [eng.pool[j] for j in ix]; b, S = E.stored(eng.m, sub, Kmax, eng.dev, a.chunk)
        for x in A:
            y, args = x.draw_train(a.pbs); x.pr.opt.zero_grad(set_to_none=True)
            for q0 in range(0, a.pbs, a.pmicro):
                q1 = min(a.pbs, q0 + a.pmicro)
                sA, sB = x.scores(eng.m, sub[q0:q1], tuple(t_[q0:q1] for t_ in b), [s_[q0:q1] for s_ in S], sl(args, q0, q1), eng.dirs.get(x.t))
                (F.cross_entropy(torch.stack([sA, sB], 1), y[q0:q1]) * ((q1 - q0) / a.pbs)).backward()
            x.pr.opt.step()
        del S
        if cu:
            torch.cuda.synchronize(eng.dev)
        ts.append(time.time() - t1)
    w = ts[min(10, n // 4):]
    return {"group": gi, "keys": [x.key for x in A], "Kmax": Kmax, "steps_timed": n, "sec_per_step": float(np.mean(w)), "sec_per_step_median": float(np.median(w)),
            "peak_gib": torch.cuda.max_memory_allocated(eng.dev) / 2 ** 30 if cu else None, "gpu": torch.cuda.get_device_name(eng.dev) if cu else "cpu",
            "note": "timing only: batches from np rng 12345, plant / label draws from fresh probe generators; no eval, no output"}


def group_sel(a, n):
    sel = list(range(n)) if a.groups in ("", "all") else [int(g) for g in a.groups.split(",")]
    assert all(0 <= g < n for g in sel), (sel, n)
    if a.ckpt_dir:
        assert len(sel) == 1, "--ckpt_dir takes exactly one --groups index (one group per process / host)"
    return sel


def merge_files(paths, out_p, kind):
    """per-group files -> one file. All metas must agree except provenance / timing / group fields; arms (cert) or probe / plant_s (A1) united."""
    F_ = [json.load(open(p)) for p in paths]; skip = {"provenance", "sec", "groups_run"}
    base = {k: v for k, v in F_[0]["meta"].items() if k not in skip}
    for f, p in zip(F_[1:], paths[1:]):
        o = {k: v for k, v in f["meta"].items() if k not in skip}
        assert o == base, "meta mismatch in %s: %s" % (p, sorted(k for k in set(o) | set(base) if o.get(k) != base.get(k)))
    meta = dict(base); meta["provenance_by_group"] = {str(g): f["meta"]["provenance"] for f in F_ for g in f["meta"]["groups_run"]}
    meta["group_files"] = {os.path.basename(p): sha_file(p) for p in paths}; done = sorted(g for f in F_ for g in f["meta"]["groups_run"])
    assert done == list(range(len(meta["groups"]))), "groups %s != all %d" % (done, len(meta["groups"]))
    if kind == "cert":
        arms = {}
        for f in F_:
            assert not set(arms) & set(f["arms"]); arms.update(f["arms"])
        wjson_plain(out_p, {"meta": meta, "arms": arms, "def": F_[0].get("def")})
    else:
        pr, pl = {}, {}
        for f in F_:
            assert not set(pr) & set(f["probe"]); pr.update(f["probe"])
            for t, v in f["plant_s"].items():
                d = pl.setdefault(t, {k: v[k] for k in ("eps", "eps_rel", "pcs")})
                for k in ("lin", "cexor"):
                    if k in v:
                        assert k not in d; d[k] = v[k]
        o = {"meta": meta, "probe": pr, "plant_s": pl}
        for k in ("k2_guard", "diag_align", "meta_pre", "k2_guard_sha256"):
            o[k] = F_[0].get(k)
        wjson_plain(out_p, {**o, "def": F_[0].get("def")})
    print("MERGED", len(paths), "->", out_p, flush=True)


def mode_cert(a, dev):
    t0 = time.time(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    mr = json.load(open(a.m_rule)); M = mr["m"]; mult = float(M); sha = sha_file(__file__)
    if a.model == "ref":
        m, info = E.load(REF_RUN, dev); info = {"model_run": REF_RUN, "model_sha256": sha_file(os.path.join(REF_RUN, "model.pt")), **{k: info.get(k) for k in ("arch", "arch_class", "source", "items_sha", "steps_run")}}
    else:
        m, info = E.untrained(dev, 16); info = {"model": "langvb_eval.untrained(cal_n 16)", "untrained_cal": info}
    held, pool = build_pool(a.pool_n, a.hold_n); dirs0 = plant_dirs_seeded(m, pool, [0, 1, 2], dev, a.chunk, 22750); dirs = {t: scaled(dirs0[t], mult) for t in dirs0}
    raw = E.PS.test_set(a.per_D); TE = {D: [E.ev(it) for it in its] for D, its in raw.items()}
    keys = sorted([("probe", 0)] + [(k, t) for t in (0, 1, 2) for k in ("cexor", "lin", "noplant", "shuffled")], key=lambda k: (k[1], k[0]))
    G = a.group; groups = [keys[i:i + G] for i in range(0, len(keys), G)]; sel = group_sel(a, len(groups))
    meta = {"section": 222, "registered": "PS_v1.md at 7174fc3b, section 3 (S1); Note 222-1 (one group per host)", "set": "S1", "model_kind": a.model, **info,
            "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": "fp32", "tf32": False,
            "script_sha256": sha, "m": M, "m_rule_flag": mr["flag"], "m_rule_sha256": sha_file(a.m_rule),
            "dirs_seed": 22750, "eps": {str(t): dirs[t]["eps"] for t in dirs}, "eps_rel": {str(t): dirs[t]["eps_rel"] for t in dirs},
            "eps_base": {str(t): dirs0[t]["eps"] for t in dirs0}, "pool_stream": 22700, "pool_n": len(pool), "held_n": len(held), "batches": 22701,
            "streams": dict(STREAMS), "shuffled_plant_streams": "fresh instances of xor_coin / xor_signs (train), true targets on 22754 / 22756 (held-out)",
            "steps": a.steps, "eval_every": a.eval_every, "ext_rule": "curve[-1] - curve[-5] > 0.02 (last 2k gain) -> +%d steps once" % a.steps,
            "pbs": a.pbs, "pmicro": a.pmicro, "bs": a.bs, "chunk": a.chunk, "groups": [[("probe0" if k[0] == "probe" else "%s_t%d" % k) for k in g] for g in groups],
            "test_set": {"per_D": a.per_D, "test_items_sha": E.PS.set_sha([it for D in sorted(raw) for it in raw[D]]), "probe0_D": [D for D in range(3, 13) if D in TE]},
            "ckpt_every": a.ckpt_every, "provenance": provenance(), "groups_run": sel}
    print("SETUP", round(time.time() - t0), "s m", M, "eps", meta["eps"], "groups", meta["groups"], "run", sel, flush=True)
    eng = Engine(m, pool, held, dev, a, dirs, 22701, "CERT", True, sha)
    if a.time_steps:
        tj = {**time_group(eng, sel[0], groups[sel[0]], lambda k, t: Arm(m, k, t, dev), a.time_steps), "provenance": meta["provenance"], "model_kind": a.model, "script_sha256": sha}
        wjson_plain(os.path.join(a.out, "timing.json"), tj); print("TIMING", json.dumps({k: tj[k] for k in ("group", "Kmax", "sec_per_step", "peak_gib", "gpu")}), flush=True); return
    files = []
    for gi in sel:
        gp = os.path.join(a.out, "cert_%s.g%d.json" % (a.model, gi)); files.append(gp)
        if a.resume and os.path.exists(gp):
            print("group", gi, "file exists - skipped", flush=True); continue
        tg = time.time(); A, ext = eng.run_group(gi, groups[gi], lambda k, t: Arm(m, k, t, dev)); arms = {}
        for x in A:
            c = x.pr.curve; arms[x.key] = {"kind": x.kind, "t": x.t, "curve": [list(v) for v in c], "LATE": float(np.mean([v for _, v in c[-3:]])), "steps_run": x.pr.steps,
                                           "extended": x.key in ext, "n_held": len(held), "group": gi}
        p0 = [x for x in A if x.kind == "probe"]
        if p0:
            x = p0[0]; Ds = meta["test_set"]["probe0_D"]; its = [it for D in Ds for it in TE[D]]; Dl = [it["D"] for it in its]; ok = eng.evaluate([x], its, 0)[x.key]
            arms["probe0"].update({"test_acc": float(np.mean(ok)), "test_acc_by_D": {str(D): float(np.mean([o for o, d in zip(ok, Dl) if d == D])) for D in Ds},
                                   "n_test": len(ok), "items": {str(D): [o for o, d in zip(ok, Dl) if d == D] for D in Ds}})
            print("CERT probe0 test", round(arms["probe0"]["test_acc"], 4), flush=True)
        del A
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        gm = {**meta, "groups_run": [gi], "sec": round(time.time() - tg)}
        wjson_plain(gp, {"meta": gm, "arms": arms, "n_distinct_seen_group": len(eng.seen), "def": __doc__}); print("WROTE", gp, "group", gi, "done", gm["sec"], "s", flush=True)
    if sel == list(range(len(groups))):
        merge_files(files, os.path.join(a.out, "cert_%s.json" % a.model), "cert")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("m_rule", "cert"), required=True); ap.add_argument("--model", choices=("untrained", "ref"), default="untrained")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--pool_n", type=int, default=200000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--pbs", type=int, default=64)
    ap.add_argument("--pmicro", type=int, default=16); ap.add_argument("--group", type=int, default=7); ap.add_argument("--per_D", type=int, default=1000)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--hold_n", type=int, default=1000); ap.add_argument("--k2_ntr", type=int, default=20000); ap.add_argument("--k2_nte", type=int, default=2000)
    ap.add_argument("--out", default=OUT, help="output dir (m_rule.json; cert: per-group cert_<model>.g<i>.json, + cert_<model>.json when every group ran in this process)")
    ap.add_argument("--m_rule", default=os.path.join(OUT, "m_rule.json")); ap.add_argument("--groups", default="all", help="Note 222-1: comma list of group indices (default all)")
    ap.add_argument("--ckpt_dir", default="", help="full-state checkpoint dir (ckpt_last.pt + ckpt_fingerprint.json); needs exactly one --groups index")
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--time_steps", type=int, default=0, help="G-TIME: time this many steps of the selected group -> <out>/timing.json, exit")
    ap.add_argument("--loss_log", default="", help="resume test only: per-step per-probe CE (repr) jsonl"); ap.add_argument("--stop_at", type=int, default=0, help="resume test only: exit 9 after this step")
    ap.add_argument("--merge", default="", help="comma list of per-group cert files -> --merge_out"); ap.add_argument("--merge_out", default="")
    a = ap.parse_args(); globals()["OUT"] = a.out
    if a.merge:
        return merge_files(a.merge.split(","), a.merge_out, "cert")
    os.makedirs(a.out, exist_ok=True)
    (mode_m_rule if a.mode == "m_rule" else mode_cert)(a, a.device)


if __name__ == "__main__":
    main()
