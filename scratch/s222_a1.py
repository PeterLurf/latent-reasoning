"""§222 PROBE-S part 2 on A1 (PS_v1.md at 7174fc3b, section 4; layout agreed with Theory 10-21): S1 = §218's registered probe (langvb_eval.Probe, pr.scores,
10,000 steps, bs 64, held-out 1,000 every 500, the 20k extension rule, lockstep groups of G-EVMEM's 8 at pmicro 16 / chunk 16 / eval_bs 16, fp32, the 4060)
with the S1 plants (PLANT-lin-S antisymmetric, PLANT-xor-S content-equivariant) at m (scratch/gate_s222/m_rule.json) on A1's OWN plant directions
(langvb_eval.plant_dirs, rng 22550, on A1's pool rows; pool / held-out PS.Stream 22500 as §218).
  A1 plant streams (PS §3): lin z = zdraw's first array, 22551 train / 22553 held-out + test; xor: coin T = the first array and s1 = the second array
  (0 -> -1, 1 -> +1) of zdraw on 22552 train / 22554 held-out + test; s3 = choice([-1, 1]) on a new generator 22558 train / 22559 held-out + test;
  s2 = s1, s4 = -s3. Each probe has its own generator instances (grouping-invariant, as the certification); held-out / test generators restart per read.
  Parts:  k2guard  K2-anti-lin (logreg, z 22571 / 22572) and K2-cexor (rule oracle, 22611-22614) at t 0 / 1 / 2 on A1's planted option rows at m,
                   split pool[2000:22000] / [22000:24000] (PS §4).
          diag     DIAG-ALIGN (descriptive; read only with the probe part): on A1's unplanted held-out option rows at t 0-2, centred as K2.
          probe    PROBE(t) for t in T_SET (§218's out_p layout) + plant_s lin / cexor at t 0-2 (test-set read on D 3..8, as §218's plant read).
  k2guard / diag go to <run>/eval_lvb_s222_pre.json; the probe part writes <run>/eval_lvb_s222.json (merging the pre file) only at its end
  (per-group progress in <run>/eval_lvb_s222.partial.json), so score_s222 never reads a part file.
    python -B scratch/s222_a1.py --run boxes/runs_fsq/lvb_sym_s1 --parts k2guard,diag | probe --device cuda:0 [--resume]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.langvb_eval as E  # noqa: E402
import s222_cert as C  # noqa: E402

A1_STREAMS = {"lin_z": [22551, 22553], "xor_coin_s1": [22552, 22554], "xor_s3": [22558, 22559]}


def wjson(p, o):
    tmp = p + ".tmp"; json.dump(o, open(tmp, "w", newline="\n"), indent=1); os.replace(tmp, p); print("WROTE", p, flush=True)


def a1_xor_draw(rzs, r3, n, dev):
    """T = zdraw's first array, s1 = its second (0 -> -1, 1 -> +1); s3 from r3 -> (T, signs [s1, s1, s3, -s3])."""
    T, b = zdraw_np(rzs, n); s1 = 2 * b - 1; s3 = r3.choice([-1, 1], size=n)
    return torch.tensor(T, device=dev), torch.tensor(np.stack([s1, s1, s3, -s3], 1), device=dev, dtype=torch.float32)


def zdraw_np(r_, n):
    return r_.integers(0, 2, size=n), r_.integers(0, 2, size=n)


class A1Arm:
    def __init__(self, m, kind, t, dev):
        self.kind, self.t, self.dev = kind, t, dev; self.pr = E.Probe(m, 1e-4)
        self.rz = np.random.default_rng(A1_STREAMS["lin_z"][0]); self.rx = np.random.default_rng(A1_STREAMS["xor_coin_s1"][0]); self.r3 = np.random.default_rng(A1_STREAMS["xor_s3"][0])

    @property
    def key(self):
        return "%s%d" % (self.kind, self.t)

    def labels(self, n, held):
        dv = self.dev
        if self.kind == "lin":
            z1, _ = C.zdraw(np.random.default_rng(A1_STREAMS["lin_z"][1]) if held else self.rz, n, dv); return (1 - z1).long(), (z1,)
        if self.kind == "cexor":
            rx, r3 = (np.random.default_rng(A1_STREAMS["xor_coin_s1"][1]), np.random.default_rng(A1_STREAMS["xor_s3"][1])) if held else (self.rx, self.r3)
            T, sg = a1_xor_draw(rx, r3, n, dv); return T.long(), (T, sg)
        return torch.zeros(n, dtype=torch.long, device=dv), ()

    def scores(self, m, its, b, S, args, d):
        x, pad, lens, si, sp, dR, dP = b; Vt = S[self.t]
        if self.kind == "probe":
            return self.pr.scores(m, self.t, S, Vt, sp, dR, dP, lens)
        d1 = torch.tensor([E.LV.dec_ids(it["o1"]) for it in its], device=self.dev); d2 = torch.tensor([E.LV.dec_ids(it["o2"]) for it in its], device=self.dev)
        V = C.plant_lin_s(Vt, its, args[0], d) if self.kind == "lin" else C.plant_xor_s(Vt, its, d, *args)
        return self.pr.scores(m, self.t, S, V, sp, d1, d2, lens)


def rows_slot_keys(m, its, ts, keys, dev, chunk, bs):
    """unplanted rows at the given slot keys -> {t: (N, len(keys), D) float64 cpu}."""
    out = {t: [] for t in ts}
    for ch in E.chunks(range(len(its)), bs * 4):
        sub = [its[j] for j in ch]; b, S = E.stored(m, sub, max(ts), dev, chunk); ar = torch.arange(len(sub), device=dev)
        js = [torch.tensor([it[k] for it in sub], device=dev) for k in keys]
        for t in ts:
            out[t].append(torch.stack([S[t][ar, j] for j in js], 1).double().cpu())
        del S
    return {t: torch.cat(v, 0) for t, v in out.items()}


def diag_align(Rh, mu, d, o1_is_gold):
    """Rh (N, 4, D): rows o1, o2, gold (oR), other (oP); centred by mu (K2's pooled train option-row mean)."""
    u = d["u"].double().cpu(); u2 = d["u2"].double().cpu(); C_ = Rh - mu; diff = C_[:, 2] - C_[:, 3]; out = {"n": len(Rh), "ref_1_over_768": 1 / 768}
    for nm, v in (("u", u), ("u2", u2)):
        p = diff @ v; out["dprime_" + nm] = float(p.mean() / p.std().clamp_min(1e-12))
        g = diff.mean(0); out["cos2_%s_g" % nm] = float((g @ v) ** 2 / (g @ g).clamp_min(1e-30))
        rows = torch.cat([C_[:, 0], C_[:, 1]], 0) @ v; out["var_%s_over_l50" % nm] = float(rows.var() / d["eig"][49])
    y = o1_is_gold.double(); sd = (C_[:, 0] - C_[:, 1]) @ u; out["gold_acc_sign_diff_u"] = float(((sd > 0).double() == y).double().mean())
    sp = torch.sign(C_[:, :2] @ u) * torch.sign(C_[:, :2] @ u2)
    pick = torch.where(sp[:, 0] > sp[:, 1], torch.ones_like(y), torch.where(sp[:, 0] < sp[:, 1], torch.zeros_like(y), torch.full_like(y, 0.5)))
    out["gold_acc_cexor_rule"] = float(torch.where(pick == 0.5, torch.full_like(y, 0.5), (pick == y).double()).mean())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--parts", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool_n", type=int, default=200000); ap.add_argument("--hold_n", type=int, default=1000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--pbs", type=int, default=64); ap.add_argument("--pmicro", type=int, default=16); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--per_D", type=int, default=1000); ap.add_argument("--k2_ntr", type=int, default=20000); ap.add_argument("--k2_nte", type=int, default=2000)
    ap.add_argument("--ts", default=",".join(str(t) for t in E.T_SET)); ap.add_argument("--m_rule", default="scratch/gate_s222/m_rule.json"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); dev = a.device; parts = a.parts.split(","); t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    mr = json.load(open(a.m_rule)); M = mr["m"]; mult = float(M)
    m, info = E.load(a.run, dev); info = {"model_run": a.run, "model_sha256": C.sha_file(os.path.join(a.run, "model.pt")), **{k: info.get(k) for k in ("arch", "arch_class", "source", "items_sha", "steps_run")}}
    excl = E.PS.exclusion_sigs(); s = E.PS.Stream(22500, excl); rng = np.random.default_rng(22500); Dseq = rng.integers(3, 9, size=a.hold_n + a.pool_n)
    held = [E.ev(s.draw(int(D))) for D in Dseq[:a.hold_n]]; pool = [E.ev(s.draw(int(D))) for D in Dseq[a.hold_n:]]
    dirs0 = E.plant_dirs(m, pool, list(E.PLANT_TS), dev, a.chunk); dirs = {t: C.scaled(dirs0[t], mult) for t in dirs0}
    meta = {"section": 222, "registered": "PS_v1.md at 7174fc3b, section 4", "set": "S1", **info, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "dtype": "fp32", "tf32": False, "commit": os.popen("git rev-parse --short HEAD").read().strip(), "script_sha256": C.sha_file(__file__),
            "cert_script_sha256": C.sha_file(C.__file__), "m": M, "m_rule_flag": mr["flag"], "dirs_seed": 22550, "pool_stream": 22500, "pool_n": len(pool),
            "held_n": len(held), "batches": 22501, "plant_streams": dict(A1_STREAMS), "eps": {str(t): dirs[t]["eps"] for t in dirs}, "eps_base": {str(t): dirs0[t]["eps"] for t in dirs0}}
    print("SETUP", round(time.time() - t0), "s m", M, flush=True)
    pre_p = os.path.join(a.run, "eval_lvb_s222_pre.json"); pre = json.load(open(pre_p)) if os.path.exists(pre_p) else {}
    if "k2guard" in parts or "diag" in parts:
        n1, n2 = 2000 + a.k2_ntr, 2000 + a.k2_ntr + a.k2_nte; Rtr = C.rows_opt(m, pool[2000:n1], (0, 1, 2), dev, a.chunk, a.bs); Rte = C.rows_opt(m, pool[n1:n2], (0, 1, 2), dev, a.chunk, a.bs)
        pre["meta_pre"] = {**meta, "k2_split": "pool[2000:%d] / pool[%d:%d]" % (n1, n1, n2), "k2_streams": {"lin_z": [22571, 22572], "cexor_coin": [22611, 22613], "cexor_signs": [22612, 22614]}}
        if "k2guard" in parts:
            pre["k2_guard"] = {}
            for t in (0, 1, 2):
                e = dirs[t]["eps"]; al = C.k2_anti_lin(Rtr[t], Rte[t], dirs0[t], e); cx = C.k2_cexor(Rtr[t], Rte[t], dirs0[t], e)
                pre["k2_guard"][str(t)] = {"anti_lin": al["anti_lin_logreg"], "cexor": cx["cexor_rule"], "cexor_logreg": cx["cexor_logreg"], "cexor_rule_ties": cx["cexor_rule_ties"],
                                           "anti_lin_diffproj_acc": al["anti_lin_diffproj_acc"], "eps": e, "n_train": len(Rtr[t]), "n_test": len(Rte[t])}
            print("K2GUARD done", round(time.time() - t0), "s", flush=True)
        if "diag" in parts:
            Rh = rows_slot_keys(m, held, (0, 1, 2), ("o1_slot", "o2_slot", "oR_slot", "oP_slot"), dev, a.chunk, a.bs)
            o1g = torch.tensor([it["o1_slot"] == it["oR_slot"] for it in held])
            pre["diag_align"] = {str(t): diag_align(Rh[t], Rtr[t].reshape(-1, Rtr[t].shape[-1]).mean(0), dirs0[t], o1g) for t in (0, 1, 2)}
            print("DIAG done", round(time.time() - t0), "s", flush=True)
        wjson(pre_p, pre)
    if "probe" not in parts:
        return
    ts = [int(x) for x in a.ts.split(",")]; raw = E.PS.test_set(a.per_D); TE = {D: [E.ev(it) for it in its] for D, its in raw.items()}
    keys = sorted([("probe", t) for t in ts] + [(k, t) for t in E.PLANT_TS for k in ("cexor", "lin")], key=lambda k: (k[1], k[0]))
    groups = [keys[i:i + a.group] for i in range(0, len(keys), a.group)]
    meta.update({"steps": a.steps, "eval_every": a.eval_every, "ext_rule": "curve[-1] - curve[-5] > 0.02 -> +%d steps once" % a.steps, "pbs": a.pbs, "pmicro": a.pmicro,
                 "bs": a.bs, "chunk": a.chunk, "groups": [["%s%d" % k for k in g] for g in groups], "test_items_sha": E.PS.set_sha([it for D in sorted(raw) for it in raw[D]])})
    part_p = os.path.join(a.run, "eval_lvb_s222.partial.json"); out_p, out_pl, done_g = {}, {}, []
    if a.resume and os.path.exists(part_p):
        P = json.load(open(part_p)); assert P["meta"]["script_sha256"] == meta["script_sha256"] and P["meta"]["m"] == M; out_p, out_pl, done_g = P["probe"], P["plant_s"], P["groups_done"]
    seen = set()

    def evaluate(A, its, Kmax):
        ok = {x.key: [] for x in A}; lab = {x.key: x.labels(len(its), True) for x in A}
        for ch in E.chunks(range(len(its)), a.bs):
            sub = [its[j] for j in ch]; b, S = E.stored(m, sub, Kmax, dev, a.chunk); ix = torch.tensor(ch, device=dev)
            with torch.no_grad():
                for x in A:
                    y, args = lab[x.key]; sA, sB = x.scores(m, sub, b, S, tuple(v[ix] for v in args), dirs[x.t] if x.kind != "probe" else None)
                    ok[x.key] += ((sA > sB).long() == (1 - y[ix])).long().tolist()
            del S
        return ok

    def train(A, n_steps, rb):
        Kmax = max(x.t for x in A)
        for _ in range(n_steps):
            ix = rb.integers(0, len(pool), a.pbs); seen.update(int(j) for j in ix); sub = [pool[j] for j in ix]; b, S = E.stored(m, sub, Kmax, dev, a.chunk)
            for x in A:
                y, args = x.labels(a.pbs, False); x.pr.opt.zero_grad(set_to_none=True)
                for q0 in range(0, a.pbs, a.pmicro):
                    q1 = min(a.pbs, q0 + a.pmicro); bq = tuple(t_[q0:q1] for t_ in b); Sq = [s_[q0:q1] for s_ in S]
                    sA, sB = x.scores(m, sub[q0:q1], bq, Sq, C.sl(args, q0, q1), dirs[x.t] if x.kind != "probe" else None)
                    (F.cross_entropy(torch.stack([sA, sB], 1), y[q0:q1]) * ((q1 - q0) / a.pbs)).backward()
                x.pr.opt.step(); x.pr.steps += 1
            del S
            st = A[0].pr.steps
            if st % a.eval_every == 0:
                ok = evaluate(A, held, Kmax)
                for x in A:
                    x.pr.curve.append((st, float(np.mean(ok[x.key]))))
                print("A1S222 step", st, len(A), "probes", round(time.time() - t0), "s", flush=True)   # values not printed (A1 readings stay unread)

    for gi, gk in enumerate(groups):
        if gi in done_g:
            continue
        A = [A1Arm(m, k, t, dev) for k, t in gk]; rb = np.random.default_rng(22501); print("A1S222 group", gi, [x.key for x in A], flush=True); train(A, a.steps, rb)
        ext = [x for x in A if len(x.pr.curve) >= 5 and x.pr.curve[-1][1] - x.pr.curve[-5][1] > 0.02]
        if ext:
            print("A1S222 extend:", len(ext), "probes", flush=True); train(ext, a.steps, rb)
        for x in A:
            Ds = [D for D in range(3, 13 if x.kind == "probe" else 9) if D in TE]; its = [it for D in Ds for it in TE[D]]; Dl = [it["D"] for it in its]
            ok = evaluate([x], its, x.t)[x.key]; byD = {str(D): float(np.mean([o for o, d_ in zip(ok, Dl) if d_ == D])) for D in Ds}
            rec = {"acc_by_D": byD, "items": {str(D): [o for o, d_ in zip(ok, Dl) if d_ == D] for D in Ds}, "curve": x.pr.curve, "steps_run": x.pr.steps, "extended": x in ext, "group": gi}
            if x.kind == "probe":
                out_p[str(x.t)] = rec
            else:
                d = out_pl.setdefault(str(x.t), {"eps": dirs[x.t]["eps"], "eps_rel": dirs[x.t]["eps_rel"], "pcs": dirs[x.t]["eig"]})
                rec["acc"] = float(np.mean(ok)); rec["items_flat"] = ok; d[x.kind] = rec
        done_g.append(gi); del A
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print("A1S222 group", gi, "done", round(time.time() - t0), "s", flush=True)
        wjson(part_p, {"meta": meta, "probe": out_p, "plant_s": out_pl, "groups_done": done_g})
    pre = json.load(open(pre_p)) if os.path.exists(pre_p) else {}
    meta["sec"] = round(time.time() - t0)
    wjson(os.path.join(a.run, "eval_lvb_s222.json"), {"meta": meta, "probe": out_p, "plant_s": out_pl, "k2_guard": pre.get("k2_guard"), "diag_align": pre.get("diag_align"),
                                                      "meta_pre": pre.get("meta_pre"), "probe_n_distinct_seen": len(seen), "def": __doc__})


if __name__ == "__main__":
    main()
