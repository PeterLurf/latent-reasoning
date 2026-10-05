"""THEORY §115 (eval only): snapping on §108's Markov models (mk108) - (A) no snaps at all, (B) adaptive snapping by
latent/text disagreement. Writes <run>/eval_s115.json.
(A) nosnap: the Markov recurrence from the identity text at position 0, then [EOT, carried latent, op] segments for every
    position (no text restart). Keys: nosnap.cumulative {128, 256, 512, 2048}; nosnap.per_handoff_by_position (P(right at
    t | right at t-1), mean per 64-position bin); nosnap.drift every 64 positions {latent_norm (mean ||carried latent||),
    norm_ratio (/ mean norm at positions 1-4), cos_to_snapped (cosine with the carried latent at the same position of the
    standard k = 4 snapped rollout of the same problem)}.
(B) trigger: at every hand-off t >= 1 both the latent path [EOT, latent_{t-1}, op_t] and the text path [EOT, decoded text of
    state t-1, op_t] are run; if their slot decodes agree the latent path continues, else the chain snaps (continues from
    the text path's latent and decode); a forced snap after 64 hand-offs without one. Stats on hand-offs whose chain is
    right up to t-1: first_error_detection (rows' first wrong latent-path hand-off: share where the paths disagreed;
    also per_event), false_snap_rate (latent right but snapped (not forced) / hand-offs), miss_rate (latent wrong but the
    paths agreed / hand-offs), forced_snap_rate; trigger.cumulative {512, 2048}; snaps_per_problem {mean, median}.
    fixed_k4: the standard rollout's cumulative and snaps per problem (L / 4 - 1).
    python -m boxes.snap115 --run boxes/runs_s5/s108_mk4_s1 [--device cuda:0]
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.gen97 import NST, STTOK, amp_ctx, decode_to_el  # noqa: E402
from boxes.mk108 import EOT, MK108  # noqa: E402

CAP = 64


@torch.no_grad()
def seg(m, prev, ops_t):
    """One Markov segment [EOT, prev (B, P, D), op] -> (carried latent s_in (B, D), slot decode tokens (B, 6))."""
    g = m.g; B = ops_t.shape[0]; eot = g.embed(torch.full((B, 1), EOT, device=ops_t.device))
    x = torch.cat([eot, prev.to(eot.dtype), g.embed(m.optok[ops_t])], 1)
    h, past = g(x, None); s_in = h[:, -1]; hs, past = g(s_in[:, None], past)
    return s_in, m.greedy(hs[:, 0], past)


@torch.no_grad()
def nosnap(m, X, Y, dev, amp, bs, snap_lat=None):
    """Returns pred (N, L) and carried latents every 64 positions (N, L/64, D) (float32)."""
    N, L = X.shape; pred = np.full((N, L), -1); keep = list(range(0, L, 64)); lat = np.zeros((N, len(keep), 768), np.float32)
    early = np.zeros(N)
    for s in range(0, N, bs):
        sl = slice(s, s + bs); n = len(X[sl]); prev = m.g.embed(torch.tensor(np.tile(STTOK[0], (n, 1)), device=dev)); ki = 0; en = []
        for t in range(L):
            with amp_ctx(amp):
                s_in, d = seg(m, prev, torch.tensor(X[sl, t], device=dev))
            pred[sl, t] = decode_to_el(d.cpu().numpy())[0]; prev = s_in[:, None]
            if t < 4:
                en.append(s_in.float().norm(dim=-1).cpu().numpy())
            if t in keep:
                lat[sl, ki] = s_in.float().cpu().numpy(); ki += 1
        early[sl] = np.mean(en, 0)
    return pred, lat, early


@torch.no_grad()
def snapped(m, X, dev, amp, bs, k=4):
    """The standard rollout (snap every k), returning pred and the carried latents every 64 positions."""
    N, L = X.shape; pred = np.full((N, L), -1); keep = list(range(0, L, 64)); lat = np.zeros((N, len(keep), 768), np.float32)
    for s in range(0, N, bs):
        sl = slice(s, s + bs); n = len(X[sl]); start = torch.tensor(np.tile(STTOK[0], (n, 1)), device=dev); ki = 0
        for c in range(0, L, k):
            with amp_ctx(amp):
                _, d, ins = m.chunk(start, torch.tensor(X[sl, c:c + k], device=dev), decode="all")
            toks = d.cpu().numpy(); pred[sl, c:c + k] = decode_to_el(toks.reshape(-1, NST))[0].reshape(n, -1)
            for jj in range(ins.shape[1]):
                if c + jj in keep:
                    lat[sl, ki] = ins[:, jj].float().cpu().numpy(); ki += 1
            start = torch.tensor(toks[:, -1], device=dev)
    return pred, lat


@torch.no_grad()
def trigger(m, X, Y, dev, amp, bs):
    N, L = X.shape; pred = np.full((N, L), -1); snaps = np.zeros(N, int)
    ev = {"handoffs": 0, "false_snap": 0, "miss": 0, "forced": 0, "lat_wrong": 0, "lat_wrong_detected": 0}
    first_det = []
    for s in range(0, N, bs):
        sl = slice(s, s + bs); n = len(X[sl]); Yb = Y[sl]; g = m.g
        dec = torch.tensor(np.tile(STTOK[0], (n, 1)), device=dev)
        with amp_ctx(amp):
            s_in, dec = seg(m, g.embed(dec), torch.tensor(X[sl, 0], device=dev))
        el = decode_to_el(dec.cpu().numpy())[0]; pred[sl, 0] = el; since = np.zeros(n, int); seen_err = np.zeros(n, bool)
        for t in range(1, L):
            op = torch.tensor(X[sl, t], device=dev)
            with amp_ctx(amp):
                sl_lat, d_lat = seg(m, s_in[:, None], op); sl_txt, d_txt = seg(m, g.embed(dec), op)
            agree = (d_lat == d_txt).all(1).cpu().numpy(); forced = since >= CAP; sn = (~agree) | forced
            e_lat = decode_to_el(d_lat.cpu().numpy())[0]; e_txt = decode_to_el(d_txt.cpu().numpy())[0]
            ok_prev = (pred[sl, :t] == Yb[:, :t]).all(1); lat_ok = e_lat == Yb[:, t]
            ev["handoffs"] += int(ok_prev.sum()); ev["false_snap"] += int((ok_prev & lat_ok & ~agree).sum())
            ev["miss"] += int((ok_prev & ~lat_ok & agree).sum()); ev["forced"] += int((ok_prev & forced & agree).sum())
            ev["lat_wrong"] += int((ok_prev & ~lat_ok).sum()); ev["lat_wrong_detected"] += int((ok_prev & ~lat_ok & ~agree).sum())
            fe = ok_prev & ~lat_ok & ~seen_err
            first_det += list((~agree)[fe]); seen_err |= fe
            snt = torch.tensor(sn, device=dev)
            s_in = torch.where(snt[:, None], sl_txt, sl_lat); dec = torch.where(snt[:, None], d_txt, d_lat)
            pred[sl, t] = np.where(sn, e_txt, e_lat); snaps[sl] += sn; since = np.where(sn, 0, since + 1)
    h = max(1, ev["handoffs"]); C = pred == Y
    return {"cumulative": {str(L): float(C.all(1).mean())}, "cum_curve": {str(t): float(C[:, :t].all(1).mean()) for t in (128, 256, 512, 1024, 2048) if t <= L},
            "snaps_per_problem": {"mean": float(snaps.mean()), "median": float(np.median(snaps))},
            "first_error_detection": float(np.mean(first_det)) if first_det else None, "n_first_errors": len(first_det),
            "per_event_detection": ev["lat_wrong_detected"] / max(1, ev["lat_wrong"]),
            "false_snap_rate": ev["false_snap"] / h, "miss_rate": ev["miss"] / h, "forced_snap_rate": ev["forced"] / h, "events": ev}


@torch.no_grad()
def g97_step(m, x, past):
    """gen97 chunk step: x (B, T, D) through the cache, then the slot; -> (cache after the slot, slot decode tokens)."""
    g = m.g; h, past = g(x, past); s_in = h[:, -1]; hs, past = g(s_in[:, None], past)
    return past, m.greedy(hs[:, 0], past)


@torch.no_grad()
def trigger_g97(m, X, Y, dev, amp):
    """§115 (B) for gen97 (§105 masked fork): latent path = continue the row's chunk cache with op t; text path = a new chunk
    from the decoded text of state t-1. Rows' caches are kept per row and batched by cache length."""
    N, L = X.shape; g = m.g; pred = np.full((N, L), -1); snaps = np.zeros(N, int); since = np.zeros(N, int)
    ev = {"handoffs": 0, "false_snap": 0, "miss": 0, "forced": 0, "lat_wrong": 0, "lat_wrong_detected": 0}; first_det = []; seen = np.zeros(N, bool)
    E = lambda toks: g.embed(torch.tensor(toks, device=dev))  # noqa: E731
    def batched(rows_x, idx):                                               # text path for rows idx: fresh chunks
        with amp_ctx(amp):
            pa, d = g97_step(m, rows_x, None)
        return [[(k[i:i + 1], v[i:i + 1]) for k, v in pa] for i in range(len(idx))], d.cpu().numpy()
    caches, dec0 = batched(torch.cat([E(np.tile(STTOK[0], (N, 1))), g.embed(m.optok[torch.tensor(X[:, 0], device=dev)])], 1), range(N))
    dec = dec0; pred[:, 0] = decode_to_el(dec)[0]
    for t in range(1, L):
        opv = g.embed(m.optok[torch.tensor(X[:, t], device=dev)]); d_lat = np.zeros((N, NST), np.int64); new_lat = [None] * N
        lens = np.array([c[0][0].shape[2] for c in caches])
        for ln in np.unique(lens):                                         # latent path, batched by cache length
            ii = np.where(lens == ln)[0]; past = [(torch.cat([caches[i][l][0] for i in ii]), torch.cat([caches[i][l][1] for i in ii])) for l in range(len(caches[0]))]
            with amp_ctx(amp):
                pa, d = g97_step(m, opv[ii], past)
            d_lat[ii] = d.cpu().numpy()
            for r_, i in enumerate(ii):
                new_lat[i] = [(k[r_:r_ + 1], v[r_:r_ + 1]) for k, v in pa]
        new_txt, d_txt = batched(torch.cat([E(dec), opv], 1), range(N))
        agree = (d_lat == d_txt).all(1); forced = since >= CAP; sn = (~agree) | forced
        e_lat = decode_to_el(d_lat)[0]; e_txt = decode_to_el(d_txt)[0]
        ok_prev = (pred[:, :t] == Y[:, :t]).all(1); lat_ok = e_lat == Y[:, t]
        ev["handoffs"] += int(ok_prev.sum()); ev["false_snap"] += int((ok_prev & lat_ok & ~agree).sum()); ev["miss"] += int((ok_prev & ~lat_ok & agree).sum())
        ev["forced"] += int((ok_prev & forced & agree).sum()); ev["lat_wrong"] += int((ok_prev & ~lat_ok).sum()); ev["lat_wrong_detected"] += int((ok_prev & ~lat_ok & ~agree).sum())
        fe = ok_prev & ~lat_ok & ~seen; first_det += list((~agree)[fe]); seen |= fe
        caches = [new_txt[i] if sn[i] else new_lat[i] for i in range(N)]; dec = np.where(sn[:, None], d_txt, d_lat)
        pred[:, t] = np.where(sn, e_txt, e_lat); snaps += sn; since = np.where(sn, 0, since + 1)
    h = max(1, ev["handoffs"]); C = pred == Y
    return {"cumulative": {str(L): float(C.all(1).mean())}, "cum_curve": {str(t): float(C[:, :t].all(1).mean()) for t in (128, 256, 512) if t <= L},
            "snaps_per_problem": {"mean": float(snaps.mean()), "median": float(np.median(snaps))},
            "first_error_detection": float(np.mean(first_det)) if first_det else None, "n_first_errors": len(first_det),
            "per_event_detection": ev["lat_wrong_detected"] / max(1, ev["lat_wrong"]),
            "false_snap_rate": ev["false_snap"] / h, "miss_rate": ev["miss"] / h, "forced_snap_rate": ev["forced"] / h, "events": ev}


def run_markov(m, run, dev, amp, bs=250, test_n=s5.TEST_N, ext_n=500, data_dir=os.path.join(ROOT, "bench", "data", "s5"), k=4, parts="A,B", max_len=0):
    """§115 (A)/(B) on a loaded mk108-style Markov model (also called in-process by mk108 --s115 for §118); writes
    <run>/eval_s115.json. No-snap uses only the latent chain: no test-time text path, reader or decode feedback."""
    t0 = time.time(); m.eval()
    Xte, Yte = s5.load_csv(os.path.join(data_dir, "S5=512.csv"), limit=test_n); s5.check_labels(Xte, Yte)
    Xx, Yx = s5.load_csv(os.path.join(data_dir, "S5=2048.csv"), limit=ext_n); s5.check_labels(Xx, Yx)
    if max_len:
        Xte, Yte, Xx, Yx = Xte[:, :max_len], Yte[:, :max_len], Xx[:, :max_len], Yx[:, :max_len]
    out = {"run": run, "k": k, "eval_dtype": str(amp), "test_n": len(Xte), "ext_n": len(Xx), "cap": CAP}
    if "A" in parts:
        p5, l5, e5 = nosnap(m, Xte, Yte, dev, amp, bs); ps, ls = snapped(m, Xte, dev, amp, bs, k)
        px, _, _ = nosnap(m, Xx, Yx, dev, amp, bs // 2)
        C = p5 == Yte; ho = [(C[:, t] & C[:, t - 1]).sum() / max(1, C[:, t - 1].sum()) for t in range(1, C.shape[1])]
        nrm = np.linalg.norm(l5, axis=-1); cos = (l5 * ls).sum(-1) / (nrm * np.linalg.norm(ls, axis=-1) + 1e-9)
        out["nosnap"] = {"cumulative": {**{str(t): float(C[:, :t].all(1).mean()) for t in (128, 256, 512) if t <= C.shape[1]}, "2048": float((px == Yx).all(1).mean())},
                         "per_handoff_by_position": {str(b): float(np.mean(ho[b:b + 64])) for b in range(0, len(ho), 64)},
                         "drift": {str(t): {"latent_norm": float(nrm[:, i].mean()), "norm_ratio": float((nrm[:, i] / e5).mean()), "cos_to_snapped": float(cos[:, i].mean())}
                                   for i, t in enumerate(range(0, Xte.shape[1], 64))}}
        Cs = ps == Yte; out["fixed_k4"] = {"cumulative": {str(t): float(Cs[:, :t].all(1).mean()) for t in (128, 256, 512) if t <= Cs.shape[1]}, "snaps_per_problem": Xte.shape[1] // k - 1}
        print("A", json.dumps(out["nosnap"]["cumulative"]), flush=True)
    if "B" in parts:
        tr = trigger(m, Xte, Yte, dev, amp, bs); tx = trigger(m, Xx, Yx, dev, amp, bs // 2)
        tr["cumulative"][str(Xx.shape[1])] = tx["cumulative"][str(Xx.shape[1])]; tr["ext_2048"] = tx; out["trigger"] = tr
        print("B", json.dumps({k: tr[k] for k in ("cumulative", "snaps_per_problem", "first_error_detection", "false_snap_rate", "miss_rate")}), flush=True)
    out["sec"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(run, "eval_s115.json"), "w", newline="\n"), indent=1); print("DONE", run, flush=True)


def main_g97(a, dev, amp):
    """§115 (B) on a gen97 run (g105_mask4_s*): n problems at 512 (114-layout approval: n = 500, 512 only)."""
    from boxes.gen97 import Gen97, rollout
    Xte, Yte = s5.load_csv(os.path.join(a.data_dir, "S5=512.csv"), limit=a.test_n); s5.check_labels(Xte, Yte)
    if a.max_len:
        Xte, Yte = Xte[:, :a.max_len], Yte[:, :a.max_len]
    args = json.load(open(os.path.join(a.run, "args.json"))); m = Gen97(s5.load_gpt2(dev), "gen")
    m.mask = args.get("fork") == "mask"; m.to(dev); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev)); m.eval()
    t0 = time.time(); tr = trigger_g97(m, Xte, Yte, dev, amp)
    ps, _, _ = rollout(m, Xte, a.k, dev, amp, a.bs); Cs = ps == Yte
    out = {"run": a.run, "model": "gen97", "fork": args.get("fork"), "k": a.k, "eval_dtype": str(amp), "test_n": len(Xte), "cap": CAP, "trigger": tr,
           "fixed_k4": {"cumulative": {str(t): float(Cs[:, :t].all(1).mean()) for t in (128, 256, 512) if t <= Cs.shape[1]}, "snaps_per_problem": Xte.shape[1] // a.k - 1},
           "sec": round(time.time() - t0)}
    json.dump(out, open(os.path.join(a.run, "eval_s115.json"), "w", newline="\n"), indent=1)
    print("B", json.dumps({k: tr[k] for k in ("cumulative", "snaps_per_problem", "first_error_detection", "false_snap_rate", "miss_rate")}), json.dumps(out["fixed_k4"]), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--test_n", type=int, default=s5.TEST_N); ap.add_argument("--ext_n", type=int, default=500); ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--parts", default="A,B"); ap.add_argument("--data_dir", default=os.path.join(ROOT, "bench", "data", "s5"))
    ap.add_argument("--max_len", type=int, default=0, help="smoke only: truncate sequences"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="mk108", choices=["mk108", "g97"], help="g97: (B) only, n = test_n at 512 (layout: 500)")
    a = ap.parse_args(); dev = a.device; t0 = time.time()
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    if a.model == "g97":
        main_g97(a, dev, amp); return
    m = MK108(s5.load_gpt2(dev)).to(dev); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev)); m.eval()
    run_markov(m, a.run, dev, amp, a.bs, a.test_n, a.ext_n, a.data_dir, a.k, a.parts, a.max_len)


if __name__ == "__main__":
    main()
