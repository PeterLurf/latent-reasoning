"""DAS-USED pilot (docs/THEORY_DASUSED_DRAFT_2026-10-14.md rev 2, 271db224, §7; THEORY notes 203-2 / 203-2a). A NEW file: boxes/*.py are imported
read-only (halt173.load_model, nested170, deltamem); DM.FAST_ADD = args.fast_add, DM.CHUNK_DELTA = 0, READ_CLAMP None. The state patch runs in this
script's own pass loop (no monkeypatch): S_t -> patch -> m.write for passes t+1..K -> m.dec_logits (= think(override=(t, .))).
Stage P0a (this file's first block; forward only): the CONDITIONED whole-state SERIAL (draft §2.5, notes 203-2 / 203-2a)
  (i)  s170 re-run: SPOOL (n = 8, rng 17300, 2,200 items), serial_items(rng 17200) (an exact copy that also returns the hybrid ITEM; gate SI: its
       recipients / donors / hybrid values equal nested170.serial_items'), swap the donor's whole state after pass t in {1, 2, 4, 6} (t < K), K passes.
       Per pair: the clean answers (K' = K, no swap) of recipient / donor / hybrid item and the swapped answer. GATE S170: the unconditioned hybrid /
       donor / recipient rates equal eval.json s170.serial[t] within 0.02 where eval has t.
  (ii) n 3-8 extension: the same donor construction on the test items (all <= 1,000 per n), rng 20371, swaps at t in {1, 2}; eligibility per t: the
       root has a bare leaf, h != root, height(root) >= 2t + 1, hybrid answer not in {own, donor}.
  Readings per (model, t[, n]): hybrid / own / donor / other shares on ALL rows and on CLEAN rows (ccc: recipient, donor and hybrid item all answered
  correctly without a swap), each with its N (note 203-2: N < 100 -> N/A).
Every block records card / dtype ("bf16 autocast trunk, fp32 memory branch") / bs_eval / torch / driver / script git hash / wall seconds.
-> <run>/eval_dasused.json (block "serial_cond") + <run>/eval_dasused_rows.npz (serial_i_t*, serial_ii_t*)
    python scratch/dasused.py <run> --device cuda:1 --stage P0 [--bs 250] [--memcheck]"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402

DTYPE = "bf16 autocast trunk, fp32 memory branch"


def git_hash(path):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", path], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return None


def height(nd):
    return 0 if nd.op is None else 1 + max(height(nd.l), height(nd.r))


def serial_items_h(items, rng):
    """nested170.serial_items VERBATIM (same rng draws, same keep rule), additionally returning the hybrid ITEM (hb: h's operands replaced, the bare
    operand kept) and the handle path. -> (rec, don, hyb_value, hyb_item)."""
    rec, don, hyb, hyb_it = [], [], [], []
    for it in items:
        root = it["root"]
        if root.op is None:
            continue
        side = "l" if root.l.op is None else ("r" if root.r.op is None else None)
        path = N._leftmost_ready(root)
        if side is None or path is None or len(path) == 0:
            continue
        got = None
        for _ in range(10):
            d = N._copy(root); hn = d
            for p_ in path:
                hn = getattr(hn, p_)
            v = N.apply(hn.op, hn.l.v, hn.r.v); v2 = int(rng.choice([x for x in range(N.P) if x != v])); prs = N.PAIRS[(hn.op, v2)]; a_, b_ = prs[int(rng.integers(0, len(prs)))]
            hb = N._copy(root); hh = hb
            for p_ in path:
                hh = getattr(hh, p_)
            hh.l.v, hh.r.v = a_, b_; hn.l.v, hn.r.v = a_, b_; bare = getattr(d, side); bare.v = int(rng.choice([x for x in range(N.P) if x != bare.v]))
            try:
                got = (N.value(hb), N.value(d)); break
            except KeyError:
                continue
        if got is None:
            continue
        hyb_v, don_v = got
        if hyb_v != N.value(root) and hyb_v != don_v:
            rec.append(it); don.append({"syms": N.render(d), "t": don_v, "root": d, "n": it["n"]}); hyb.append(hyb_v)
            hyb_it.append({"syms": N.render(hb), "t": hyb_v, "root": hb, "n": it["n"]})
    return rec, don, hyb, hyb_it


class Runner:
    def __init__(self, m, dev, bs):
        self.m, self.dev, self.bs = m, dev, bs
        amp = torch.bfloat16 if (dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8) else None
        self.ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
        self.amp = amp

    @torch.no_grad()
    def states(self, items, t):
        """S after t think passes from the zero state (nested170 inject_states), per chunk of bs -> list of chunk tensors."""
        out = []
        for i in range(0, len(items), self.bs):
            ch = items[i:i + self.bs]; x, msk = self.m.think_inputs(ch, self.dev)
            with self.ac():
                S = self.m.mem.zero_state(len(ch), self.dev)
                for _ in range(t):
                    S = self.m.write(S, x, msk)
            out.append(S)
        return out

    @torch.no_grad()
    def answers(self, items, K, override=None):
        """nested170.inject_answers: K passes from the zero state; override = (t, [chunk states]) replaces S after pass t; argmax over the 11 values."""
        out = []
        for ci, i in enumerate(range(0, len(items), self.bs)):
            ch = items[i:i + self.bs]; x, msk = self.m.think_inputs(ch, self.dev)
            with self.ac():
                S = self.m.mem.zero_state(len(ch), self.dev)
                for t in range(1, K + 1):
                    S = self.m.write(S, x, msk)
                    if override is not None and override[0] == t:
                        S = override[1][ci].to(S.dtype)
                lg = self.m.dec_logits(S, ch, self.dev)
            out.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=int)


def shares(got, own, don, hyb, mask):
    n_ = int(mask.sum())
    if not n_:
        return {"n": 0}
    g, o, d, h = got[mask], own[mask], don[mask], hyb[mask]
    return {"n": n_, "hybrid": float((g == h).mean()), "own": float((g == o).mean()), "donor": float((g == d).mean()), "other": float(((g != h) & (g != o) & (g != d)).mean()),
            "na": n_ < 100}


def run_serial(R, rec, don, hyb_v, hyb_it, ts, K, tag, rows):
    """clean answers once, then the swaps at each t -> {t: {all, clean, (by_n)}}; rows[tag_t*] = per-pair arrays."""
    own = np.array([N.value(it["root"]) for it in rec]); dv = np.array([it["t"] for it in don]); hv = np.array(hyb_v); nn = np.array([it["n"] for it in rec])
    c_rec, c_don, c_hyb = R.answers(rec, K), R.answers(don, K), R.answers(hyb_it, K); ccc = (c_rec == own) & (c_don == dv) & (c_hyb == hv); out = {}
    for t, elig in ts:
        Sd = R.states(don, t); got = R.answers(rec, K, override=(t, Sd)); e = np.array(elig, dtype=bool)
        blk = {"all": shares(got, own, dv, hv, e), "clean": shares(got, own, dv, hv, e & ccc),
               "clean_rec_don_only": shares(got, own, dv, hv, e & (c_rec == own) & (c_don == dv))}
        if len(set(nn.tolist())) > 1:
            blk["by_n"] = {str(n_): {"all": shares(got, own, dv, hv, e & (nn == n_)), "clean": shares(got, own, dv, hv, e & ccc & (nn == n_))} for n_ in sorted(set(nn.tolist()))}
        out[str(t)] = blk
        rows["%s_t%d" % (tag, t)] = np.stack([np.arange(len(rec)), nn, own, dv, hv, c_rec, c_don, c_hyb, got, e.astype(int)], 1)
    rows["%s_hash" % tag] = np.array([int(hashlib.sha1(" ".join(map(str, it["syms"])).encode()).hexdigest()[:12], 16) for it in rec], dtype=np.int64)
    out["clean_acc"] = {"recipient": float((c_rec == own).mean()), "donor": float((c_don == dv).mean()), "hybrid_item": float((c_hyb == hv).mean()), "n_ccc": int(ccc.sum()), "n": len(rec)}
    out["row_cols"] = ["pair", "n", "y_own", "y_donor", "y_hyb", "clean_rec", "clean_don", "clean_hyb", "swapped", "eligible"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--stage", default="P0", choices=["P0", "P1", "P2", "B"])
    ap.add_argument("--bs", type=int, default=250); ap.add_argument("--memcheck", action="store_true"); a = ap.parse_args()
    rd = os.path.join(ROOT, "boxes", "runs_fsq", a.run); kind, m, ra = H.load_model(rd, a.device); assert kind == "nested", kind
    DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = 0; DM.READ_CLAMP = None; K = int(ra["think_k"]); t0 = time.time(); R = Runner(m, a.device, a.bs)
    meta = {"card": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu", "dtype": DTYPE if R.amp is not None else "fp32", "bs_eval": a.bs,
            "torch": torch.__version__, "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
            if a.device.startswith("cuda") else None, "script_git": git_hash("scratch/dasused.py")}
    TE = N.test_set(int(ra.get("per_n") or 1000)); r_sp = np.random.default_rng(17300); SPOOL = []
    for _ in range(int(ra.get("serial_pool") or 2200)):
        r__, t__ = N.generate(r_sp, 8); s__ = N.render(r__); SPOOL.append({"n": 8, "syms": s__, "t": t__, "depth": N.depth(s__), "root": r__})
    if a.memcheck:                                                          # the largest shape: n = 8 rows at bs, K passes, decode
        torch.cuda.reset_peak_memory_stats(a.device); R.answers(SPOOL[:a.bs], K, override=(2, R.states(SPOOL[:a.bs], 2)))
        print(a.run, "memcheck peak %.2f GiB (bs %d, K %d)" % (torch.cuda.max_memory_allocated(a.device) / 2 ** 30, a.bs, K), flush=True); return
    fp = os.path.join(rd, "eval_dasused.json"); res = json.load(open(fp)) if os.path.exists(fp) else {}
    res.update({"run": a.run, "K": K, "add_norm": ra.get("add_norm") or None, "meta": meta}); rows = {}
    if a.stage == "P0":
        rec, don, hyb_v, hyb_it = serial_items_h(SPOOL, np.random.default_rng(17200)); r0, d0, h0 = N.serial_items(SPOOL, np.random.default_rng(17200))
        gate_si = len(r0) == len(rec) and all(x is y for x, y in zip(r0, rec)) and [d["syms"] for d in d0] == [d["syms"] for d in don] and list(h0) == list(hyb_v)
        ts_i = [(t, [True] * len(rec)) for t in (1, 2, 4, 6) if t < K]
        s_i = run_serial(R, rec, don, hyb_v, hyb_it, ts_i, K, "serial_i", rows)
        ev = json.load(open(os.path.join(rd, "eval.json")))["s170"]["serial"]; g_ = {}
        for t in (str(x) for x, _ in ts_i):
            if t in ev:
                a_ = s_i[t]["all"]; g_[t] = {k_: (a_[{"recipient": "own"}.get(k_, k_)], ev[t][k_]) for k_ in ("hybrid", "donor", "recipient")}
        gate_s170 = {"pairs": g_, "max_abs_diff": max(abs(x - y) for d_ in g_.values() for x, y in d_.values()) if g_ else None}
        gate_s170["pass"] = gate_s170["max_abs_diff"] is not None and gate_s170["max_abs_diff"] <= 0.02
        print(a.run, "(i) n_pairs", len(rec), "SI", gate_si, "S170", gate_s170["max_abs_diff"], "clean", s_i["clean_acc"], flush=True)
        ext = [it for it in TE if 3 <= it["n"] <= 8]; rec2, don2, hv2, hi2 = serial_items_h(ext, np.random.default_rng(20371))
        hroot = [height(it["root"]) for it in rec2]; ts_ii = [(t, [h_ >= 2 * t + 1 for h_ in hroot]) for t in (1, 2)]
        s_ii = run_serial(R, rec2, don2, hv2, hi2, ts_ii, K, "serial_ii", rows)
        print(a.run, "(ii) n_pairs", len(rec2), {t: {"all": s_ii[t]["all"], "clean": s_ii[t]["clean"]} for t in ("1", "2")}, flush=True)
        res["serial_cond"] = {"s170": s_i, "ext": s_ii, "gates": {"SI_reproduces_serial_items": bool(gate_si), "S170": gate_s170},
                              "def": "notes 203-2 / 203-2a: whole-state swap of the donor's S_t, K passes; clean = recipient, donor and hybrid item correct without a swap; "
                                     "(ii) eligibility per t: bare root leaf, h != root, height(root) >= 2t+1, hybrid not in {own, donor}; n < 100 -> na",
                              "pool_i": "SPOOL n 8 rng 17300 (2,200), serial_items rng 17200", "pool_ii": "test items n 3-8 (all), rng 20371", **meta, "sec": round(time.time() - t0), "done": True}
    json.dump(res, open(fp, "w", newline="\n"), indent=1)
    rp = os.path.join(rd, "eval_dasused_rows.npz"); old = dict(np.load(rp)) if os.path.exists(rp) else {}; old.update(rows); np.savez_compressed(rp, **old)
    print(a.run, "P0 done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
