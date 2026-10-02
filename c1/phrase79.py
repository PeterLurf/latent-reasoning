"""THEORY §79 (75ec955; phrasings c410cfe theory/s79_phrasings.py): a latent automaton whose operations arrive as
natural-language paraphrases; transfer to held-out phrasings.
Task: S3 boxes, all 6 elements as ops (OPS of s79_phrasings, semantics `apply`, unit-tested there); start = the
identity arrangement (code 0); answer = the final arrangement (6-way), the only label. Lengths uniform in 1-8 per batch
(all rows of a batch share the length): amendment 79-2 replaces
uniform 1-8 with §73's std curriculum (4 equal stages, lengths {1}, {1,2}, {1,2,4}, {1,2,4,8}); --curriculum uniform
reproduces pilot 1. Every op occurrence is rendered in a training family drawn
uniformly (swap order random), so each training PHRASING (op x family x swap order) is its own transition symbol:
the exact marginal runs over the (phrasing, M, M) table. 8 families -> 72 phrasings; the 2/4-family sweep uses T1-T2 /
T1-T4.
Cell (§73 recipe): M 12, transition call [code_emb, phrasing tokens, " step"]; h_code at position 0, h_op at the step
marker; next = W ReLU((U std(h_code)) * (V std(h_op))); decoder [code_emb, " answer"] -> ans_head(std(h_dec)); std =
per-coordinate standardisation with statistics of the TRAINING table (all training phrasings x M; M decoder calls),
live during training, frozen after training and applied unchanged to held-out phrasings (sigma-floor variant
sigma' = max(sigma, median(sigma)/10)). tau 4 -> 1 over the first half, AdamW lr 1e-4, 8,000 updates, bs 512.
Arms: pfstd (frozen pretrained GPT-2), rfstd (frozen random-init GPT-2), bow (no transformer: h_code = code, h_op =
code + sum of pretrained wte over the phrasing tokens, h_dec = code), tab (TAB-phrase: a free (phrasing, M, M) logit
table + decoder table; unseen phrasings are undefined -> agreement 0, certificate false by construction).
Eval (eval.json): reachable codes R (BFS from code 0 under every training phrasing); reference next code ref[c, op] =
the majority next code over op's training phrasings at c (their consistency reported); training certificate (product
automaton with every training phrasing as an edge) and accuracy N 8-128 with random training phrasings. Per held-out
family H1-H4 and controls POS / NEG, each with frozen sigma and with the sigma-floor: paraphrase agreement over c in R x
the family's phrasings (Th_H[c, p] == ref[c, op(p)]); family certificate (the family's phrasings as edges); accuracy
N 8-128 with every op in the family's phrasing; NEG ("copy box i into box j") is scored against swap(i, j) (verb
sensitivity: lower is better) on swaps only. Minimal pairs: T6 (training) and H4 (held-out) rotations, the share of
c in R where rotL's and rotR's phrasings both map to their reference next code.
    python -m c1.phrase79 --arm pfstd --seed 2081 [--families 8] --out c1/runs/p79_pfstd_s2081 --device cuda:0
"""
import argparse
import collections
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import tiktoken
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
from boxes.core import GPT2KV  # noqa: E402
from c1.core import gpt2_random_sd  # noqa: E402
import s79_phrasings as SP  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
STATES = sorted(itertools.permutations("ABC"))                        # index 0 = ("A","B","C") = the identity arrangement
NEXT = np.array([[STATES.index(SP.apply(s, o)) for o in SP.OPS] for s in STATES])   # NEXT[state, op]
STEP, ANS = ENC.encode(" step")[0], ENC.encode(" answer")[0]
M = 12; DEPTHS = (8, 16, 32, 64, 128)


class _Fixed:
    """random.Random stand-in that returns a fixed swap order: < 0.5 keeps (i, j), else swaps it."""
    def __init__(self, flip):
        self.flip = flip

    def random(self):
        return 0.9 if self.flip else 0.1


def phrasings(family):
    """[(op index, text)] for every rendering of a family (both swap orders)."""
    out = []
    for k, o in enumerate(SP.OPS):
        if family == "NEG" and o not in SP.PAIRS:
            continue
        for flip in ((False, True) if o in SP.PAIRS else (False,)):
            out.append((k, SP.render(family, o, _Fixed(flip))))
    return out


class Cell(nn.Module):
    def __init__(self, arm, seed, train_fams, hid=256):
        super().__init__()
        self.arm = arm
        self.train_ph = [p for f in train_fams for p in phrasings(f)]        # training symbols
        P = len(self.train_ph)
        if arm == "tab":
            self.T = nn.Parameter(0.01 * torch.randn(P, M, M)); self.D = nn.Parameter(0.01 * torch.randn(M, 6)); self.g = None
            return
        if arm in ("pfstd", "bow"):
            sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
        else:
            sd = gpt2_random_sd(seed)
        if arm == "bow":
            self.register_buffer("wte", sd["wte.weight"].float().clone()); self.g = None
        else:
            self.g = GPT2KV(sd); self.g.requires_grad_(False)
        self.code = nn.Parameter(0.14 * torch.randn(M, 768))
        self.U, self.V, self.W = nn.Linear(768, hid), nn.Linear(768, hid), nn.Linear(hid, M)
        self.ans_head = nn.Linear(768, 6)
        self.stats = None; self.sigma_floor = False

    def emb(self, ids):
        return self.g.embed(ids) if self.g is not None else self.wte[ids]

    def raw(self, texts):
        """raw (h_code, h_op) for every (phrasing, code) pair, rows ordered phrasing-major: row = p * M + c."""
        dev = self.code.device; toks = [ENC.encode(t) for t in texts]
        hc = torch.zeros(len(texts) * M, 768, device=dev); ho = torch.zeros_like(hc)
        groups = collections.defaultdict(list)
        for p, t in enumerate(toks):
            groups[len(t)].append(p)
        codes = torch.arange(M, device=dev)
        for L, ps in groups.items():
            tt = torch.tensor([toks[p] for p in ps], device=dev)              # (G, L)
            rows = torch.tensor([p * M + c for p in ps for c in range(M)], device=dev)
            if self.g is None:                                            # bow
                e = self.emb(tt).sum(1)                                   # (G, D)
                hc[rows] = self.code.repeat(len(ps), 1)
                ho[rows] = self.code.repeat(len(ps), 1) + e.repeat_interleave(M, 0)
                continue
            x = torch.cat([self.code.repeat(len(ps), 1)[:, None], self.emb(tt).repeat_interleave(M, 0),
                           self.emb(torch.full((len(ps) * M, 1), STEP, device=dev))], 1)
            h, _ = self.g(x)
            hc[rows] = h[:, 0]; ho[rows] = h[:, -1]
        return hc, ho

    def raw_dec(self):
        if self.g is None:
            return self.code
        dev = self.code.device
        x = torch.stack([self.code, self.emb(torch.full((M,), ANS, device=dev))], 1)
        h, _ = self.g(x)
        return h[:, -1]

    def _sig(self, s):
        return torch.maximum(s, s.median() / 10) if self.sigma_floor else s

    def compute_stats(self):
        with torch.no_grad():
            hc, ho = self.raw([t for _, t in self.train_ph]); hd = self.raw_dec()
            self.stats = tuple((t.mean(0), t.std(0) + 1e-5) for t in (hc, ho, hd))

    def tables(self, tau):
        if self.arm == "tab":
            return (self.T / tau).log_softmax(-1), self.D.log_softmax(-1)
        hc, ho = self.raw([t for _, t in self.train_ph]); hd = self.raw_dec()
        (mc, sc), (mo, so), (md, sd) = [(t.mean(0), t.std(0) + 1e-5) for t in (hc, ho, hd)]
        lt = self.W(torch.relu(self.U((hc - mc) / sc) * self.V((ho - mo) / so))).view(-1, M, M)
        return (lt / tau).log_softmax(-1), self.ans_head((hd - md) / sd).log_softmax(-1)

    @torch.no_grad()
    def hard(self, texts):
        """(Th (P, M) argmax next code per phrasing, Dh (M,)) with the frozen statistics."""
        if self.arm == "tab":
            return None, self.D.argmax(-1).cpu().numpy()
        hc, ho = self.raw(texts); hd = self.raw_dec(); (mc, sc), (mo, so), (md, sd) = self.stats
        th = self.W(torch.relu(self.U((hc - mc) / self._sig(sc)) * self.V((ho - mo) / self._sig(so)))).argmax(-1).view(-1, M)
        dh = self.ans_head((hd - md) / self._sig(sd)).argmax(-1)
        return th.cpu().numpy(), dh.cpu().numpy()


def marginal_loss(logT, logD, sym, y):
    B, N = sym.shape; la = torch.full((B, M), -math.inf, device=logT.device); la[:, 0] = 0.0
    for t in range(N):
        la = torch.logsumexp(la[:, :, None] + logT[sym[:, t]], 1)
    return -torch.logsumexp(la + logD[:, y].T, 1).mean()


def certificate(edges, Dh):
    """edges: list of (op index, Th column (M,)) -- product automaton from (code 0, identity state 0)."""
    seen, fr, bad = set(), [(0, 0)], 0
    while fr:
        c, s = fr.pop()
        for o, col in edges:
            nx = (int(col[c]), int(NEXT[s, o]))
            if nx not in seen:
                seen.add(nx); fr.append(nx); bad += int(Dh[nx[0]] != nx[1])
    return {"certified": bad == 0, "bad_pairs": bad, "reachable_pairs": len(seen)}


def roll_acc(edges, Dh, seed, n=2000):
    """hard accuracy at N 8-128: each op occurrence uses a random phrasing of that op from `edges`."""
    rng = np.random.default_rng(seed); by_op = collections.defaultdict(list)
    for o, col in edges:
        by_op[o].append(col)
    ops_avail = sorted(by_op); out = {}
    for N in DEPTHS:
        ops = rng.choice(ops_avail, (n, N)); c = np.zeros(n, int); s = np.zeros(n, int)
        for t in range(N):
            cols = np.stack([by_op[o][rng.integers(len(by_op[o]))] for o in ops[:, t]])
            c = cols[np.arange(n), c]; s = NEXT[s, ops[:, t]]
        out[str(N)] = float((Dh[c] == s).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pfstd", "rfstd", "bow", "tab"])
    ap.add_argument("--families", type=int, default=8, choices=[2, 4, 8]); ap.add_argument("--seed", type=int, default=2081)
    ap.add_argument("--updates", type=int, default=8000); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--tau0", type=float, default=4.0)
    ap.add_argument("--curriculum", default="std", choices=["std", "uniform"], help="79-2: std {1}->{1,2}->{1,2,4}->{1,2,4,8}")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = a.device
    fams = list(SP.TRAIN)[: a.families]
    m = Cell(a.arm, a.seed, fams).to(dev); P = len(m.train_ph)
    sym_by_op = collections.defaultdict(list)
    for p, (o, _) in enumerate(m.train_ph):
        sym_by_op[o].append(p)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**vars(a), "train_families": fams, "train_phrasings": P}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    for u in range(1, a.updates + 1):
        if a.curriculum == "std":                                       # amendment 79-2: §73's std curriculum
            L = int(rng.choice((1, 2, 4, 8)[: min(3, (u - 1) * 4 // a.updates) + 1]))
        else:                                                             # the original reading 2 (pilot 1 only)
            L = int(rng.integers(1, 9))
        ops = rng.integers(0, 6, (a.bs, L))
        sym = np.vectorize(lambda o: sym_by_op[o][rng.integers(len(sym_by_op[o]))])(ops)
        st = np.zeros(a.bs, int)
        for t in range(L):
            st = NEXT[st, ops[:, t]]
        tau = 1.0 + (a.tau0 - 1.0) * max(0.0, 1.0 - (u - 1) / (0.5 * a.updates))
        logT, logD = m.tables(tau)
        loss = marginal_loss(logT, logD, torch.tensor(sym, device=dev), torch.tensor(st, device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
        if u % 200 == 0 or u == a.updates:
            log.write(json.dumps({"update": u, "L": L, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}) + "\n"); log.flush(); hist = []
    m.eval(); res = {"arm": a.arm, "seed": a.seed, "families": a.families, "train_sec": round(time.time() - t0)}
    if a.arm == "tab":
        Tt = m.T.argmax(-1).cpu().numpy(); Dh = m.D.argmax(-1).cpu().numpy()
    else:
        m.compute_stats(); Tt, Dh = m.hard([t for _, t in m.train_ph])            # frozen statistics from here on
    tr_edges = [(o, Tt[p]) for p, (o, _) in enumerate(m.train_ph)]
    seen, fr = {0}, [0]
    while fr:
        c = fr.pop()
        for _, col in tr_edges:
            if int(col[c]) not in seen:
                seen.add(int(col[c])); fr.append(int(col[c]))
    R = sorted(seen); ref = np.zeros((M, 6), int); consistent = []
    for c in R:
        for o in range(6):
            nx = [int(Tt[p][c]) for p in sym_by_op[o]]
            ref[c, o] = collections.Counter(nx).most_common(1)[0][0]; consistent.append(len(set(nx)) == 1)
    res["train"] = {"certificate": certificate(tr_edges, Dh), "acc": roll_acc(tr_edges, Dh, 7900), "reachable_codes": R,
                    "phrasing_consistency": float(np.mean(consistent))}
    torch.save({k: v for k, v in m.state_dict().items() if not k.startswith(("g.", "wte"))}, os.path.join(a.out, "trainable.pt"))
    res["eval"] = {}
    evals = list(SP.HELDOUT) + ["POS", "NEG", "T6"]
    for floor in (False, True):
        if a.arm == "tab" and floor:
            break
        m.sigma_floor = floor; key = "sigma_floor" if floor else "frozen_sigma"
        for f in evals:
            ph = phrasings(f)
            if a.arm == "tab":
                if f == "T6" and "T6" in fams:
                    idx = [m.train_ph.index(p) for p in ph]; Th = Tt[idx]
                else:
                    res["eval"].setdefault(f, {})[key] = {"agreement": 0.0, "certificate": {"certified": False}, "acc": None,
                                                          "note": "undefined for an unseen phrasing (TAB-phrase): 0 by construction"}
                    continue
            else:
                Th, _ = m.hard([t for _, t in ph])
            agree = float(np.mean([Th[p][c] == ref[c, o] for p, (o, _) in enumerate(ph) for c in R]))
            edges = [(o, Th[p]) for p, (o, _) in enumerate(ph)]
            r = {"agreement": agree, "chance": 1 / max(1, len(R)), "n_phrasings": len(ph)}
            if f != "NEG":
                r["certificate"] = certificate(edges, Dh); r["acc"] = roll_acc(edges, Dh, 7901)
            if f in ("T6", "H4"):
                pl = [p for p, (o, _) in enumerate(ph) if SP.OPS[o] == "rotL"][0]; pr = [p for p, (o, _) in enumerate(ph) if SP.OPS[o] == "rotR"][0]
                kL, kR = SP.OPS.index("rotL"), SP.OPS.index("rotR")
                r["minimal_pair_acc"] = float(np.mean([Th[pl][c] == ref[c, kL] and Th[pr][c] == ref[c, kR] for c in R]))
            res["eval"].setdefault(f, {})[key] = r
    m.sigma_floor = False
    if a.arm != "tab":
        q = lambda t: [float(t.min()), float(t.median()), float(t.max())]      # noqa: E731
        res["sigma_min_med_max"] = {k: q(st[1]) for k, st in zip(("code", "op", "dec"), m.stats)}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({"train_cert": res["train"]["certificate"]["certified"], "acc128": res["train"]["acc"]["128"],
                              "agree": {f: round(v["frozen_sigma"]["agreement"], 3) for f, v in res["eval"].items()}}), flush=True)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
