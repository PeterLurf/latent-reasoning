"""THEORY §63 (frontier F4): a behavioural round-trip translator for silent thoughts.

For one frozen toy reasoner R, learn
  D : thought h (d) -> a hard set of node ids (the text; multi-label over the graph's present ids, ST gradient)
  E : (set, step i) -> thought
through R's behaviour only (R is never trained):
  round trip : run R with thought i replaced by E(D(h_i), i); its final answers must equal R's unpatched answers
  edit       : add one node u (unreachable from START) to the decoded set; R's answers must become
               (unpatched answers) OR (reachable from u), by BFS on the graph: the IIT-style edit loss
Training graphs: long chains (latent.length_gen, widths 2,4,6, K' = 12) and ordinary training graphs (K = R's K), half
each; i ~ U{1..6}. The set's meaning is never supervised.
Evaluation on the long-chain probe (rollout.probe(512), seed 4247, K' = 12), at i = 2 (primary) and 3:
  RT        : agreement of the hard-set round-trip run with the unpatched run on all 24 chain/look-alike queries
  IIA       : §34's edit test on the text. u = the look-alike of the deepest reachable-chain node j in the decoded set;
              ins = both look-alikes at j+2, j+3 answered reachable after the edit; keep = all reachable-chain answers
              unchanged; IIA = mean(ins & keep) over graphs whose decoded set contains a chain node with j <= 8
  controls  : RT_mean (h_i <- the probe-mean thought at step i), RT_mismatch (decode another graph's h_i)
  semantics : precision / recall of the decoded set against the true set reached within i hops (evaluation only)
    python theory/f4_translator.py --model runs/p30_s102 --seed 1 --out theory/runs/f4/p30_s102_t1 [--device cuda:0]
    python theory/f4_translator.py --swap theory/runs/f4/p30_s102_t1 theory/runs/f4/p30_s102_t2   # evaluator swap
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_arch import load_arch  # noqa: E402
from latent.data import make_batch  # noqa: E402
from latent.graphs import V, sample_graphs  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from rollout import load, probe  # noqa: E402

KMAX = 12


class Translator(nn.Module):
    def __init__(self, d, hid=512, step=True):
        super().__init__()
        self.step = step                                       # False: the encoder sees the set only (no age channel)
        self.dec = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, V))
        self.enc = nn.Sequential(nn.Linear(V + KMAX + 1, hid), nn.GELU(), nn.Linear(hid, d))
        self.enc_lin = nn.Linear(V, d)

    def decode(self, h, present):
        z = self.dec(h).masked_fill(~present, -30.0)
        p = torch.sigmoid(z)
        return (p > 0.5).float() + p - p.detach()            # hard set forward, sigmoid straight-through backward

    def encode(self, s, i):
        oh = F.one_hot(torch.full((s.shape[0],), i, device=s.device), KMAX + 1).float() * (1.0 if self.step else 0.0)
        return self.enc(torch.cat([s, oh], 1)) + self.enc_lin(s)


def present_mask(src, dst):
    m = torch.zeros(src.shape[0], V, dtype=torch.bool, device=src.device)
    m.scatter_(1, src, True); m.scatter_(1, dst, True)
    return m


def reach_from(src, dst, u, steps=KMAX + 2):
    """(B, V) bool: nodes reachable from u (u excluded) by true BFS; numpy arrays in."""
    B = len(u); r = np.zeros((B, V), bool); r[np.arange(B), u] = True
    for _ in range(steps):
        nxt = r.copy()
        hit = r[np.arange(B)[:, None], src]
        np.logical_or.at(nxt, (np.repeat(np.arange(B), src.shape[1])[hit.ravel()], dst.ravel()[hit.ravel()]), True)
        r = nxt
    r[np.arange(B), u] = False
    return r


def load_model(path, dev):
    m, c = (load_arch if path.startswith("theory/") else load)(path, dev)
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, c


def run_patched(m, src, dst, start, Kp, Q, i, h_new):
    return m.run(src, dst, start, Kp, qnodes=Q, patch=(lambda j, h: h_new if j == i else h))["logit"]


def batch(rng, kind, n, K, dev):
    T = lambda a: torch.as_tensor(a, device=dev)
    if kind == "long":
        g, br = long_chain_graphs(rng, n, [2, 4, 6]); bi = np.arange(n)
        q = np.concatenate([g.nodes[bi, br[k]] for k in ("A", "B", "C", "D")], 1)
        return g, T(g.src), T(g.dst), T(g.start), T(q).long(), KMAX
    g = sample_graphs(rng, n, split="train")
    b = make_batch(rng, g, 16, dev)
    return g, b["src"], b["dst"], b["start"], b["q"], K


def train(a):
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    m, c = load_model(a.model, dev); K = c["K"]
    tr = Translator(m.d if hasattr(m, "d") else 256, step=not a.no_step).to(dev)
    opt = torch.optim.Adam(tr.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time()
    for step in range(1, a.steps + 1):
        g, src, dst, start, Q, Kp = batch(rng, "long" if step % 2 else "train", a.bs, K, dev)
        i = int(rng.integers(1, 6))                    # thought i feeds step i + 1, so i <= K - 1 (K = 6 batches)
        with torch.no_grad():
            o = m.run(src, dst, start, Kp, qnodes=Q, keep=True)
            y0 = (o["logit"] > 0).float(); h = o["thoughts"][:, i]
        pres = present_mask(src, dst)
        s = tr.decode(h, pres)
        l_rt = F.binary_cross_entropy_with_logits(run_patched(m, src, dst, start, Kp, Q, i, tr.encode(s, i)), y0)
        # edit: add one node u unreachable from START and not in the hard set
        dist = g.dist(); s_h = (s > 0.5).cpu().numpy()
        cand = (dist == 99) & ~s_h
        u = np.array([rng.choice(np.flatnonzero(row)) for row in cand])
        ru = torch.as_tensor(reach_from(g.src, g.dst, u), device=dev)
        y_edit = torch.maximum(y0, torch.gather(ru.float(), 1, Q))
        s_e = s.clone(); s_e[torch.arange(len(u)), torch.as_tensor(u, device=dev)] = 1.0
        l_ed = F.binary_cross_entropy_with_logits(run_patched(m, src, dst, start, Kp, Q, i, tr.encode(s_e, i)), y_edit)
        loss = l_rt + a.w_edit * l_ed
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == a.steps:
            log.write(json.dumps({"step": step, "l_rt": l_rt.item(), "l_edit": l_ed.item(), "set_size": float(s_h.sum(1).mean()),
                                  "sec": round(time.time() - t0)}) + "\n"); log.flush()
    torch.save({"translator": tr.state_dict(), "args": vars(a)}, os.path.join(a.out, "translator.pt"))
    res = evaluate(m, K, tr, tr, dev)
    res["args"] = vars(a)
    json.dump(res, open(os.path.join(a.out, "eval_f4.json"), "w", newline="\n"), indent=1)
    print(a.out, json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)


@torch.no_grad()
def evaluate(m, K, trD, trE, dev, steps=(2, 3)):
    g, br, _ = probe(512); n = len(g); bi = np.arange(n)
    T = lambda x: torch.as_tensor(x, device=dev)
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    Q = T(np.concatenate([reach, decoy], 1)).long()
    src, dst, start = T(g.src), T(g.dst), T(g.start)
    o = m.run(src, dst, start, KMAX, qnodes=Q, keep=True)
    y0 = o["logit"] > 0
    g8, _, _ = probe(512, seed=8)
    mu = m.run(T(g8.src), T(g8.dst), T(g8.start), KMAX, keep=True)["thoughts"].mean(0)
    pres = present_mask(src, dst)
    dist = g.dist()
    out = {}
    for i in steps:
        h = o["thoughts"][:, i]
        s = (trD.decode(h, pres) > 0.5).float()
        rt = run_patched(m, src, dst, start, KMAX, Q, i, trE.encode(s, i)) > 0
        r = {"RT": float((rt == y0).float().mean())}
        r["RT_mean"] = float(((run_patched(m, src, dst, start, KMAX, Q, i, mu[i].expand_as(h)) > 0) == y0).float().mean())
        s_mis = (trD.decode(torch.roll(h, 1, 0), torch.roll(pres, 1, 0)) > 0.5).float()
        r["RT_mismatch"] = float(((run_patched(m, src, dst, start, KMAX, Q, i, trE.encode(s_mis, i)) > 0) == y0).float().mean())
        # input-only control: the encoder fed the true set reached within i hops (BFS on the input), same step index
        s_in = T(((dist >= 1) & (dist <= i)).astype(np.float32))
        y_in = run_patched(m, src, dst, start, KMAX, Q, i, trE.encode(s_in, i)) > 0
        r["RT_input"] = float((y_in == y0).float().mean())
        # error reproduction: on queries the unpatched model gets wrong, does the round trip reproduce its answer?
        truth_q = T(np.r_[np.ones(12), np.zeros(12)][None].repeat(n, 0)) > 0.5
        wrong = y0 != truth_q
        r["model_err_rate"] = float(wrong.float().mean())
        r["err_repro"] = float((rt == y0)[wrong].float().mean()) if wrong.any() else float("nan")
        r["err_repro_input"] = float((y_in == y0)[wrong].float().mean()) if wrong.any() else float("nan")
        # semantics: decoded set vs reached within i hops (start excluded)
        truth = T((dist >= 1) & (dist <= i)); sb = s.bool()
        r["prec"] = float((sb & truth).sum() / max(sb.sum().item(), 1)); r["rec"] = float((sb & truth).sum() / max(truth.sum().item(), 1))
        r["set_size"] = float(sb.sum(1).float().mean())
        # §34-style edit on the text
        sn = s.cpu().numpy() > 0.5
        onchain = sn[bi[:, None], reach]                                   # (n, 12)
        has = onchain.any(1)
        j = np.where(has, 11 - np.argmax(onchain[:, ::-1], 1), -1)
        elig = has & (j <= 8)
        jj = np.clip(j, 0, 8)
        u = decoy[bi, jj]
        s_e = s.clone(); s_e[torch.arange(n), T(u).long()] = 1.0
        ye = (run_patched(m, src, dst, start, KMAX, Q, i, trE.encode(s_e, i)) > 0).cpu().numpy()
        y0n = y0.cpu().numpy()
        ins = ye[bi, 12 + jj + 2] & ye[bi, 12 + np.minimum(jj + 3, 11)]
        keep = (ye[:, :12] == y0n[:, :12]).all(1)
        r["eligible"] = float(elig.mean())
        r["IIA"] = float((ins & keep)[elig].mean()) if elig.any() else float("nan")
        r["ins"] = float(ins[elig].mean()) if elig.any() else float("nan"); r["keep"] = float(keep[elig].mean()) if elig.any() else float("nan")
        # claim-flip audit (additions): add a random node unreachable from START; its descendants within the remaining
        # K' - i steps should turn on among the 24 probe queries, and no other probe answer should change
        rng = np.random.default_rng(100 + i)
        cand = (dist == 99) & ~sn
        ua = np.array([rng.choice(np.flatnonzero(row)) for row in cand])
        desc = reach_from(g.src, g.dst, ua)
        dq = desc[bi[:, None], np.concatenate([reach, decoy], 1)]               # (n, 24) predicted new positives
        s_a = s.clone(); s_a[torch.arange(n), T(ua).long()] = 1.0
        ya = (run_patched(m, src, dst, start, KMAX, Q, i, trE.encode(s_a, i)) > 0).cpu().numpy()
        r["flip_on"] = float(ya[dq].mean()) if dq.any() else float("nan")
        r["flip_collateral"] = float((ya != y0n)[~dq].mean())                   # any change outside the predicted set
        out[f"i{i}"] = r
    return out


def swap(paths, dev):
    """Evaluator swap: decode with translator A, encode with translator B (and vice versa), same frozen model."""
    cks = [torch.load(os.path.join(p, "translator.pt"), map_location=dev, weights_only=False) for p in paths]
    m, c = load_model(cks[0]["args"]["model"], dev)
    trs = []
    for ck in cks:
        tr = Translator(256, step=not ck["args"].get("no_step", False)).to(dev); tr.load_state_dict(ck["translator"]); tr.eval(); trs.append(tr)
    res = {"AB": evaluate(m, c["K"], trs[0], trs[1], dev), "BA": evaluate(m, c["K"], trs[1], trs[0], dev)}
    out = os.path.join(paths[0], "eval_f4_swap.json")
    json.dump(res, open(out, "w", newline="\n"), indent=1)
    print(out, json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--out"); ap.add_argument("--swap", nargs=2)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_edit", type=float, default=1.0); ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_step", action="store_true", help="encoder without the step index (Literature's age-channel arm)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
    if a.swap:
        swap(a.swap, a.device)
    else:
        train(a)


if __name__ == "__main__":
    main()
