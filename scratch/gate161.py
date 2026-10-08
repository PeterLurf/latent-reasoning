"""THEORY §161 SHAPE-MIX pre-launch gates (56aa9af0), CPU, on the real training batches (main()'s batch(), captured at MultDelta.chain):
  (1) labels / schedule for n != 4: schedule(4, n) emits at each column's last pair and the flush (4 + n emissions); the emitted digits of
      accumulators() are a x b's digits LSB first (n = 2..8); every captured --shape_mix batch has n in the list, labels at the emission steps
      equal to those digits, and the full chain (flush included) when the curriculum draws c = 2n
  (2) both flags off: the captured batches (items, labels, trace mask, operands, every step's tokens) and the step losses bitwise equal to
      the reference module's (MULTPAIR_REF, default boxes.multpair = HEAD's code) under §155 R's arguments
  (3) the index token: with --col_index_offset 12 every segment's last token is CIDX[c + r - 1] (c from schedule(), the flush = last column + 1),
      r in 0..12 per chain; at eval (m.eval()) r = 0
    MULTPAIR_MOD=boxes._multpair161_tmp python scratch/gate161.py   -> boxes/runs_fsq/s161_gates/gate161.json (exit 1 on a failed gate)"""
import importlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

MP = importlib.import_module(os.environ.get("MULTPAIR_MOD", "boxes.multpair"))
from boxes.multcol import DTOK, gen_operands  # noqa: E402


class Stop(Exception):
    pass


def capture(M_, extra, n_steps, seed=1):
    """run M_.main() on CPU with §155 R's arguments (+ extra) for n_steps training batches; returns the captured batches."""
    got = []; orig = M_.MultDelta.chain

    def chain(self, start_emb, items, labels=None, trace_mask=None, record=False, override=None):
        if labels is None or not self.training:
            return orig(self, start_emb, items, labels, trace_mask, record, override)
        B = items.shape[0]; toks = [self.tokens(int(items[0, t]), B, items.device).cpu().numpy() for t in range(items.shape[1])]
        out = orig(self, start_emb, items, labels, trace_mask, record, override)
        got.append({"items": items.cpu().numpy(), "labels": np.asarray(labels), "tm": trace_mask.cpu().numpy(), "adig": self.adig.cpu().numpy(), "bdig": self.bdig.cpu().numpy(),
                    "pairs": np.asarray(self.pairs), "toks": toks, "r": None if getattr(self, "cidx_r", None) is None else self.cidx_r.cpu().numpy(), "loss": float(out[0].detach())})
        if len(got) >= n_steps:
            raise Stop()
        return out
    M_.MultDelta.chain = chain; d = tempfile.mkdtemp(prefix="g161_")
    sys.argv = ["multpair", "--family", "mult_pair", "--task", "4x4", "--seed", str(seed), "--carrier", "deltamem", "--mem_write", "top", "--bs", "8", "--steps", "40",
                "--sched_scale", "0.002", "--train_n", "3000", "--test_n", "16", "--device", "cpu", "--out", d] + extra
    try:
        M_.main()
    except Stop:
        pass
    finally:
        M_.MultDelta.chain = orig
    return got


def digits_lsb(v, k):
    return np.stack([(v // 10 ** i) % 10 for i in range(k)], 1)


def main():
    torch.set_num_threads(3); out = {"module": MP.__name__}; ok = True
    # (1a) the schedule / labels on random operands
    rng = np.random.default_rng(0); r1 = {}
    for nb in range(2, 9):
        A, Bd = gen_operands(rng, 64, 4), gen_operands(rng, 64, nb); acc, dig, _, pairs, col, emit = MP.accumulators(A, Bd)
        a_ = sum(A[:, i].astype(object) * 10 ** i for i in range(4)); b_ = sum(Bd[:, j].astype(object) * 10 ** j for j in range(nb)); P = digits_lsb(np.array(a_ * b_, dtype=object), 4 + nb).astype(np.int64)
        last = [t for t in range(len(pairs)) if t == len(pairs) - 1 or col[t + 1] != col[t]]
        r1[nb] = {"n_emit": int(emit.sum()), "emit_ok": list(np.nonzero(emit)[0]) == last, "digits_exact": bool((dig[:, emit] == P).all())}
    out["g1a"] = r1; out["g1a_pass"] = all(v["n_emit"] == 4 + k and v["emit_ok"] and v["digits_exact"] for k, v in r1.items()); ok &= out["g1a_pass"]
    # (1b) the captured --shape_mix batches
    gb = capture(MP, ["--shape_mix", "2,3,4,5"], 30); lut = {t: i for i, t in enumerate(DTOK)}; seen = set(); bad = []; full = 0
    for k, b in enumerate(gb):
        A = np.vectorize(lut.get)(b["adig"]); Bd = np.vectorize(lut.get)(b["bdig"]); nb = Bd.shape[1]; seen.add(nb)
        _, dig, _, pairs, col, emit = MP.accumulators(A, Bd); T = b["items"].shape[1]
        if not (nb in (2, 3, 4, 5) and (b["tm"][0] == emit[:T]).all() and (b["labels"][:, emit[:T]] == dig[:, :T][:, emit[:T]]).all() and (b["pairs"] == pairs).all()):
            bad.append(k)
        full += int(T == len(pairs))
    out["g1b"] = {"batches": len(gb), "n_seen": sorted(seen), "bad": bad, "full_chains": full}
    out["g1b_pass"] = not bad and len(seen) >= 3; ok &= out["g1b_pass"]
    # (2) flags off vs the reference module
    REF = importlib.import_module(os.environ.get("MULTPAIR_REF", "boxes.multpair"))
    torch.use_deterministic_algorithms(True, warn_only=True)                # the wte embedding backward is nondeterministic on CPU otherwise
    ga = capture(MP, [], 12); gr = capture(REF, [], 12); same = len(ga) == len(gr); torch.use_deterministic_algorithms(False)
    for x, y in zip(ga, gr):
        same &= all(np.array_equal(x[k], y[k]) for k in ("items", "labels", "tm", "adig", "bdig")) and all(np.array_equal(p_, q_) for p_, q_ in zip(x["toks"], y["toks"])) and x["loss"] == y["loss"]
    out["g2"] = {"batches": [len(ga), len(gr)], "losses": [[x["loss"] for x in ga], [y["loss"] for y in gr]], "ref": REF.__name__}
    out["g2_pass"] = bool(same) and REF.__name__ != MP.__name__; ok &= out["g2_pass"]
    # (3) the index tokens
    gi = capture(MP, ["--shape_mix", "2,3,4,5", "--col_index_offset", "12"], 20); bad3 = []; rs = []
    for k, b in enumerate(gi):
        na, nb = b["adig"].shape[1], b["bdig"].shape[1]; rs.extend(b["r"].tolist())
        for t in range(b["items"].shape[1]):
            i, j = b["pairs"][int(b["items"][0, t])]; c = na + nb if i < 0 else i + j + 1
            if not (b["toks"][t][:, -1] == np.array(MP.CIDX)[c + b["r"] - 1]).all():
                bad3.append((k, t))
    m = MP.MultDelta.__new__(MP.MultDelta); torch.nn.Module.__init__(m); m.cidx_off = 12; m.cidx_r = torch.full((4,), 7); m.eval()
    MP.set_ops(m, gen_operands(rng, 4, 4), gen_operands(rng, 4, 8), "cpu"); ev = m.tokens(len(m.pairs) - 1, 4, "cpu")[:, -1]
    out["g3"] = {"batches": len(gi), "bad": bad3[:10], "r_min": int(min(rs)), "r_max": int(max(rs)), "eval_flush_token_4x8": int(ev[0]), "expected": MP.CIDX[12 - 1]}
    out["g3_pass"] = not bad3 and 0 <= min(rs) and max(rs) <= 12 and max(rs) > min(rs) and bool((ev == MP.CIDX[11]).all()); ok &= out["g3_pass"]
    out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s161_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate161.json"), "w", newline="\n"), indent=1, default=int)
    print("GATE161", json.dumps({k: v for k, v in out.items() if k.endswith("_pass") or k in ("g1b", "g3")}, default=int), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
