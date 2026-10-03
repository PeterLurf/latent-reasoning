import os, sys
sys.path.insert(0, "D:/Latent")
import numpy as np, torch
from boxes import s5
from boxes.mult98 import load, parse, MultGen, gen_batch, gen_rollout, chunks_of, to_int, DIGIT_IDS, TOK, valid_step, K

torch.manual_seed(0); torch.set_num_threads(3)
# data: parse round trip and arithmetic consistency (P_j = a * b_j * 10^(j-1), S_j = running sum, product = S_n)
for task in ("4x4", "5x5"):
    D = load(f"D:/Latent/bench/data/mult_{task}_test.txt"); n = D["n"]
    a = to_int(D["A"]); b = to_int(D["B"]); p = to_int(D["prod"])
    assert all(x * y == z for x, y, z in zip(a, b, p)), "product"
    inv = {t: i for i, t in enumerate(DIGIT_IDS)}
    for j in range(n):
        st = D["steps"][j]
        if j == 0:
            P = to_int(np.vectorize(inv.get)(st))
        else:
            assert (st[:, 0] == TOK["+"]).all()
            body = st[:, 1:]
            if j < n - 1:
                LP = (body.shape[1] - 2) // 2
                P = to_int(np.vectorize(inv.get)(body[:, :LP])); S = to_int(np.vectorize(inv.get)(body[:, LP + 1:-1]))
            else:
                P = to_int(np.vectorize(inv.get)(body))
        want_P = [x * int(bb) * 10 ** j for x, bb in zip(a, D["B"][:, j])]
        assert P == want_P, f"P_{j+1}"
        if 0 < j < n - 1:
            want_S = [x * (y % 10 ** (j + 1)) for x, y in zip(a, b)]
            assert S == want_S, f"S_{j+1}"
            assert valid_step(st, LP, LP).all()
    print(task, "data ok: steps", [s.shape[1] for s in D["steps"]], "ans", D["ans"].shape[1])

dev = "cpu"; D = load("D:/Latent/bench/data/mult_5x5_test.txt", limit=6)
for arm in ("genm", "mask"):
    m = MultGen(s5.load_gpt2(dev), arm).eval(); g = m.g
    with torch.no_grad():
        prefix, start, ops, st, ans = [x for x in gen_batch(D, np.random.default_rng(3), 3, dev)]
        # use chunk 0 explicitly (ops 1-4) for the checks
        idx = np.arange(3); T = lambda x: torch.tensor(x)
        prefix = T(np.concatenate([DIGIT_IDS[D["A"][idx]], np.full((3, 1), TOK["*"])], 1)); ops = T(D["B"][idx, :4])
        st = [T(D["steps"][j][idx]) for j in range(4)]
        l1, d1, _, i1 = m.chunk(prefix, None, ops, step_tgt=st, step_len=[s.shape[1] for s in st])
        _, d2, _, i2 = m.chunk(prefix, None, ops, step_len=[s.shape[1] for s in st])
        assert all(torch.equal(x, y) for x, y in zip(d1, d2)) and torch.allclose(i1, i2), "forks must not change the chain"
        # fork CE == sequential teacher forcing through the same (masked or full) fork
        j = 2; x = torch.cat([g.embed(prefix), g.embed(m.digtok[ops[:, 0]])[:, None]], 1); past = None
        for jj in range(j + 1):
            h, past = g(x, past); hs, past = g(i1[:, jj][:, None], past); x = g.embed(m.digtok[ops[:, jj + 1]])[:, None]
        hslot = hs[:, 0]; tgt = st[j]
        ce_tf = m.fork_tf(hslot, past, tgt)
        fp, p0 = m.fork_past(past); seq = [g.logits(hslot)]; pp = fp; pos = p0
        for t in range(tgt.shape[1] - 1):
            h1, pp = g(g.embed(tgt[:, t])[:, None], pp, pos0=pos); seq.append(g.logits(h1[:, 0])); pos = None if pos is None else pos + 1
        lg = torch.stack(seq, 1)
        ce_seq = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape).sum(-1).mean()
        assert torch.allclose(ce_tf, ce_seq, atol=1e-3), (float(ce_tf), float(ce_seq))
        assert torch.equal(m.fork_greedy(hslot, past, 5), d1[j][:, :5]), "greedy == chunk decode"
        if arm == "mask":   # a masked fork must not depend on anything but the slot: perturb an op token's K/V
            past2 = [(k.clone(), v.clone()) for k, v in past]
            for k_, v_ in past2:
                k_[:, :, 2] += 5.0; v_[:, :, 2] += 5.0
            assert torch.allclose(m.fork_tf(hslot, past2, tgt), ce_tf), "mask leaks"
        else:
            past2 = [(k.clone(), v.clone()) for k, v in past]
            for k_, v_ in past2:
                k_[:, :, 2] += 5.0; v_[:, :, 2] += 5.0
            assert not torch.allclose(m.fork_tf(hslot, past2, tgt), ce_tf), "unmasked fork should see the context"
        per, ok, snaps = gen_rollout(m, D, dev, None, 4)
        print(arm, "ok: loss", round(float(l1), 2), "CE tf/seq", round(float(ce_tf), 3), round(float(ce_seq), 3), "rollout", per.shape, ok.shape, [s.shape for s in snaps])
    m.train(); loss, _, _, _ = m.chunk(prefix, None, ops, step_tgt=st); loss.backward()
    assert m.g.p["wte_weight"].grad.abs().sum() > 0
print("SMOKE98_UNIT_OK")
