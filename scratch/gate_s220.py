"""§220 GSM-CACHE-SWAP: CPU unit tests for scratch/s220_gcs.py (no GPU, no real-model output on GSM8K items). Writes only under scratch/_tmp/s220/.
  T1 parser (CODI test.py extract_answer_number semantics);
  T2 graph signatures / executable programs / oracle-cf on hand examples; the reviewer's signature count on the test annotations (text only);
  T3 pair drawing on the real test QUESTION TEXT + token lengths (no model; every item treated as eligible): reuse <= 3, same length, golds differ,
     each recipient once, determinism; graph-matched draw invariants (Δ = 0 row, ocf not in {gold_r, gold_d});
  T4 tiny random GPT-2 (4 layers): cache length / slot indices; self cache-write == intact bitwise; donor cache-write keeps every other row bitwise and
     changes the output path; LEAK-SEG slots; the assertion FIRES on a deliberately corrupted outside row (K and V, a non-written layer);
     official semantics with 0 pads == the bs-1 path bitwise; QMASK changes the answer-phase computation; STATE-SWAP self == intact bitwise;
     SCIT teacher-forced log-probs == the incremental greedy log-probs on the greedy continuation;
  T5 (optional, --load) load-only of the real CODI20 / zen-E checkpoints on CPU (sha256, parameter counts; no forward).
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python scratch/gate_s220.py [--load]  -> scratch/_tmp/s220/gate_s220.json"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import s220_gcs as S  # noqa: E402

torch.set_num_threads(2)
RES = []


def check(name, ok, info=""):
    RES.append({"test": name, "pass": bool(ok), "info": str(info)[:300]}); print("PASS" if ok else "FAIL", name, info, flush=True)


def t1():
    P = S.parse_official
    check("T1 last number anywhere", P("The answer is: 18") == 18.0 and P("3 apples, then 4. The answer is: 7") == 7.0)
    check("T1 commas removed", P("The answer is: 1,200") == 1200.0)
    check("T1 negative / decimal", P("x -3.5") == -3.5 and P("The answer is: 5.") == 5.0)
    check("T1 no number -> None", P("The answer is: none") is None)


def t2():
    g = S.graph_info({"cot": "<<16-3-4=9>> <<9*2=18>>", "gold": 18.0})
    check("T2 signature", g["sig"] == ("q-q-q", "r0*q") and g["exec"], g["sig"])
    h = S.graph_info({"cot": "<<20-5-1=14>> <<14*2=28>>", "gold": 28.0})
    o = S.oracle_cf(g, h, 0); check("T2 oracle-cf (recipient program on donor's step-1 result)", o == 28.0, o)
    bad = S.graph_info({"cot": "<<2*3=7>>", "gold": 7.0}); check("T2 non-executable annotation flagged", not bad["exec"])
    items = S.load_test(); GI = [S.graph_info(it) for it in items]
    nsig = len(set(x["sig"] for x in GI)); nex = sum(x["exec"] for x in GI)
    check("T2 reviewer's signature count (855 distinct; reviewer used the HF answer field)", nsig == 855, "distinct %d, executable %d / %d" % (nsig, nex, len(items)))
    return items, GI


def t3(items, GI):
    tok = __import__("boxes.codi_port", fromlist=["x"]).codi_tok(); L = [len(tok(it["q"])["input_ids"]) for it in items]; gold = [it["gold"] for it in items]
    allc = list(range(len(items)))
    p1, nod, uses = S.draw_exact(allc, gold, L); p2, _, _ = S.draw_exact(allc, gold, L)
    ok = (p1 == p2 and max(uses.values()) <= 3 and all(L[r] == L[d] and gold[r] != gold[d] and r != d for r, d in p1)
          and len(set(r for r, _ in p1)) == len(p1) and len(p1) + len(nod) == len(allc))
    check("T3 exact-length draw invariants (all items eligible)", ok, "pairs %d no-donor %d max reuse %d" % (len(p1), len(nod), max(uses.values())))
    # a CODI20-like pool: a deterministic 38% subset (NOT model correctness; text-only check of availability)
    sub = [k for k in allc if (k * 2654435761) % 1000 < 380]; p3, nod3, u3 = S.draw_exact(sub, gold, L)
    check("T3 exact-length availability on a 38% pseudo-pool", len(p3) >= 0.9 * len(sub), "pool %d pairs %d (%.3f) max reuse %d" % (len(sub), len(p3), len(p3) / len(sub), max(u3.values())))
    d0, dne, cnt, gu = S.draw_graph(sub, gold, L, GI)
    ok = all(L[x["r"]] == L[x["d"]] and x["delta"] == 0 for x in d0) and all(x["delta"] != 0 for x in dne) and \
        all(x["ocf"] not in (gold[x["r"]], gold[x["d"]]) and GI[x["r"]]["sig"] == GI[x["d"]]["sig"] for x in d0 + dne) and (max(gu.values()) if gu else 0) <= 3
    check("T3 graph-matched draw invariants (38% pseudo-pool)", ok, "d0 %d dne0 %d counts %s" % (len(d0), len(dne), cnt))


def tiny_runner():
    m, prj, tok, _ = S.load_model("tiny", "cpu"); return S.Runner(m, prj, tok, "cpu", 12)


def t4():
    R = tiny_runner(); qa = R.ids("Ann has 12 apples and buys 30 more. How many apples does Ann have?"); qb = R.ids("Bob has 41 apples and buys 7 more. How many apples does Bob have?")
    assert len(qa) == len(qb), (len(qa), len(qb))
    pa = R.prefix(qa); check("T4 cache length P+K, latent slots", pa["cache"].get_seq_length() == len(qa) + 1 + 6 and R.latent_slots(pa) == list(range(len(qa) + 1, len(qa) + 7)),
                             (pa["P"], R.latent_slots(pa), R.leak_slots(pa)))
    intact = R.decode(R.prefix(qa))
    c = R.copy(pa); ok = R.write_rows(c, R.prefix(qa), R.latent_slots(pa), R.latent_slots(pa), list(range(R.nl)), "kv", "t4self")
    check("T4 self cache-write: assert ok and decode == intact bitwise", ok and R.decode(c)["sha"] == intact["sha"])
    st = R.decode(R.prefix(qa, override={j: pa["z"][j - 1] for j in range(1, 7)})); check("T4 self STATE-SWAP == intact bitwise", st["sha"] == intact["sha"])
    pb = R.prefix(qb); c = R.copy(pa); ok = R.write_rows(c, pb, R.latent_slots(pa), R.latent_slots(pb), list(range(R.nl)), "kv", "t4donor")
    diff = any(not torch.equal(c["cache"].layers[l].keys[:, :, -6:], pa["cache"].layers[l].keys[:, :, -6:]) for l in range(R.nl))
    with torch.no_grad():
        x = R.wte(R.t([[R.EOT]])); pos = R.t([[pa["next"]]]); ci = R.copy(pa)
        l_int = R.m(inputs_embeds=x, position_ids=pos, past_key_values=ci["cache"], use_cache=True).logits
        cd = R.copy(c); l_sw = R.m(inputs_embeds=x, position_ids=pos, past_key_values=cd["cache"], use_cache=True).logits
    check("T4 donor cache-write: assert ok, latent rows replaced, <eot>-step logits change", ok and diff and not torch.equal(l_int, l_sw),
          "max|dlogit| %.2e" % float((l_int - l_sw).abs().max()))
    c = R.copy(pa); ok = R.write_rows(c, pb, R.leak_slots(pa), R.leak_slots(pb), list(range(R.nl)), "kv", "t4leak")
    check("T4 LEAK-SEG write (Lq-6..Lq-1): assert ok", ok and R.leak_slots(pa) == list(range(len(qa) - 6, len(qa))))
    n0 = len(R.fails)

    def corrupt_k(cc):
        cc.layers[0].keys[0, 0, 0, 0] += 1.0

    def corrupt_v_other_layer(cc):
        cc.layers[3].values[0, 1, R.latent_slots(pa)[0], 5] += 1e-6
    c = R.copy(pa); ok1 = R.write_rows(c, pb, R.latent_slots(pa), R.latent_slots(pb), list(range(R.nl)), "kv", "t4corruptK", _test_corrupt=corrupt_k)
    c = R.copy(pa); ok2 = R.write_rows(c, pb, R.latent_slots(pa), R.latent_slots(pb), [0, 1], "kv", "t4corruptV", _test_corrupt=corrupt_v_other_layer)
    c = R.copy(pa); ok3 = R.write_rows(c, pb, R.latent_slots(pa), R.latent_slots(pb), list(range(R.nl)), "k", "t4konly_vchanged",
                                        _test_corrupt=lambda cc: cc.layers[2].values[:, :, R.latent_slots(pa)[0]:R.latent_slots(pa)[-1] + 1].mul_(1.0001))   # a view (basic slicing), so in place
    check("T4 assertion FIRES on corrupted outside rows (K row 0; V in a non-written layer; V under a K-only write)", not ok1 and not ok2 and not ok3 and len(R.fails) == n0 + 3)
    R.fails = R.fails[:n0]
    off = R.decode(R.prefix(qa, npad=0)); check("T4 official semantics with 0 pads == bs-1 path bitwise", off["sha"] == intact["sha"])
    off3 = R.prefix(qa, npad=3); check("T4 official with 3 pads: cache includes pads", off3["cache"].get_seq_length() == len(qa) + 1 + 3 + 6)
    with torch.no_grad():
        x = R.wte(R.t([[R.EOT]])); pr1 = R.copy(pa); pr2 = R.copy(pa); pos = R.t([[pa["next"]]])
        l_a = R.m(inputs_embeds=x, position_ids=pos, past_key_values=pr1["cache"], use_cache=True).logits
        am = torch.ones(1, pa["next"] + 1, dtype=torch.long); am[0, :pa["Lq"]] = 0
        l_b = R.m(inputs_embeds=x, position_ids=pos, past_key_values=pr2["cache"], attention_mask=am, use_cache=True).logits
        pr3 = R.copy(pa); l_c = R.m(inputs_embeds=x, position_ids=pos, past_key_values=pr3["cache"], attention_mask=torch.ones(1, pa["next"] + 1, dtype=torch.long), use_cache=True).logits
    check("T4 QMASK changes the answer-phase logits; an all-ones mask matches no mask", not torch.allclose(l_a, l_b) and torch.allclose(l_a, l_c, atol=1e-6),
          "max|all-ones - none| %.2e" % float((l_a - l_c).abs().max()))
    # SCIT vs incremental: score the greedy continuation's own tokens through the teacher-forced path
    with torch.no_grad():
        c = R.copy(pa); cache = c["cache"]; pos = pa["next"]; x = R.wte(R.t([[R.EOT]])); lps = []; gen = []
        for _ in range(8):
            o = R.m(inputs_embeds=x, position_ids=R.t([[pos]]), past_key_values=cache, use_cache=True); cache = o.past_key_values
            lp = torch.log_softmax(o.logits[0, -1, :R.V - 1].float(), -1); nxt = int(lp.argmax()); lps.append(float(lp[nxt])); gen.append(nxt); x = R.wte(R.t([[nxt]])); pos += 1
        c = R.copy(pa); xx = R.wte(R.t([[R.EOT] + gen[:-1]])); o = R.m(inputs_embeds=xx, position_ids=torch.arange(pa["next"], pa["next"] + 8)[None], past_key_values=c["cache"], use_cache=True)
        tf = torch.log_softmax(o.logits[0, :, :R.V - 1].float(), -1).gather(1, R.t(gen)[:, None])[:, 0]
    check("T4 SCIT teacher-forced log-probs == incremental (8 greedy tokens)", np.allclose(tf.numpy(), lps, atol=1e-4), "max diff %.2e" % float(np.abs(tf.numpy() - np.array(lps)).max()))
    sc = R.scit(pa, {"a": "18", "b": "1200"}); check("T4 SCIT candidates scored (tail = number + EOS)", sc["a"]["tail_ntok"] == 2 and sc["b"]["tail_ntok"] >= 2 and np.isfinite(sc["a"]["sum"]), sc)


def t5():
    for mname in ("codi20", "zen"):
        m, prj, tok, info = S.load_model(mname, "cpu")
        check("T5 load-only %s" % mname, info.get("params_loaded", 0) > 0 and m.config.vocab_size == 50260 and m.config._attn_implementation == "eager",
              {k: info[k] for k in ("model_sha256", "params_loaded", "path")})
        del m, prj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--load", action="store_true"); a = ap.parse_args()
    t1(); items, GI = t2(); t3(items, GI); t4()
    if a.load:
        t5()
    out = os.path.join(ROOT, "scratch", "_tmp", "s220"); os.makedirs(out, exist_ok=True)
    json.dump({"tests": RES, "all_pass": all(r["pass"] for r in RES)}, open(os.path.join(out, "gate_s220.json"), "w", newline="\n"), indent=1)
    print("ALL PASS" if all(r["pass"] for r in RES) else "SOME FAILED", "%d/%d" % (sum(r["pass"] for r in RES), len(RES)))
    sys.exit(0 if all(r["pass"] for r in RES) else 1)


if __name__ == "__main__":
    main()
