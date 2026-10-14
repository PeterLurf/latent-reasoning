"""§200 GSM8K-AUG PILOT gates (contract 58947b26 + clarification 200-2; draft §3.6). Dumps boxes/runs_fsq/gate_s200/gate.json after every part.
  g1 data: train filter counts (wrong 2,959 + unparsed-only 1,094 of 384,622 lines = 1.05%); the parse round-trip on the first 10k train items; test ==
     GSM8K's 1,319 in order, valid 500; the FILLER-N strings exactly 16 / 32 GPT-2 tokens, no digits; relabel_names / resample raise on test / valid items
  g2 R_nu on 10k train items: injective, consistent (every occurrence replaced, the counts kept), same-gender pool names only, equations / answer unchanged
  g3 R_x: parse coverage; identity (x' = x renders every recomputable item byte-exactly - checked inside parse_recomputable, counted here); 10k accepted
     draws: every equation 'exact', answer == last result, each changed literal keeps its integer-digit count / decimals / integrality; acceptance rate;
     H_inf: the median over 300 recomputable items of the answer's max-probability over 100 resamples
  g4 fp64 (m.double(), Tensor.float -> double, CPU): cached vs uncached (--cache_lower) on a GSM8K batch through the harness's stepaux path (K 4 passes +
     the aux per pass + the answer CE): losses and every parameter gradient within 1e-12 relative
  g5 gradflow (fp32, fade 1, gamma 0.5): the answer CE alone and the aux CE alone each give nonzero finite gradients on the memory (wo, gate) and the trunk
  g6 eval path: greedy_after batched == one item at a time (memory read and plain trunk), cot_batched == a per-item greedy loop, on 24 test items (2070
     fp32, a 1-step model); ans_ok on fixed cases
  g7 the plain-path smoke (construct + 1 step + eval, no new flags) and a mem_check at the longest Aug item (K 10 + aux, bs 64 micro 16 -> 4 items) on argv[1]
  g8 near-duplicates: s200_test_meta's static scan (flagged >= 0.5) and its resampled-draw scan (0 exact test questions)
    python scratch/gate_s200.py [cuda dev, default cuda:2]"""
import collections
import json
import math
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_stepaux as H  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_s200"); OUT = {}; SP = os.path.join(ROOT, "scratch", "_gate_s200_tmp")


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"; os.makedirs(SP, exist_ok=True)
    try:
        R, cnt = G.load_aug("train"); T, _ = G.load_aug("test", filter_bad=False); V, _ = G.load_aug("valid", filter_bad=False)
        Qo = [json.loads(l)["question"].strip() for l in open(os.path.join(ROOT, "bench", "data", "gsm8k_test.jsonl"), encoding="utf-8")]
        meta = json.load(open(os.path.join(G.AUG, "s200_test_meta.json"), encoding="utf-8")); f16, f32 = meta["filler"]["k16"], meta["filler"]["k32"]
        raised = 0
        for x in (T[0], V[0]):
            for fn in (lambda x: G.relabel_names(x, G.load_names(), np.random.default_rng(0)), lambda x: G.resample(x, {"used_q": []}, np.random.default_rng(0))):
                try:
                    fn(x)
                except AssertionError:
                    raised += 1
        OUT["g1"] = {"filter": cnt, "filtered_frac": round((cnt["wrong"] + cnt["unparsed"]) / cnt["lines"], 5), "roundtrip_10k": sum(G.render_line(r) == r["_line"] for r in R[:10000]),
                     "test_n": len(T), "test_equals_gsm8k_order": [t["q"] for t in T] == Qo, "valid_n": len(V), "filler_tokens": [len(ENC.encode(f16)), len(ENC.encode(f32))],
                     "filler_no_digits": not re.search(r"\d", f16 + f32), "transforms_raise_on_test_valid": raised}
        OUT["g1_pass"] = (cnt["wrong"] == 2959 and cnt["unparsed"] == 1094 and OUT["g1"]["roundtrip_10k"] == 10000 and OUT["g1"]["test_equals_gsm8k_order"] and len(T) == 1319
                          and len(V) == 500 and OUT["g1"]["filler_tokens"] == [16, 32] and OUT["g1"]["filler_no_digits"] and raised == 4); dump(); print("g1", OUT["g1_pass"], flush=True)
        pools = G.load_names(); gen = {**{n: "male" for n in pools["male"]}, **{n: "female" for n in pools["female"]}}; rng = np.random.default_rng(1); ok = collections.Counter()
        for r in R[:10000]:
            y, k = G.relabel_names(r, pools, rng); ok["items"] += 1
            if not k:
                continue
            ok["relabelled"] += 1; old = sorted(w for w in set(G.CAPW.findall(r["q"])) if w in gen)
            new_words = re.findall(r"\b[A-Z][a-z]+\b", y["q"]); oldw = re.findall(r"\b[A-Z][a-z]+\b", r["q"])
            mp = {a: b for a, b in zip(oldw, new_words) if a != b}
            ok["eqs_ans_same"] += y["eqs"] == r["eqs"] and y["ans"] == r["ans"]
            ok["injective"] += len(set(mp.values())) == len(mp) and set(mp) == set(old)
            wc = lambda w, t: len(re.findall(r"\b%s\b" % re.escape(w), t))       # noqa: E731
            ok["consistent"] += all(wc(a, r["q"]) == wc(b, y["q"]) and wc(a, y["q"]) == 0 for a, b in mp.items())
            ok["same_gender"] += all(gen.get(b) == gen[a] for a, b in mp.items())
            ok["length_same_words"] += len(new_words) == len(oldw)
        n_ = ok["relabelled"]; OUT["g2"] = {**dict(ok), "pool_sizes": [len(pools["male"]), len(pools["female"])]}
        OUT["g2_pass"] = n_ > 0 and all(ok[k] == n_ for k in ("eqs_ans_same", "injective", "consistent", "same_gender", "length_same_words")); dump(); print("g2", OUT["g2_pass"], flush=True)
        why = collections.Counter(); TP = []
        for r in R:
            tp, w = G.parse_recomputable(r); why[w] += 1; TP.append(tp)
        rng = np.random.default_rng(2); acc = tried = 0; bad = collections.Counter(); keep = 0
        for r, tp in zip(R, TP):
            if tp is None:
                continue
            y, a = G.resample(r, tp, rng); tried += 1; acc += a
            if not a:
                continue
            keep += 1
            if any(G.check_eq(e) != "exact" for e in y["eqs"]):
                bad["eq_not_exact"] += 1
            if abs(G.safe_eval(y["eqs"][-1].split("=")[1].replace(",", "")) - G.fnum(G.NUMTOK.search(y["ans"]).group(0))) > 1e-9 * max(1, abs(G.fnum(G.NUMTOK.search(y["ans"]).group(0)))):
                bad["ans_ne_last"] += 1
            o_ = [m_.group(0) for m_ in G.NUM.finditer(r["q"])]; n2 = [m_.group(0) for m_ in G.NUM.finditer(y["q"])]
            if len(o_) != len(n2):
                bad["literal_count"] += 1
            else:
                for u, v in zip(o_, n2):
                    if u != v and (len(u.replace(",", "").split(".")[0]) != len(v.replace(",", "").split(".")[0]) or G.decimals(u) != G.decimals(v)):
                        bad["digits_or_decimals"] += 1
            if keep >= 10000:
                break
        rr = np.random.default_rng(3); ids = [i for i, tp in enumerate(TP) if tp is not None]; hinf = []
        for i in rr.permutation(ids)[:300]:
            c = collections.Counter(G.resample(R[i], TP[i], rr)[0]["ans"] for _ in range(100)); hinf.append(max(c.values()) / 100)
        OUT["g3"] = {"parse": dict(why), "recomputable_frac": round(why["ok"] / len(R), 4), "identity_failures_excluded": why["identity_render"], "accept_rate": round(acc / tried, 4),
                     "checked_accepted": keep, "bad": dict(bad), "answer_maxprob_median_100": float(np.median(hinf)), "answer_maxprob_p90": float(np.percentile(hinf, 90))}
        OUT["g3_pass"] = keep >= 10000 and not bad; dump(); print("g3", OUT["g3_pass"], flush=True)
        _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double(); DM.FAST_ADD = True
        try:
            g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = ProsMem(g).double()
            with torch.no_grad():
                m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
            its = [G.encode(r, ENC, EOS) for r in sorted(R[:400], key=lambda r: -len(r["eqs"]))[:6]]; x, msk = m.think_inputs(its, "cpu"); b4 = {}

            def path(c):
                S = m.mem.zero_state(len(its), "cpu").double(); hl = m.lower(x) if c else None; per = [[] for _ in its]
                for t in range(1, 5):
                    S = m.write(S, x, msk, hl); idx = [i for i, it in enumerate(its) if t <= it["depth"]]
                    if idx:
                        lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, "cpu"); ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                        for r_, i in enumerate(idx):
                            per[i].append(ce[r_])
                return m.dec_loss(S, its, "cpu") + torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()
            res_ = []
            for c in (False, True):
                m.zero_grad(set_to_none=True); l_ = path(c); l_.backward(); res_.append((l_.detach(), grads(m)))
            (l0, g0), (l1, g1) = res_
            b4 = {"loss_rel_diff": float((l0 - l1).abs() / l0.abs()), "same_keys": set(g0) == set(g1),
                  "grad_rel": max(float((g0[k] - g1[k]).norm() / g0[k].norm().clamp_min(1e-300)) for k in g0), "n_grads": len(g0), "dtype": str(l0.dtype)}
        finally:
            torch.Tensor.float = _f
        OUT["g4"] = b4; OUT["g4_pass"] = b4["loss_rel_diff"] <= 1e-12 and b4["grad_rel"] <= 1e-12 and b4["same_keys"] and b4["dtype"] == "torch.float64"; dump(); print("g4", OUT["g4_pass"], flush=True)
        g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = ProsMem(g)
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        its = [G.encode(r, ENC, EOS) for r in R[:6]]; x, msk = m.think_inputs(its, "cpu"); gf = {}
        for term in ("answer", "aux"):
            m.zero_grad(set_to_none=True); S = m.mem.zero_state(len(its), "cpu"); per = []
            for t in range(1, 5):
                S = m.write(S, x, msk); idx = [i for i, it in enumerate(its) if t <= it["depth"]]
                if idx:
                    lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, "cpu"); per.append(Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100))
            l_ = m.dec_loss(S, its, "cpu") if term == "answer" else torch.stack(per).mean(); l_.backward()
            nm = lambda p: float(p.grad.abs().sum()) if p.grad is not None and torch.isfinite(p.grad).all() else 0.0   # noqa: E731
            gf[term] = {"mem_wo": nm(m.mem.wo.weight), "mem_gate": nm(m.mem.gate.weight), "trunk_any": sum(nm(p) for n, p in m.g.named_parameters())}
        OUT["g5"] = gf; OUT["g5_pass"] = all(v > 0 for d in gf.values() for v in d.values()); dump(); print("g5", OUT["g5_pass"], flush=True)
        ok_cases = [H.ans_ok(" 18", "18"), H.ans_ok(" 1,800", "1800"), H.ans_ok(" 12.5", "12.5"), not H.ans_ok(" 17", "18"), not H.ans_ok(None, "18"), H.ans_ok(" 5.0", "5")]
        g = s5.load_gpt2(dev); torch.manual_seed(1); m = ProsMem(g).to(dev).eval()
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        ac = lambda: torch.autocast("cpu", enabled=False)                       # noqa: E731
        Dt = [G.encode(r, ENC, EOS) for r in T[:24]]; x, msk = m.think_inputs(Dt, dev)
        with torch.no_grad():
            S = m.think(m.mem.zero_state(len(Dt), dev), x, msk, 4)
        pre = [[EOT] + it["pre"] + [ANS] for it in Dt]
        bat = H.greedy_after(H.Fwd(g, m.mem, S), pre, dev, ac, 8); one = [H.greedy_after(H.Fwd(g, m.mem, S[i:i + 1]), [pre[i]], dev, ac, 1)[0] for i in range(len(Dt))]
        batp = H.greedy_after(H.Fwd(g), [it["pre"] for it in Dt], dev, ac, 8); onep = [H.greedy_after(H.Fwd(g), [it["pre"]], dev, ac, 1)[0] for it in Dt]
        cb = H.cot_batched(g, [it["pre"] for it in Dt], 40, dev, None, 8); cn = []
        for it in Dt:
            seq = list(it["pre"]); hit = False
            with torch.no_grad():
                for _ in range(40):
                    t_ = int(g.logits(g(g.embed(torch.tensor([seq], device=dev)), None)[0][0, -1:]).float().argmax(-1))
                    if t_ in (ANS, EOS):
                        hit = t_ == ANS; break
                    seq.append(t_)
            cn.append((seq, hit))
        OUT["g6"] = {"ans_ok_cases": ok_cases, "greedy_mem_batched_eq_single": bat == one, "greedy_plain_batched_eq_single": batp == onep, "cot_batched_eq_loop": cb == cn,
                     "sample": bat[:3]}
        OUT["g6_pass"] = all(ok_cases) and bat == one and batp == onep and cb == cn; dump(); print("g6", OUT["g6_pass"], flush=True)
        del g, m; torch.cuda.empty_cache(); env = {**os.environ, "PYTHONPATH": ROOT}; s7 = {}
        o_ = os.path.join(SP, "plain"); r = subprocess.run([sys.executable, "-m", "boxes.gsm8k_stepaux", "--arm", "stepaux", "--think_k", "3", "--steps", "1", "--bs", "4", "--train_n", "500",
                                                            "--test_n", "4", "--valid_n", "4", "--fit_n", "4", "--extras_n", "4", "--eval_bs", "4", "--device", dev, "--out", o_], capture_output=True, text=True, env=env)
        s7["plain_rc"] = r.returncode; s7["plain_err"] = r.stderr[-800:] if r.returncode else ""; s7["plain_eval"] = os.path.exists(os.path.join(o_, "eval.json"))
        o_ = os.path.join(SP, "mem"); r = subprocess.run([sys.executable, "-m", "boxes.gsm8k_stepaux", "--arm", "stepaux", "--think_k", "10", "--aux_fade", "24000,24000", "--cache_lower",
                                                          "--fast_add", "--chunk_delta", "64", "--bs", "64", "--micro", "16", "--mem_check", "--device", dev, "--out", o_], capture_output=True, text=True, env=env)
        s7["mem_rc"] = r.returncode; s7["mem"] = json.load(open(os.path.join(o_, "mem_check.json"))) if r.returncode == 0 else r.stderr[-800:]
        OUT["g7"] = s7                                                      # the composite worst case: the longest question (213 tokens, 214 with EOT) + the longest aux text at t <= K (memory scales with tokens, not m)
        OUT["g7_pass"] = s7["plain_rc"] == 0 and s7["plain_eval"] and s7["mem_rc"] == 0 and s7["mem"]["longest_question_tokens"] == 213 == s7["mem"]["max_question_tokens"] and s7["mem"]["aux_tokens"] == s7["mem"]["max_aux_tokens"]
        dump(); print("g7", OUT["g7_pass"], flush=True)
        OUT["g8"] = {"static_flagged_ge_0.5": int(sum(meta["dedupe_flag"])), "static_max": max(meta["dedupe_score"]), "resampled_draw_scan": meta["resampled_draw_scan"]}
        OUT["g8_pass"] = meta["resampled_draw_scan"]["exact_test_question"] == 0; dump(); print("g8", OUT["g8_pass"], flush=True)
    finally:
        shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get("g%d_pass" % i) for i in range(1, 9))); dump(); print(json.dumps({k: v for k, v in OUT.items() if k.endswith("pass")}), flush=True)


if __name__ == "__main__":
    main()
