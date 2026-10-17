"""§210 notes 210-2 / 210-3 ($0, local fp32; CODI-20 = boxes/runs_fsq/gsm_codi_20ep_s1/trainable.pt, LoRA merged as the host eval; codi_probe.run, bs 16).
  --part b -> eval_latent_use.json (primary), CODI's GSM8K test (1,319):
     ksweep {K: {acc, by_steps, items}} for K in 0,1,2,3,4,5,6,8,10;
     capture: the 6 continuous-thought INPUTS at K 6 (codi_probe.run latent_capture); its per-item outputs must equal ksweep K 6 (asserted);
     swap (210-3): recipients = the items correct at K 6 (the clean pool), each used once; donor = np rng 21002 choice among clean items with a DIFFERENT
       gold (never the recipient); the donor's 6 latent inputs (computed on the donor question) replace the recipient's at every latent position
       (codi_probe.run latent_override); follow = pred == donor gold, own = pred == recipient gold, other = neither; pool_size, n_pairs;
       coincidence_floor = P(recipient's intact K-6 pred == donor gold) on the same pairs;
     pos {k: {follow, own, other}}: only latent input k swapped (later positions computed by the model), k 1..6, same pairs;
     noise {acc}: every latent input of every item replaced by a Gaussian vector (np rng 21003) scaled to that input's own norm;
     items: every pair's outputs (ids, golds, intact pred, all-swap pred, pos-k preds).
  --part a -> eval_ood.json (descriptive): acc at K 6 and K 0 on CODI's released OOD eval sets, loaded exactly as CODI's test.py (zhenyi4/codi, main):
     gsm-hard = juyoung-trl/gsm-hard 'train' (instruction / response); multi-arith = ChilleD/MultiArith 'test' (final_ans); svamp = ChilleD/SVAMP
     'train' + 'test' concatenated (question_concat / Answer); answers parsed as CODI does (split '####', drop ',', float; unparseable -> inf); per-item outputs.
    python scratch/s210_latent_use.py --part b --device cuda:1"""
import argparse
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.codi_port as CP  # noqa: E402

RUN = "boxes/runs_fsq/gsm_codi_20ep_s1"


def load(dev):
    m, prj = CP.build(dev); TP = dict(CP.trainable(m, prj)); sd = torch.load(os.path.join(RUN, "trainable.pt"), map_location=dev, weights_only=False)
    for n, p_ in TP.items():
        p_.data.copy_(sd[n])
    m.eval(); prj.eval(); mm, pj = CP.merged_fp32(m, prj, dev); del m
    return mm, pj, CP.codi_tok(), CP.cp_mod()


def rows_out(rows):
    return [{"id": i, "pred": r["pred"], "gold": r["gold"], "ok": bool(r["ok"]), "txt": r["txt"], "ntok": r["ntok"]} for i, r in enumerate(rows)]


def by_steps(rows):
    by = {}
    for r in rows:
        if r["nsteps"] is not None:
            by.setdefault(r["nsteps"], []).append(r["ok"])
    return {str(k): [sum(v) / len(v), len(v)] for k, v in sorted(by.items())}


def same(a, b):
    return abs(a - b) < 1e-4


@torch.no_grad()
def part_b(dev, limit=None):
    mm, pj, tok, cp = load(dev); data = cp.gsm8k(10_000); data = data[:limit] if limit else data; out = {"ksweep": {}}
    for K in (0, 1, 2, 3, 4, 5, 6, 8, 10):
        o = cp.run(mm, pj, tok, data, K, dev=dev, bs=16); out["ksweep"][str(K)] = {"acc": o["acc"], "by_steps": by_steps(o["rows"]), "items": rows_out(o["rows"])}
        print("KSWEEP", K, round(o["acc"], 4), flush=True)
    cap = []; o6 = cp.run(mm, pj, tok, data, 6, dev=dev, bs=16, latent_capture=cap); lat = torch.cat(cap, 0)            # (N, 6, D)
    assert [r["pred"] for r in o6["rows"]] == [r["pred"] for r in out["ksweep"]["6"]["items"]], "capture run != ksweep K 6"
    N = len(data); gold = [r["gold"] for r in o6["rows"]]; pred6 = [r["pred"] for r in o6["rows"]]; clean = [i for i in range(N) if o6["rows"][i]["ok"]]
    rng = np.random.default_rng(21002); pairs = []
    for i in clean:
        pool = [j for j in clean if j != i and not same(gold[j], gold[i])]
        if pool:
            pairs.append((i, int(pool[int(rng.integers(0, len(pool)))])))
    sub = [data[i] for i, _ in pairs]; dl = lat[[d for _, d in pairs]]

    def score(rows):
        f = [same(r["pred"], gold[d]) for r, (_, d) in zip(rows, pairs)]; o_ = [same(r["pred"], gold[i]) for r, (i, _) in zip(rows, pairs)]
        return {"follow": float(np.mean(f)), "own": float(np.mean(o_)), "other": float(np.mean([not a and not b for a, b in zip(f, o_)]))}, [r["pred"] for r in rows]
    oa = cp.run(mm, pj, tok, sub, 6, dev=dev, bs=16, latent_override={"lat": dl, "pos": set(range(1, 7))}); sw, pa = score(oa["rows"])
    sw.update({"pool_size": len(clean), "n_pairs": len(pairs), "coincidence_floor": float(np.mean([same(pred6[i], gold[d]) for i, d in pairs]))})
    print("SWAP all", sw, flush=True)
    pos, ppk = {}, {}
    for k in range(1, 7):
        ok_ = cp.run(mm, pj, tok, sub, 6, dev=dev, bs=16, latent_override={"lat": dl, "pos": {k}}); pos[str(k)], ppk[k] = score(ok_["rows"]); print("SWAP pos", k, pos[str(k)], flush=True)
    g = torch.Generator().manual_seed(21003); nz = torch.randn(lat.shape, generator=g, dtype=lat.dtype); nz = nz / nz.norm(dim=-1, keepdim=True) * lat.norm(dim=-1, keepdim=True)
    on = cp.run(mm, pj, tok, data, 6, dev=dev, bs=16, latent_override={"lat": nz, "pos": set(range(1, 7))}); print("NOISE", round(on["acc"], 4), flush=True)
    items = [{"recipient": i, "donor": d, "gold_r": gold[i], "gold_d": gold[d], "pred_intact": pred6[i], "pred_swap_all": pa[q], **{"pred_pos%d" % k: ppk[k][q] for k in range(1, 7)}}
             for q, (i, d) in enumerate(pairs)]
    out.update({"swap": sw, "pos": pos, "noise": {"acc": on["acc"], "items": rows_out(on["rows"]), "def": "np/torch rng 21003 Gaussian per (item, position), scaled to that latent input's own L2 norm"},
                "items": items, "latent_norm_mean_by_pos": [float(v) for v in lat.norm(dim=-1).mean(0)], "n": N})
    return out


def ood_sets():
    from datasets import load_dataset, concatenate_datasets
    out = {}
    d = load_dataset("juyoung-trl/gsm-hard"); out["gsm-hard"] = (d["train"], "instruction", "response", "juyoung-trl/gsm-hard 'train'")
    d = load_dataset("ChilleD/MultiArith"); out["multi-arith"] = (d["test"], "question", "final_ans", "ChilleD/MultiArith 'test'")
    d = load_dataset("ChilleD/SVAMP"); out["svamp"] = (concatenate_datasets([d["train"], d["test"]]), "question_concat", "Answer", "ChilleD/SVAMP 'train' + 'test'")
    return out


def codi_answer(x):
    x = str(x); x = x.split("####")[-1] if "####" in x else x; x = x.replace(",", "")
    try:
        return float(x)
    except ValueError:
        return float("inf")


@torch.no_grad()
def part_a(dev, limit=None):
    mm, pj, tok, cp = load(dev); out = {}
    for name, (ds, qn, an, src) in ood_sets().items():
        data = [(ex[qn].strip().replace("  ", " "), codi_answer(ex[an]), None) for ex in ds]; data = data[:limit] if limit else data; r_ = {"source": src, "n": len(data)}
        for K in (6, 0):
            o = cp.run(mm, pj, tok, data, K, dev=dev, bs=16); r_[str(K)] = {"acc": o["acc"], "items": rows_out(o["rows"])}; print("OOD", name, K, round(o["acc"], 4), flush=True)
        out[name] = r_
    out["def"] = "CODI test.py (github.com/zhenyi4/codi, main) dataset ids / splits / fields / answer parsing; questions .strip().replace('  ', ' ')"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["a", "b"]); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--limit", type=int, default=0, help="smokes only")
    a = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    res = (part_b if a.part == "b" else part_a)(a.device, a.limit or None)
    res.update({"run": RUN, "card": torch.cuda.get_device_name(a.device), "dtype": "fp32 (LoRA merged)", "bs": 16, "limit": a.limit or None})
    fo = os.path.join(RUN if not a.limit else "scratch/_tmp", "eval_latent_use.json" if a.part == "b" else "eval_ood.json")
    json.dump(res, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo); print("DONE", fo, flush=True)


if __name__ == "__main__":
    main()
