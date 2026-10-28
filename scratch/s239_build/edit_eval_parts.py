"""§239 LONG-RUN (LR_v1 §2, 1e8b2033; Note 239-1): inject_aw_eval gets part_long + run_long + the CLI option --kprime_list (default '' = the current main path,
unchanged). run_long writes ONLY <run>/eval_long.json (eval.json and every other file untouched). One passes() call per d serves every K' (acc(8) from the same call,
Note 239-1). Edits the generator inputs; each asserted once."""
p = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(p, encoding="utf-8").read()
s += '''

@torch.no_grad()
def part_long(m, g, sets, a, dev, bs, kplist, ds):
    """§239 LONG-RUN (LR_v1 §2): on TEST2 of the run's law, d in ds, ONE pass sequence per batch up to max(kplist) (every K' from the same call, Note 239-1): answers
    at each K'; s_norm[K'] = the mean ||S_K'|| (Frobenius over the whole state); s_cos[K'] = the mean cos(S_K', S_8) (flattened, per item); nonfinite = items whose
    S_K' has a non-finite entry. kplist must hold 8 and nothing below it (the cos reference)."""
    assert 8 in kplist and min(kplist) >= 8, ("kprime_list needs 8 and K' >= 8", kplist)
    T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}
    for d in ds:
        nrm = {k: [] for k in kplist}; cs = {k: [] for k in kplist}; nf = {k: 0 for k in kplist}; ref = {}

        def grab(t, S, sl):
            if t == 8:
                ref["S"] = S.float().flatten(1).clone()
            if t in nrm:
                f = S.float().flatten(1); nrm[t].append(f.norm(dim=1).cpu()); cs[t].append(torch.nn.functional.cosine_similarity(f, ref["S"], dim=1).cpu())
                nf[t] += int((~torch.isfinite(f)).any(dim=1).sum())
        A, _ = passes(m, T[d], dev, bs, set(kplist), on_state=grab); y = np.array([it["ans"] for it in T[d]])
        out["by_d"][str(d)] = {"acc": {str(k): float((A[k] == y).mean()) for k in kplist}, "s_norm": {str(k): float(torch.cat(nrm[k]).mean()) for k in kplist},
                               "s_cos": {str(k): float(torch.cat(cs[k]).mean()) for k in kplist}, "nonfinite": {str(k): nf[k] for k in kplist}}
        out["items"][str(d)] = items_rec(T[d], {"pred": {str(k): A[k].tolist() for k in kplist}})
    out["def"] = ("LONG-RUN (§239 LR_v1 §2): TEST2 of the run's law, d " + ",".join(str(d) for d in ds) + "; one pass sequence to max K' per batch; acc / s_norm (mean ||S_K'||) / "
                  "s_cos (mean cos(S_K', S_8)) per K'; nonfinite = items with a non-finite S_K' entry (COLLAPSE by LR_v1 §5)")
    return out


def run_long(out, dev, bs, kplist, ds):
    """§239: load <out>/model.pt as load() does, build the run law's sets (set shas asserted vs args.json), part_long, write ONLY <out>/eval_long.json."""
    t1 = time.time(); a, m, g = load(out, dev); assert m is not None, "LONG-RUN: inject runs only"; sets = IO.build_sets(mul_s_p=law(a))
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§239: set sha mismatch vs args.json", k_)
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    r_ = part_long(m, g, sets, a, dev, bs, kplist, ds)
    r_.update({"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu", "eval_bs": bs, "kprime_list": kplist, "ds": ds,
               "law": law(a), "set_law": sets["shas"].get("set_law", "default"), "think_k": a["think_k"], "module_sha256": IO.sha_file(os.path.abspath(__file__)),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if str(dev).startswith("cuda") else None, "sec": round(time.time() - t1)})
    json.dump(r_, open(os.path.join(out, "eval_long.json"), "w", newline="\\n")); print("LONG done", out, r_["sec"], "s", flush=True)
'''
open(p, "w", encoding="utf-8", newline="\n").write(s); print("eval parts: + part_long, run_long")
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
o = "]\nfor o, n in E:"
add = '''# §239 LONG-RUN: the --kprime_list CLI option (default '' = the unchanged run() path) -> run_long (eval_long.json only)
(\'    a = ap.parse_args(); run(a.run, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs); print("DONE", a.run, flush=True)\',
 \'    ap.add_argument("--kprime_list", default="", help="§239: e.g. 8,16,32,64 -> eval_long.json only (run_long); default empty = the parts path")\\n\'
 \'    ap.add_argument("--long_ds", default="2,3,4,5", help="§239: the d values of run_long")\\n\'
 \'    a = ap.parse_args()\\n\'
 \'    if a.kprime_list:\\n\'
 \'        run_long(a.run, a.device, a.eval_bs, [int(x) for x in a.kprime_list.split(",")], [int(x) for x in a.long_ds.split(",")]); return\\n\'
 \'    run(a.run, [p_ for p_ in a.parts.split(",") if p_], a.device, a.eval_bs); print("DONE", a.run, flush=True)\'),
]
for o, n in E:'''
assert s.count(o) == 1; s = s.replace(o, add); open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw_eval: + main edit")
