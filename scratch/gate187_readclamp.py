"""note 187-8 / 8a READ-CLAMP gates (deltamem.READ_CLAMP / read_clamp, halt173 --readclamp):
  (1) READ_CLAMP None == the pre-change modules bitwise on every path (gate187_posoffset.outputs: FAST off / on, mask / none, lower, fwd_read, dec_reads)
  (2) clamp mode with every reference +inf == unclamped, bitwise, on every path
  (3) read_clamp on a fixed tensor: a ref below some norms clamps them, every clamped vector keeps its direction (cos 1 to 1e-6) and ends at norm = ref;
      rescale mode puts every vector at the mean norm, direction kept
  (4) smoke: readclamp_fields on a small fixture (s184_reweight_s1, 40 ref items, 8 test items per n) - all keys present, some read clamped on n 9..12
    python scratch/gate187_readclamp.py <pre deltamem.py> <pre deltas5.py> <pre nested170.py>   -> boxes/runs_fsq/s187_gates/gate187_readclamp.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

torch.set_num_threads(3)
import gate187_posoffset as GP  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.deltas5 as D5  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import boxes.halt173 as H  # noqa: E402


class InfRef(dict):
    def __missing__(self, k):
        return torch.full((6,), float("inf"))


def main():
    OUT = {}; DM.FAST_ADD = True
    kind, m, a = H.load_model("boxes/runs_fsq/s184_reweight_s1", "cpu"); TE = N.test_set(1000); items = [it for it in TE if it["n"] == 6][:8]
    with torch.no_grad():
        DM.READ_CLAMP = None; o0 = GP.outputs(DM, D5, N, m, items)
        DM.READ_CLAMP = {"mode": "clamp", "ref": InfRef(), "ref_mean": None, "kmax": {"think": 12, "dec": 12, "aux": 8}, "hits": [], "path": "dec", "k": 3}
        oinf = GP.outputs(DM, D5, N, m, items); DM.READ_CLAMP = None
        pdm = GP.load_pre("pre_dm", sys.argv[1], "boxes.deltamem"); pd5 = GP.load_pre("pre_d5", sys.argv[2], "boxes.deltas5"); pn = GP.load_pre("pre_n", sys.argv[3], "boxes.nested170")
        cls_ = m.mem.__class__; m.DM = pdm; m.mem.__class__ = pdm.DeltaMem; pre = GP.outputs(pdm, pd5, pn, m, items)
        m.DM = DM; m.mem.__class__ = cls_; sys.modules["boxes.deltamem"], sys.modules["boxes.deltas5"], sys.modules["boxes.nested170"] = DM, D5, N
    OUT["g1"] = {k: bool(torch.equal(o0[k], pre[k])) for k in o0}; OUT["g1_pass"] = all(OUT["g1"].values())
    OUT["g2"] = {k: bool(torch.equal(o0[k], oinf[k])) for k in o0}; OUT["g2_pass"] = all(OUT["g2"].values())
    g = torch.Generator().manual_seed(0); oa = torch.randn(2, 5, 6, 64, generator=g) * torch.rand(2, 5, 6, 1, generator=g) * 3
    nrm = oa.norm(dim=-1); ref = nrm.flatten(0, 1).median(0).values
    DM.READ_CLAMP = {"mode": "clamp", "ref": {("dec", 1): ref}, "kmax": {"dec": 12}, "hits": [], "path": "dec", "k": 1}; oc = DM.read_clamp(oa)
    DM.READ_CLAMP = {"mode": "rescale", "ref_mean": {("dec", 1): ref}, "kmax": {"dec": 12}, "hits": [], "path": "dec", "k": 1}; orr = DM.read_clamp(oa); DM.READ_CLAMP = None
    cos = torch.nn.functional.cosine_similarity(oc, oa, dim=-1); above = nrm > ref
    OUT["g3"] = {"n_clamped": int(above.sum()), "min_cos": float(cos.min()), "clamped_norm_max_rel_err": float(((oc.norm(dim=-1) - ref).abs() / ref)[above].max()),
                 "unclamped_unchanged": bool(torch.equal(oc[~above], oa[~above])), "rescale_min_cos": float(torch.nn.functional.cosine_similarity(orr, oa, dim=-1).min()),
                 "rescale_norm_max_rel_err": float(((orr.norm(dim=-1) - ref).abs() / ref).max())}
    OUT["g3_pass"] = OUT["g3"]["n_clamped"] > 0 and OUT["g3"]["min_cos"] > 1 - 1e-6 and OUT["g3"]["clamped_norm_max_rel_err"] < 1e-5 and OUT["g3"]["unclamped_unchanged"] \
        and OUT["g3"]["rescale_min_cos"] > 1 - 1e-6 and OUT["g3"]["rescale_norm_max_rel_err"] < 1e-5
    sub = [it for n in range(3, 13) for it in [x for x in TE if x["n"] == n][:8]]; N.test_set = lambda per_n=1000: sub
    r = H.readclamp_fields(m, a, "cpu", None, bs=8, n_ref=40)
    fc = r["clamped"]["frac_clamped"]; anyc = any(v > 0 for n in ("9", "10", "11", "12") for v in fc.get(n, {}).values())
    OUT["g4"] = {"keys": sorted(r), "arms": {k: sorted(r[k]) for k in ("unclamped", "clamped", "rescaled")}, "n_items_rows": [len(r[k]["items"]) for k in ("unclamped", "clamped", "rescaled")],
                 "any_clamp_on_deep": anyc, "row0_clamped": r["clamped"]["items"][0], "indist": {k: r[k]["indist"] for k in ("unclamped", "clamped", "rescaled")},
                 "reset": DM.READ_CLAMP is None}
    OUT["g4_pass"] = anyc and all(n_ == 32 for n_ in OUT["g4"]["n_items_rows"]) and "clamped_t3_6" in r["clamped"]["items"][0] and OUT["g4"]["reset"]
    OUT["all_pass"] = bool(OUT["g1_pass"] and OUT["g2_pass"] and OUT["g3_pass"] and OUT["g4_pass"])
    json.dump(OUT, open("boxes/runs_fsq/s187_gates/gate187_readclamp.json", "w", newline="\n"), indent=1, default=str); print(json.dumps(OUT, default=str)[:3000])


if __name__ == "__main__":
    main()
