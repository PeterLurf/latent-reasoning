"""Local diagnostic for gate_s269 step0's perm_d3 'h' mismatch (CPU): per pass, max |h_ctrl - h_nk|, the positions / tokens where they differ, whether those are
padding or name positions, and the same comparison CTRL vs CTRL (a second identical forward) for the CPU run-to-run floor.
    LATENT_ROOT=... G269_DEV=cpu python -B scratch/s269_build/diag_step0_h.py"""
import json
import os
import sys

ROOT = os.environ["LATENT_ROOT"]; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import torch  # noqa: E402
import gate_s269 as G  # noqa: E402
import scratch.s222_det_wrap as DW  # noqa: E402
from boxes.inject_st import cue_inputs  # noqa: E402
from boxes.nk import NAME_IDS  # noqa: E402

DW.apply(); M = G._models(); a, mc, _ = M[0]; _, mn, _ = M[1]; K = a["think_k"]; its = [x for x in G._items(a) if x[0] == "perm_d3"][0][2]; out = []
with torch.no_grad():
    xs, msk = cue_inputs(mc, its, G.DEV); Sc = mc.mem.zero_state(len(its), G.DEV); Sn = mn.mem.zero_state(len(its), G.DEV); Sc2 = mc.mem.zero_state(len(its), G.DEV)
    names = torch.tensor(NAME_IDS)
    for t in range(1, K + 1):
        j = min(t, len(xs)) - 1; x = xs[j]
        Sc, hc, _ = mc.pass_aw(Sc, x, msk, mc.lower(x)); Sn, hn, _ = mn.pass_aw(Sn, x, msk, mn.lower(x)); Sc2, hc2, _ = mc.pass_aw(Sc2, x, msk, mc.lower(x))
        df = (hc - hn).abs().amax(-1); dd = (hc - hc2).abs().amax(-1); bad = (df > 0).nonzero().tolist()
        out.append({"t": t, "max_ctrl_nk": float(df.max()), "n_pos_ctrl_nk": len(bad), "max_ctrl_ctrl": float(dd.max()), "n_pos_ctrl_ctrl": int((dd > 0).sum()),
                    "pos": [(b, p, int(x[b, p]), bool(msk[b, p]), bool(torch.isin(x[b, p], names))) for b, p in bad[:8]],
                    "nan_nk": bool(torch.isnan(hn).any()), "S12_equal": bool(torch.equal(Sc, Sn[:, :mn.NB]))})
json.dump(out, open(ROOT + "/scratch/_tmp/s269g_local/diag_step0_h.json", "w"), indent=1); [print(o) for o in out]
