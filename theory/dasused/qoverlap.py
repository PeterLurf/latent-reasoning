"""DAS-USED draft (docs/THEORY_DASUSED_DRAFT_2026-10-14.md) fact check F-Q, CPU, read-only on model.pt: (1) the decoder prompt [EOT, expr, ' ='] and the
think input [EOT, expr, ' .'] give IDENTICAL residuals entering the memory block at every shared prefix position (causal attention; the memory enters
at block mem.layer), so the downstream read queries are a fixed function of the item text; (2) how much a rank-1 edit keyed at the decoder's ' =' query
(the G-use / U4 swap) is visible to later think-pass reads: per delta head, |cos(qhat_eq, qhat_j)| over think positions j, and the share of qhat_eq
inside the span of that item's think queries (99% energy). -> theory/dasused/qoverlap.json
    python theory/dasused/qoverlap.py"""
import ctypes
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("BOXES_NB", "5")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
ROOT = "D:/Latent"; sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402

RUNS = {"s185_rms_s1": "converged", "s187_nopass_s1": "converged", "s185_rms_s3": "failed", "s182_nofade_s1": "failed"}


@torch.no_grad()
def queries(m, toks):
    g = m.g; x = torch.tensor([toks]); h6 = DM.gpt_lower(g, g.embed(x), m.mem.layer)[0].float(); D = h6.shape[-1]; HA, HD = m.mem.h_add, m.mem.h_del
    q = m.mem.qkv(m.mem.ln(h6))[..., :D].view(len(toks), HA + HD, DM.DH)
    return h6, Fn.normalize(q[:, HA:], dim=-1)


def main():
    rng = np.random.default_rng(20401); items = []
    for n in range(3, 9):
        for _ in range(17):
            r, t = N.generate(rng, n); items.append(N.render(r))
    out = {"def": __doc__.split("->")[0].strip(), "n_items": len(items), "items": "N.generate rng 20401, 17 per n in 3..8", "runs": {}}
    for run, cls in RUNS.items():
        kind, m, a = H.load_model(os.path.join(ROOT, "boxes", "runs_fsq", run), "cpu"); m.eval()
        pref_max, cos_last, cos_pref, in_span, span_rank = 0.0, [], [], [], []
        for syms in items:
            th = [N.EOT] + N.ids(syms) + [N.TID[" ."]]; de = [N.EOT] + N.ids(syms) + [N.TID[" ="]]; L = len(th) - 1
            h_t, q_t = queries(m, th); h_d, q_d = queries(m, de)
            pref_max = max(pref_max, float((h_t[:L] - h_d[:L]).abs().max()))
            qe = q_d[-1]                                                    # (HD, DH) the decoder's ' =' query
            cl = (q_t[-1] * qe).sum(-1).abs(); cp = (q_t[:L] * qe[None]).sum(-1).abs().mean(0)
            cos_last.append(cl.numpy()); cos_pref.append(cp.numpy()); sh, rk = [], []
            for h in range(q_t.shape[1]):
                U, s, _ = torch.linalg.svd(q_t[:, h].T, full_matrices=False); e = (s ** 2).cumsum(0) / (s ** 2).sum(); r_ = int((e < 0.99).sum()) + 1
                sh.append(float(((U[:, :r_].T @ qe[h]) ** 2).sum())); rk.append(r_)
            in_span.append(sh); span_rank.append(rk)
        cl, cp, isp, sr = (np.array(v_) for v_ in (cos_last, cos_pref, in_span, span_rank))
        out["runs"][run] = {"class": cls, "prefix_h6_max_abs_diff": pref_max,
                            "abs_cos_eq_vs_dot_by_head": [round(float(v_), 4) for v_ in cl.mean(0)],
                            "abs_cos_eq_vs_prefix_mean_by_head": [round(float(v_), 4) for v_ in cp.mean(0)],
                            "share_qeq_in_think_query_span_by_head": [round(float(v_), 4) for v_ in isp.mean(0)],
                            "think_query_span_rank99_mean_by_head": [round(float(v_), 2) for v_ in sr.mean(0)],
                            "think_len_mean": float(np.mean([len(s_) + 2 for s_ in items]))}
        print(run, json.dumps(out["runs"][run]), flush=True)
        del m
    json.dump(out, open(os.path.join(ROOT, "theory", "dasused", "qoverlap.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
