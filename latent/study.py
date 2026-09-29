"""Stages B-D: discovery (probes, site selection) -> frozen protocol -> confirmation.

    python -m latent.study discover runs/main_s0 --device cuda:0
    python -m latent.study confirm  runs/main_s0 --protocol docs/protocol_frozen.json --device cuda:0

Discovery uses only 'discovery'-split families; confirmation only 'confirm'-split
families. Everything selected in discovery (site k*, subspace kinds, rank, donor
rules) is written to the protocol file before any confirmation graph is generated.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .competence import load
from .graphs import NPB, WMAX, sample_graphs
from .interventions import (bootstrap_ci, control_effects, effect_matrix, summarize,
                            upfront_effects)
from .model import LatentReasoner
from .probes import apply_probe, branch_auc, collect, fit_probe, labels, matched_auc


def _probes_all_steps(model, K, g_fit, device, kind="reach"):
    H = collect(model, g_fit, K, device)
    Y, M = labels(g_fit, kind)
    return [fit_probe(H[:, k], Y, M, device=device) for k in range(K + 1)], H


def _save_probes(probes, path):
    torch.save([{k: v.cpu() for k, v in p.items()} for p in probes], path)


def _load_probes(path, device):
    return [{k: v.to(device) for k, v in p.items()} for p in torch.load(path, weights_only=False)]


def decodability(model, probes, g, K, device):
    """Matched AUC (reachable vs shape-matched decoys) per step, overall and per width."""
    H = collect(model, g, K, device)
    out = {}
    for k in range(K + 1):
        s = apply_probe(probes[k], H[:, k]).cpu().numpy()
        overall, per_depth = matched_auc(s, g)
        by_w = {}
        for w in np.unique(g.width):
            idx = np.nonzero(g.width == w)[0]
            by_w[int(w)] = matched_auc(s[idx], g.subset(idx))[0]
        ba = branch_auc(s, g)
        out[k] = {"matched_auc": overall, "by_bdepth": per_depth, "by_width": by_w,
                  "branch_auc_by_width": {int(w): float(np.nanmean(ba[g.width == w])) for w in np.unique(g.width)}}
    return out


def discover(run, device, n_fit=3000, n_val=1500, out_dir=None):
    t0 = time.time()
    model, c = load(run, device)
    K = c["K"]
    out_dir = out_dir or os.path.join(run, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1001)
    g_fit = sample_graphs(rng, n_fit, split="discovery")
    g_val = sample_graphs(rng, n_val, split="discovery")
    probes, H_fit = _probes_all_steps(model, K, g_fit, device)
    _save_probes(probes, os.path.join(out_dir, "probes_reach.pt"))
    rep = {"run": run, "decodability_val": decodability(model, probes, g_val, K, device)}

    # controls for decodability: label shuffle across graphs; untrained same-architecture model
    Y, M = labels(g_fit)
    perm = rng.permutation(len(g_fit))
    Hv = collect(model, g_val, K, device)
    shuf = {}
    for k in range(K + 1):
        p = fit_probe(H_fit[:, k], Y[perm], M[perm], device=device)
        shuf[k] = matched_auc(apply_probe(p, Hv[:, k]).cpu().numpy(), g_val)[0]
    rep["shuffled_label_auc"] = shuf
    torch.manual_seed(0)
    rnd = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(K, 8), mode=c["mode"],
                         emb=c.get("emb", "slots"), d_id=c.get("d_id", 64), init=c.get("init", "torch")).to(device).eval()
    rp, _ = _probes_all_steps(rnd, K, g_fit, device)
    rep["untrained_model_auc"] = {k: v["matched_auc"] for k, v in decodability(rnd, rp, g_val, K, device).items()}

    # site selection (exploratory, discovery graphs only): full-vector donor patch at W=4
    g4 = sample_graphs(rng, 300, widths=[4], split="discovery")
    prof = {}
    for k in range(K):
        s = summarize(effect_matrix(model, g4, K, k, kind="full", device=device))
        prof[k] = {m_: bootstrap_ci(v) for m_, v in s.items() if m_ != "matrix"}
        s2 = summarize(effect_matrix(model, g4, K, k, kind="probe", probe=probes[k], device=device))
        prof[k]["probe_targeted"] = bootstrap_ci(s2["targeted"])
        prof[k]["probe_collateral_attached"] = bootstrap_ci(s2["collateral_attached"])
        print(f"  k={k} full: tgt {prof[k]['targeted'][0]:.3f} col {prof[k]['collateral_attached'][0]:.3f} | "
              f"probe: tgt {prof[k]['probe_targeted'][0]:.3f} col {prof[k]['probe_collateral_attached'][0]:.3f}", flush=True)
    rep["site_profile_W4"] = prof
    score = {k: prof[k]["targeted"][0] - prof[k]["collateral_attached"][0] for k in prof}
    rep["k_star"] = int(max(score, key=score.get))
    rep["sec"] = time.time() - t0
    json.dump(rep, open(os.path.join(out_dir, "discovery.json"), "w"), indent=1)
    print(f"{run}: k* = {rep['k_star']}  ({rep['sec']:.0f}s)")
    return rep


@torch.no_grad()
def fit_pca_bases(model, K, device, rng, n=300, widths=(2, 4, 8), rmax=40):
    """Top-rmax principal directions of single-branch-removal donor differences, per step.

    Discovery split only. Differences are pooled over widths and (uncentred) SVD'd so
    the mean difference direction is included.
    """
    from .graphs import move_branch
    from .interventions import run_thoughts
    diffs = {k: [] for k in range(K)}
    for W in widths:
        g = sample_graphs(rng, n, widths=[W], split="discovery")
        att = np.stack([np.nonzero(a)[0] for a in g.attached])
        h = run_thoughts(model, g, K, device=device)
        for w in range(W):
            hd = run_thoughts(model, move_branch(g, att[:, w], False), K, device=device)
            for k in range(K):
                diffs[k].append((hd[:, k] - h[:, k]).float())
    bases = {}
    for k in range(K):
        D = torch.cat(diffs[k])
        _, S, Vh = torch.linalg.svd(D, full_matrices=False)
        bases[k] = Vh[:rmax].T.contiguous().cpu()                  # (d, rmax)
    return bases


def discover_subspace(run, device, ranks=(6, 20, 40)):
    """Discovery step 2: data-driven low-rank subspace candidates (exploratory, logged)."""
    model, c = load(run, device)
    K = c["K"]
    out_dir = os.path.join(run, "discovery")
    rng = np.random.default_rng(2002)
    bases = fit_pca_bases(model, K, device, rng)
    torch.save(bases, os.path.join(out_dir, "pca_bases.pt"))
    # Haufe activation patterns of the reach probes: A_k = Cov(h_k) W_raw,k  (discovery thoughts)
    from .probes import probe_directions
    probes = _load_probes(os.path.join(out_dir, "probes_reach.pt"), device)
    Hc = collect(model, sample_graphs(rng, 3000, split="discovery"), K, device)
    patterns = [torch.cov(Hc[:, k].T) @ probe_directions(probes[k]) for k in range(K + 1)]
    torch.save([p_.cpu() for p_ in patterns], os.path.join(out_dir, "patterns.pt"))
    g4 = sample_graphs(rng, 300, widths=[4], split="discovery")      # fresh discovery graphs
    rep = json.load(open(os.path.join(out_dir, "discovery.json")))
    prof = {}
    for k in range(K):
        full = summarize(effect_matrix(model, g4, K, k, kind="full", device=device))
        prof[k] = {"full": {m_: bootstrap_ci(v)[0] for m_, v in full.items() if m_ != "matrix"}}
        for r in ranks:
            s = summarize(effect_matrix(model, g4, K, k, kind="pca", rank=r, pca=bases, device=device))
            sr = summarize(effect_matrix(model, g4, K, k, kind="random", rank=r, device=device))
            prof[k][f"pca{r}"] = {m_: bootstrap_ci(v)[0] for m_, v in s.items() if m_ != "matrix"}
            prof[k][f"random{r}"] = {m_: bootstrap_ci(v)[0] for m_, v in sr.items() if m_ != "matrix"}
        line = " | ".join(f"{n}: {v['targeted']:+.3f}/{v['collateral_attached']:.3f}" for n, v in prof[k].items())
        print(f"  k={k} (targeted/collateral) {line}", flush=True)
    rep["subspace_profile_W4"] = prof
    json.dump(rep, open(os.path.join(out_dir, "discovery.json"), "w"), indent=1)
    return prof


def confirm(run, protocol, device, out_dir=None):
    t0 = time.time()
    P = json.load(open(protocol))
    model, c = load(run, device)
    K = c["K"]
    k_star = P["k_star_by_run"].get(run, P["k_star_default"])
    probes = _load_probes(os.path.join(run, "discovery", "probes_reach.pt"), device)
    patterns = torch.load(os.path.join(run, "discovery", "patterns.pt"))
    for k_, p_ in enumerate(probes):
        p_["pattern"] = patterns[k_].to(device)
    pca = torch.load(os.path.join(run, "discovery", "pca_bases.pt"))
    out_dir = out_dir or os.path.join(run, "confirm")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(P["seed"])
    res = {"run": run, "k_star": k_star, "protocol": P}
    split = P.get("split", "confirm")      # smoke tests pass "discovery"

    # decodability on confirm families
    gdec = sample_graphs(rng, P["n_decode"], split=split)
    res["decodability"] = decodability(model, probes, gdec, K, device)

    # branch-effect matrices across widths
    res["effects"] = {}
    for W in P["widths"]:
        gW = sample_graphs(rng, P["n_per_width"], widths=[W], split=split)
        for cond in P["conditions"]:
            if "widths" in cond and W not in cond["widths"]:
                continue
            for k in cond.get("steps", [k_star]):
                r = effect_matrix(model, gW, K, k, kind=cond["kind"], probe=probes[k],
                                  direction=cond["direction"], donor_mode=cond.get("donor", "hub"),
                                  device=device, seed=P["seed"] + W, rank=cond.get("rank"), pca=pca)
                kname = cond["kind"] + (str(cond["rank"]) if cond.get("rank") else "")
                key = f"W{W}|k{k}|{kname}|{cond['direction']}|{cond.get('donor', 'hub')}"
                s = summarize(r)
                sd = {f"targeted_bdepth{j}": summarize(r, min_bdepth=j)["targeted"] for j in (0, 2, 4)}
                # depth-resolved targeted effect (exactly depth j)
                exact = {}
                for j in range(NPB):
                    rr = dict(r)
                    msk = r["bdepth"] == j
                    base = np.where(msk[:, None, :], r["base"], np.nan)
                    pat = np.where(msk[:, None, :], r["patched"], np.nan)
                    rr["base"], rr["patched"] = base, pat
                    with np.errstate(all="ignore"):
                        exact[j] = bootstrap_ci(_targeted_nan(rr))
                res["effects"][key] = {
                    **{m_: bootstrap_ci(v) for m_, v in s.items() if m_ != "matrix"},
                    **{m_: bootstrap_ci(v) for m_, v in sd.items()},
                    "targeted_exact_bdepth": exact,
                    "matrix_mean": np.nanmean(s["matrix"], 0).tolist(),
                    "stats": r["stats"]}
                e = res["effects"][key]
                print(f"  {key}: tgt {e['targeted'][0]:+.3f} [{e['targeted'][1]:+.3f},{e['targeted'][2]:+.3f}] "
                      f"col {e['collateral_attached'][0]:.3f} dec {e['collateral_decoy'][0]:.3f}", flush=True)
                if cond.get("save_raw"):
                    np.savez_compressed(os.path.join(out_dir, f"raw_{key.replace('|', '_')}.npz"),
                                        base=r["base"], patched=r["patched"], q_branch=r["q_branch"],
                                        bdepth=r["bdepth"], att_idx=r["att_idx"])
        # negative controls
        for ctl in P["controls"]:
            if ctl == "same_status" and (NPB and (10 - W) < 2):
                continue
            v, dn = control_effects(model, gW, K, k_star, ctl, device=device, seed=P["seed"])
            res["effects"][f"W{W}|k{k_star}|control|{ctl}"] = {"mean_abs_dP": bootstrap_ci(v), "diff_norm": dn}
        # conventional (upfront) format
        for kind in P["upfront_kinds"]:
            u = upfront_effects(model, gW, K, k_star, kind=kind, probe=probes[k_star], device=device,
                                seed=P["seed"] + W, pca=pca)
            N = len(gW)
            diag = u["base"] - u["patched"][np.arange(N), u["slot_t"]]
            off = np.array([np.abs(u["base"][n] - np.delete(u["patched"][n], u["slot_t"][n])).mean()
                            if W > 1 else np.nan for n in range(N)])
            res["effects"][f"W{W}|k{k_star}|upfront|{kind}"] = {
                "targeted": bootstrap_ci(diag), "collateral": bootstrap_ci(off) if W > 1 else None,
                "base_acc": float((u["base"] > 0.5).mean())}
    # downstream representation trace for the primary condition
    res["downstream"] = {}
    for W in P.get("trace_widths", []):
        gW = sample_graphs(rng, P["n_trace"], widths=[W], split=split)
        for kind in ("full", "pattern"):
            res["downstream"][f"W{W}|{kind}"] = downstream_trace(model, probes, gW, K, k_star, kind)
    res["sec"] = time.time() - t0
    json.dump(res, open(os.path.join(out_dir, "confirm.json"), "w"), indent=1)
    print(f"{run}: confirmation done in {res['sec']:.0f}s")
    return res


@torch.no_grad()
def downstream_trace(model, probes, g, K, k, kind="full"):
    """After editing branch i at thought k, does branch i stay readable in later thoughts?

    Returns, per later step k' > k, the mean probe logit of edited-branch nodes minus the
    mean over same-graph decoy nodes, for (unpatched recipient, patched, donor).
    """
    dev = next(model.parameters()).device
    r = effect_matrix(model, g, K, k, kind=kind, probe=probes[k], device=dev, keep_thoughts=True)
    th_p = torch.as_tensor(r["thoughts"], device=dev)             # (N, W, K+1, d)
    from .interventions import run_thoughts
    th_r = run_thoughts(model, g, K, device=str(dev))              # (N, K+1, d)
    N, W = th_p.shape[:2]
    bi = np.arange(N)
    out = {}
    for kp in range(k + 1, K + 1):
        s_rec = apply_probe(probes[kp], th_r[:, kp]).cpu().numpy()   # (N, V)
        rows_p, rows_r = [], []
        for w in range(W):
            br = r["att_idx"][:, w]
            s_pat = apply_probe(probes[kp], th_p[:, w, kp]).cpu().numpy()
            ids = g.nodes[bi, br]                                     # (N, NPB)
            dec = np.where(~g.attached[:, :, None], g.nodes, -1).reshape(N, -1)
            for n in range(N):
                d_ids = dec[n][dec[n] >= 0]
                rows_p.append(s_pat[n, ids[n]].mean() - s_pat[n, d_ids].mean())
                rows_r.append(s_rec[n, ids[n]].mean() - s_rec[n, d_ids].mean())
        out[kp] = {"recipient_margin": bootstrap_ci(np.array(rows_r)),
                   "patched_margin": bootstrap_ci(np.array(rows_p))}
    return out


def _targeted_nan(r):
    base, pat = r["base"], r["patched"]
    N, W, _ = pat.shape
    sign = 1.0 if r["direction"] == "remove" else -1.0
    out = np.full((N, W), np.nan)
    for n in range(N):
        for w in range(W):
            m = r["q_branch"][n] == r["att_idx"][n, w]
            d = base[n, w, m] - pat[n, w, m]
            d = d[~np.isnan(d)]
            if len(d):
                out[n, w] = sign * d.mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["discover", "subspace", "confirm"])
    p.add_argument("run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--protocol", default="docs/protocol_frozen.json")
    a = p.parse_args()
    if a.stage == "discover":
        discover(a.run, a.device)
    elif a.stage == "subspace":
        discover_subspace(a.run, a.device)
    else:
        confirm(a.run, a.protocol, a.device)


if __name__ == "__main__":
    main()

