"""Aggregate competence + confirmation outputs across seeds into results/summary.json.

    python -m latent.aggregate
"""
from __future__ import annotations

import json
import os

import numpy as np

MAIN = {"s0": "runs/r3_pc_gated_gpt2", "s1": "runs/main_s1", "s2": "runs/main_s2"}
CONTROLS = {"pause": "runs/pause_s0", "k0": "runs/k0_s0", "pause1": "runs/pause_s1", "pause2": "runs/pause_s2",
            "k0_1": "runs/k0_s1", "k0_2": "runs/k0_s2"}


def followup_block():
    """Fresh-confirmation follow-up results for the report (docs/protocol2_frozen.json)."""
    C = _load("results/followup_confirm.json")
    if C is None:
        return None
    fb = {"confirm": C, "cache_margins": {}, "attention_doses": {}, "dose_curves": {}}
    for s, run in MAIN.items():
        S = _load(os.path.join(run, "followup", "fresh", "summary.json"))
        if S is None:
            continue
        fb["cache_margins"][s] = {W: {c: R["cache"]["conds"][c]["margin_by_step"] for c in ("base", "T", "TL", "clamp", "L")}
                                  for W, R in S["results"].items()}
        fb["attention_doses"][s] = {W: {b: {k: v[0] for k, v in e.items()} for b, e in R["attn"]["doses"].items()}
                                    for W, R in S["results"].items()}
        fb["dose_curves"][s] = {W: {d: {kind: {lam: {"targeted": e["targeted"], "collateral": e["collateral_mean"][0],
                                                      "norm": e["perturb_norm"]} for lam, e in kd["doses"].items()}
                                         for kind, kd in R[f"dose_{d}"]["kinds"].items()} for d in ("remove", "insert")}
                                for W, R in S["results"].items()}
    fb["route_depth"] = {}
    for s, run in MAIN.items():
        S = _load(os.path.join(run, "followup", "fresh_cachedepth", "summary.json"))
        if S is not None:
            fb["route_depth"][s] = {W: {d: {"clean_gap": v["clean_gap"], **{c: v["shares"].get(c) for c in ("T", "L", "TL", "Gq", "clamp")}}
                                        for d, v in R["cache"]["by_depth"].items()} for W, R in S["results"].items()}
    for name, run in (("p0", "runs/pause_s0"), ("p1", "runs/pause_s1"), ("p2", "runs/pause_s2")):
        S = _load(os.path.join(run, "followup", "fresh", "summary.json"))
        if S is not None:
            fb["attention_doses"][name] = {W: {b: {k: v[0] for k, v in e.items()} for b, e in R["attn"]["doses"].items()}
                                           for W, R in S["results"].items()}
    return fb
WIDTHS = list(range(1, 9))


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {"seeds": {}, "competence": {}, "across_seeds": {}}
    import glob
    comp = []
    for p in sorted(glob.glob("results/competence_*.json")):
        if "matched" not in p:
            comp.extend(_load(p))
    for c in comp:
        name = next((k for k, v in {**MAIN, **CONTROLS}.items() if v == c["run"]), None)
        if name is None:            # Part III models (variants, held-out seeds) are reported via protocol3_confirm
            continue
        out["competence"][name] = {
            str(Kp): {f: {"bal_acc": b[f]["bal_acc"], "by_depth": b[f]["by_depth"], "by_width": b[f]["by_width"]}
                      for f in ("delayed", "upfront")}
            for Kp, b in c["budgets"].items()}
    for s, run in MAIN.items():
        r = _load(os.path.join(run, "confirm", "confirm.json"))
        d = _load(os.path.join(run, "discovery", "discovery.json"))
        if r is None:
            continue
        E = r["effects"]
        k = r["k_star"]
        seed = {"k_star": k, "effects": {}, "depth": {}, "steps": {}, "controls": {}, "upfront": {},
                "decode": {}, "downstream": r["downstream"],
                "discovery_site_profile": d.get("site_profile_W4") if d else None,
                "discovery_subspace_profile": d.get("subspace_profile_W4") if d else None,
                "discovery_decode": {kk: v["matched_auc"] for kk, v in d["decodability_val"].items()} if d else None,
                "shuffled_auc": d.get("shuffled_label_auc") if d else None,
                "untrained_auc": d.get("untrained_model_auc") if d else None}
        conds = ["full|remove|hub", "pattern|remove|hub", "probe|remove|hub", "pca20|remove|hub",
                 "pca6|remove|hub", "random6|remove|hub", "random20|remove|hub", "normrand|remove|hub",
                 "full|insert|hub", "pattern|insert|hub", "full|remove|graft", "pattern|remove|graft"]
        for cnd in conds:
            seed["effects"][cnd] = {W: {m: E[f"W{W}|k{k}|{cnd}"][m] for m in
                                        ("targeted", "collateral_attached", "collateral_signed", "collateral_decoy")}
                                    for W in WIDTHS if f"W{W}|k{k}|{cnd}" in E}
        for cnd in ("full|remove|hub", "pattern|remove|hub"):
            seed["depth"][cnd] = {W: E[f"W{W}|k{k}|{cnd}"]["targeted_exact_bdepth"] for W in WIDTHS}
            seed["steps"][cnd] = {W: {kk: {m: E[f"W{W}|k{kk}|{cnd}"][m] for m in ("targeted", "collateral_attached")}
                                      for kk in range(5) if f"W{W}|k{kk}|{cnd}" in E}
                                  for W in (2, 4, 8)}
            seed["depth_k0"] = {W: E[f"W{W}|k0|full|remove|hub"]["targeted_exact_bdepth"] for W in (2, 4, 8)
                                if f"W{W}|k0|full|remove|hub" in E}
        for W in WIDTHS:
            for ctl in ("identity", "same_status"):
                key = f"W{W}|k{k}|control|{ctl}"
                if key in E:
                    seed["controls"].setdefault(ctl, {})[W] = E[key]["mean_abs_dP"]
            for kind in ("full", "pattern"):
                key = f"W{W}|k{k}|upfront|{kind}"
                if key in E:
                    seed["upfront"].setdefault(kind, {})[W] = {"targeted": E[key]["targeted"],
                                                              "collateral": E[key]["collateral"],
                                                              "base_acc": E[key]["base_acc"]}
        seed["matrix"] = {cnd: {W: E[f"W{W}|k{k}|{cnd}"]["matrix_mean"] for W in (2, 4, 8)}
                          for cnd in ("full|remove|hub", "pattern|remove|hub")}
        D = r["decodability"]
        seed["decode"] = {"matched_auc_by_step": {kk: v["matched_auc"] for kk, v in D.items()},
                          "branch_auc_by_width_at_kstar": D[str(k)]["branch_auc_by_width"],
                          "branch_auc_by_width_by_step": {kk: v["branch_auc_by_width"] for kk, v in D.items()}}
        out["seeds"][s] = seed

    # across-seed summary: mean and range of point estimates
    seeds = list(out["seeds"].values())
    if seeds:
        agg = {}
        for cnd in seeds[0]["effects"]:
            agg[cnd] = {}
            for W in WIDTHS:
                vals = {m: [s_["effects"][cnd][W][m][0] for s_ in seeds if W in s_["effects"][cnd]]
                        for m in ("targeted", "collateral_attached", "collateral_signed", "collateral_decoy")}
                agg[cnd][W] = {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                                   "n_seeds": len(v)} for m, v in vals.items() if v}
        out["across_seeds"]["effects"] = agg
        out["across_seeds"]["branch_auc"] = {
            W: [s_["decode"]["branch_auc_by_width_at_kstar"][str(W)] for s_ in seeds] for W in WIDTHS}
    out["followup"] = followup_block()
    cm = {}
    for p_ in sorted(glob.glob("results/competence_matched*.json")):
        cm.update(_load(p_))
    if cm:
        names = {v: k for k, v in {**MAIN, **CONTROLS}.items()}
        out["competence_matched"] = {names.get(r, r): v for r, v in cm.items()}
    pf = _load("results/preflight.json")
    if pf:
        out["preflight"] = {k: v for k, v in pf.items() if k != "donor_audit"}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/summary.json", "w"), indent=1)
    print("wrote results/summary.json with seeds", list(out["seeds"]), "competence", list(out["competence"]))


if __name__ == "__main__":
    main()
