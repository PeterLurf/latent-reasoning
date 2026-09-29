"""Build the Part IV (PROTOCOL4) report from results/*.json:
    report/protocol4_report.md    (plain text, diff-able)
    report/protocol4_report.html  (self-contained page, light/dark, inline SVG charts)
    report/protocol4_report.pdf   (headless Edge/Chrome print of the HTML, if a browser is found)

    python report/build_protocol4_report.py
Every number in the tables and charts is read from the result files; the prose states the readings.
"""
import html
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def J(*p):
    f = os.path.join(ROOT, *p)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


P4 = J("results", "protocol4_confirm.json")
VAR = J("results", "protocol4_variants.json")["verdicts"]
CS = J("results", "confirm_sweep.json")
CSV = J("results", "confirm_sweep_verdicts.json")
FA = J("results", "frontier_amp.json")
FAF = J("results", "frontier_amp_fork.json")
SP = J("results", "stall_pilot.json")
SPN = J("results", "stall_pilot_nostep.json")
RJ = J("results", "rescore_gateJ.json")
BILL = J("results", "runpod_billing_2026-09-25.json")
M, V = P4["models"], P4["verdicts"]
LOCAL = {"cuda:0": "RTX 4060 (local)", "cuda:1": "RTX 3070 (local)", "cuda:2": "RTX 2070 (local)"}


def hw(run):
    h = os.path.join(ROOT, "runs", f"{run}.host.txt")
    if os.path.exists(h):
        line = open(h).readline()
        m_ = re.search(r"NVIDIA (.+?) \(UUID", line)
        return (m_.group(1) if m_ else line.strip()) + " (RunPod)"
    c = J("runs", run, "config.json") or {}
    return LOCAL.get(c.get("device"), c.get("device", "?"))


def amp(run):
    c = J("runs", run, "config.json") or {}
    a = c.get("amp", "auto")
    return "fp32" if a == "fp32" or c.get("device") == "cuda:2" else "bf16"


f2 = lambda x: "–" if x is None else f"{x:.3f}"
f1 = lambda x: "–" if x is None else f"{x:.2f}"
yes = lambda b: "✔" if b else "✘"

# ------------------------------------------------------------------ document model: list of blocks
B = []
h1 = lambda t: B.append(("h1", t))
h2 = lambda t: B.append(("h2", t))
h3 = lambda t: B.append(("h3", t))
p = lambda t: B.append(("p", t))
ul = lambda items: B.append(("ul", items))
table = lambda head, rows, note=None: B.append(("table", head, rows, note))
fig = lambda svg, cap: B.append(("fig", svg, cap))

# ------------------------------------------------------------------ charts (inline SVG, CSS-var colours)
W_, H_, PADL, PADR, PADT, PADB = 640, 300, 56, 20, 18, 46


def _axes(xs, ys, xlabel, ylabel, xticks, yticks):
    X = lambda v: PADL + (v - xs[0]) / (xs[1] - xs[0]) * (W_ - PADL - PADR)
    Y = lambda v: H_ - PADB - (v - ys[0]) / (ys[1] - ys[0]) * (H_ - PADT - PADB)
    s = []
    for t in yticks:
        s.append(f'<line x1="{PADL}" x2="{W_ - PADR}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{PADL - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    for t, lab in xticks:
        s.append(f'<text x="{X(t):.1f}" y="{H_ - PADB + 18}" class="tick" text-anchor="middle">{html.escape(str(lab))}</text>')
    s.append(f'<line x1="{PADL}" x2="{W_ - PADR}" y1="{H_ - PADB}" y2="{H_ - PADB}" class="axis"/>')
    s.append(f'<text x="{(PADL + W_ - PADR) / 2}" y="{H_ - 6}" class="label" text-anchor="middle">{xlabel}</text>')
    s.append(f'<text x="14" y="{(PADT + H_ - PADB) / 2}" class="label" text-anchor="middle" '
             f'transform="rotate(-90 14 {(PADT + H_ - PADB) / 2})">{ylabel}</text>')
    return X, Y, s


def _ref(Y, v, text, dash=True, left=False):
    x, anc = (PADL + 4, "start") if left else (W_ - PADR - 4, "end")
    return (f'<line x1="{PADL}" x2="{W_ - PADR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="ref{" dash" if dash else ""}"/>'
            f'<text x="{x}" y="{Y(v) - 5:.1f}" class="reftext" text-anchor="{anc}">{text}</text>')


def _mark(shape, x, y, cls, tip):
    t = f"<title>{html.escape(tip)}</title>"
    if shape == "sq":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" class="{cls} ring">{t}</rect>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="{cls} ring">{t}</circle>'


def _legend(items):
    return '<div class="legend">' + "".join(
        f'<span><svg width="14" height="14" aria-hidden="true">'
        + (f'<rect x="2" y="2" width="10" height="10" rx="2" class="{c}"/>' if sh == "sq" else f'<circle cx="7" cy="7" r="5" class="{c}"/>')
        + f'</svg>{html.escape(lab)}</span>' for lab, c, sh in items) + "</div>"


def chart_extrap():
    rows = [(r, m) for r, m in M.items() if m.get("gate") and m.get("E") is not None and m.get("d7_K16") is not None]
    X, Y, s = _axes((0.5, 5.5), (0.45, 0.75), "sweep depth E reached within six latent steps",
                    "distance-7 accuracy at K′ = 16", [(e, e) for e in range(1, 6)], [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    s.append(_ref(Y, 0.60, "extrapolation threshold 0.60"))
    s.append(f'<rect x="{X(3.5):.1f}" y="{PADT}" width="{X(5.5) - X(3.5):.1f}" height="{H_ - PADT - PADB}" class="band"/>'
             f'<text x="{X(4.5):.1f}" y="{PADT + 12}" class="reftext" text-anchor="middle">E ≥ 4: finish-early account predicts &lt; 0.60</text>')
    seen = {}
    for r, m in sorted(rows):
        e = m["E"]; k = seen.get(e, 0); seen[e] = k + 1
        x = X(e) + (k - 2) * 9
        ext = m["d7_K16"] >= 0.60
        s.append(_mark("sq" if ext else "c", x, Y(m["d7_K16"]), "s1" if ext else "s2",
                       f"{r}: E {e}, d7@K'16 {m['d7_K16']:.3f}" + (" (extrapolates)" if ext else "")))
        if ext:
            s.append(f'<text x="{x + 8:.1f}" y="{Y(m["d7_K16"]) + 4:.1f}" class="dlabel">{r}</text>')
    svg = f'<svg viewBox="0 0 {W_} {H_}" role="img" aria-label="Sweep depth E against distance-7 accuracy at K′=16">{"".join(s)}</svg>'
    return _legend([("extrapolates (≥ 0.60)", "s1", "sq"), ("does not", "s2", "c")]) + svg


def chart_fork():
    rows = []
    for r, pm in CSV["per_model"].items():
        mr = pm.get("P25", {}).get("mean_ratio")
        if mr and mr.get("2") is not None and mr.get("3") is not None:
            rows.append((r.replace("runs/", ""), (mr["2"] + mr["3"]) / 2, r in ("runs/r3_pc_gated_gpt2", "runs/main_s1", "runs/main_s2", "runs/main_s3")))
    rows.sort(key=lambda t: t[1])
    n = len(rows)
    X, Y, s = _axes((-0.5, n - 0.5), (0.3, 1.4), "model (sorted)", "fork / chain per-edge attention",
                    [], [0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    s.append(_ref(Y, 0.5, "per-edge budget ≈ 0.5", dash=False))
    s.append(_ref(Y, 0.70, "pass ≤ 0.70", left=True))
    s.append(_ref(Y, 1.0, "per-branch budget ≈ 1.0", dash=False, left=True))
    for i, (r, v, expl) in enumerate(rows):
        s.append(_mark("sq" if expl else "c", X(i), Y(min(v, 1.38)), "s2" if expl else "s1",
                       f"{r}: mean of depths 2–3 = {v:.2f}" + (" (exploratory set)" if expl else "")))
        s.append(f'<text x="{X(i):.1f}" y="{H_ - PADB + 12}" class="tick small" text-anchor="end" '
                 f'transform="rotate(-50 {X(i):.1f} {H_ - PADB + 12})">{html.escape(r)}</text>')
    svg = f'<svg viewBox="0 0 {W_} {H_ + 60}" role="img" aria-label="Fork to chain attention ratio per model">{"".join(s)}</svg>'
    svg = svg.replace(f'y="{H_ - 6}" class="label"', f'y="{H_ + 54}" class="label"')
    return _legend([("primary set", "s1", "c"), ("s0–s3 (explored before addendum E)", "s2", "sq")]) + svg


def chart_stall():
    cfgs = ["base", "wu1000", "lr5e-4", "lr5e-4_wu1000"]
    def rate(d, pre):
        out = {}
        for c in cfgs:
            v = [x for k, x in d.items() if x["config"] == c]
            out[c] = (sum(x["stalled"] for x in v), len(v))
        return out
    a, b = rate(SP, ""), rate(SPN, "nostep_")
    X, Y, s = _axes((-0.5, 3.5), (0, 0.6), "configuration (stopped at stage-1 advance or step 4000)", "share of runs that stalled",
                    [(i, c) for i, c in enumerate(cfgs)], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    bw = 34
    for i, c in enumerate(cfgs):
        for j, (d, cls, lab) in enumerate(((a, "s1", "with step embedding"), (b, "s2", "no step embedding"))):
            k, n = d[c]; v = k / n if n else 0
            x0 = X(i) - bw - 1 + j * (bw + 2)
            y0, hgt = Y(v), max(Y(0) - Y(v), 0)
            s.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw}" height="{max(hgt, 1.5):.1f}" rx="3" class="{cls}">'
                     f"<title>{lab}, {c}: {k}/{n} stalled</title></rect>")
            s.append(f'<text x="{x0 + bw / 2:.1f}" y="{y0 - 5:.1f}" class="dlabel" text-anchor="middle">{k}/{n}</text>')
    svg = f'<svg viewBox="0 0 {W_} {H_}" role="img" aria-label="Stage-1 stall share by configuration">{"".join(s)}</svg>'
    return _legend([("with step embedding (seeds 0–2)", "s1", "sq"), ("no step embedding (seeds 10–15)", "s2", "sq")]) + svg


# ------------------------------------------------------------------ content
h1("Part IV — PROTOCOL4: the stop, the budget, and the sweep's attention budget")
p("Compiled 2026-09-25 from `results/` (build: `report/build_protocol4_report.py`). Pre-registration: "
  "`docs/PROTOCOL4.md` addenda A–E; chronology and every deviation: `docs/RESEARCH_LOG.md`; running scorecard: "
  "`docs/PROTOCOL4_RESULTS.md`. Parts I–III: `report/latent_branch_report.html`.")

h2("1. Summary")
ext = [r for r, m in M.items() if m.get("gate") and m.get("d7_K16") is not None and m["d7_K16"] >= 0.60]
comp = [r for r, m in M.items() if m.get("gate")]
trained = [r for r, m in M.items() if m.get("trained")]
ul([
    f"**Neither leading account of the failure to extrapolate survived.** Taking the step index away (P19, P21) or "
    f"randomising the budget (P22) did not make models extrapolate, and finishing the sweep inside the budget (P20, "
    f"addendum G) did not stop them. {len(ext)} of {len(comp)} competent runs extrapolate at K′ = 16 "
    f"({', '.join(ext)}), with no consistent relation to the manipulated factors or to sweep depth E.",
    "**Mechanism, refined:** the sweep's layer-1 attention budget is split among frontier *edges*, not branches. "
    "At fixed width, doubling the frontier halves per-edge attention (P25: 14/18 primary models, 4/4 exploratory "
    "models on fresh graphs). The Part III width law is the frontier-size law seen through W.",
    "**Mechanism, weakened:** the single-step clock (P23, P24) holds in only about 9 of 22 competent models; the "
    "rest read a depth over several latent steps, so 'a clocked sweep in every seed' is one strategy among several.",
    f"**Reliability is the bottleneck:** {len(trained) - len(comp)} of {len(trained)} PROTOCOL4 runs "
    "failed the competence gate, almost all by stalling at curriculum stages 2–4, which the stage-1 restart rule "
    "does not cover. P21 ended unresolved at its five-seed limit for this reason.",
    f"**Operations:** 12 runs on 8 RunPod community pods (A4000 / A4500 / RTX 4000 Ada, $0.17–0.20/h), total "
    f"${BILL['totalAmount']:.2f}; everything else on three local cards; every pod deleted after checksum-verified fetch.",
])

h2("2. Questions and design")
p("Part III found that recurrent models run a clocked breadth-first sweep in layer 1 and that sweeps which "
  "finish inside the latent budget stop, while one seed and a K = 4 model followed chains beyond any trained "
  "length. PROTOCOL4 asked what keys the stop:")
ul([
    "**Halt account:** the stop is keyed to the step embedding, so removing it (P19: K = 6; P21: K = 8) or "
    "sampling K per batch (P22: K ∈ 4..8) should let competent models extrapolate.",
    "**Finish-early account** (Zhang et al. 2607.20594): a sweep that reaches full depth (E ≥ 4) inside the budget "
    "does not extrapolate (P20: K = 8; addendum G: K = 10).",
    "**Variant replications:** P13 (pre-LN feedback), P12/P17 (four layers), addendum H (wider thought).",
    "**Mechanism confirmations (addendum E):** P23 blocking is step-locked; P24 width changes amplitude, not the "
    "clock; P25 the attention budget is split per frontier edge.",
])
p("Base recipe for every run: Coconut-style recurrence, 2 layers, d = 256, batch 512, 30k steps, gated depth "
  "curriculum, annealed PARENT/CHILDREN scaffold, GPT-2 init. Gate (addendum J): overall delayed balanced "
  "accuracy ≥ 0.85 and depth-matched 5-hop ≥ 0.60. 'Extrapolates' = distance-7 accuracy ≥ 0.60 with bootstrap "
  "lower bound > 0.55. E = deepest layer-1 frontier depth reached at any step k ≤ 6.")
table(["Addendum", "Written", "Content"], [
    ["A", "before any run", "P19–P21"],
    ["B", "2026-09-24 23:40", "cloud hardware; precision held to the original cards"],
    ["C", "2026-09-25 ~00:00", "which runs train where; predictions restated"],
    ["D", "~00:45, before any evaluation", "scoring battery, contingencies, P22 (randk)"],
    ["E", "~13:40, before its code existed", "P23–P25; discloses the earlier exploratory frontier test"],
])

h2("3. Training outcomes")
rows = []
for r, m in M.items():
    if not m.get("trained"):
        continue
    rows.append([r, str(m.get("final_stage")), str(m.get("attempts")), f2(m.get("delayed")), f2(m.get("matched5")),
                 yes(m.get("gate")), "–" if m.get("E") is None else str(m["E"]), f2(m.get("d7_K12")), f2(m.get("d7_K16")),
                 hw(r), amp(r)])
for r, v in (("v1_wide_s2", VAR["H_v1_wide_s2"]), ("v2_deep_s1", VAR["P12_v2_deep_s1"]), ("v3_rawfb_s1", VAR["P13_v3_rawfb_s1"])):
    m = M.get(r) or {}
    if not any(x[0] == r for x in rows):
        rows.append([r, "", "", f2(v.get("delayed")), f2(v.get("matched5")), yes(v.get("gate")), "", "", "", hw(r), amp(r)])
table(["Run", "Final stage", "Attempts", "Delayed", "Matched-5", "Gate", "E", "d7 @K′12", "d7 @K′16", "Hardware", "Precision"], rows,
      "Attempts > 1 = stage-1 restarts under the guard. Final stage below 'full' at 30k counts as a gate failure (addendum D).")
stall_late = [r for r, m in M.items() if m.get("trained") and m.get("final_stage") not in ("full", None)]
p(f"Runs that ended below full curriculum depth: {', '.join(stall_late)}. The stage-1 restart rule cannot catch "
  "these; each cost a full 30k-step run.")

h2("4. Verdicts")
vr = []
P19 = V["P19"]; vr.append(["P19 no step embedding → extrapolates at K′=12", f"competent: {', '.join(P19['competent'])}",
                           "**not replicated**" if P19["replicated"] is False else str(P19["replicated"])])
P20 = V["P20"]["per_run"]; vr.append(["P20 K=8: E ≥ 4 and no extrapolation at K′=16",
                                      "; ".join(f"{r}: gate {yes(x['gate'])}, E {x['E']}, d7 {f2(x['d7_K16'])}" for r, x in P20.items()),
                                      "**fails**" if any(x.get("pass") is False for x in P20.values()) else "–"])
P21 = V["P21"]; vr.append(["P21 K=8 + no step embedding (decisive cell)", f"{len(P21['per_run'])} seeds, {len(P21['scorable'])} scorable",
                           f"**{P21['reading']}** (stopping rule reached)"])
P22 = V["P22"]; vr.append(["P22 randk → extrapolates at K′=16",
                           "; ".join(f"{r}: d7 {f2(x['d7_K16'])}" for r, x in P22["per_run"].items()),
                           "**fails**" if P22["pass"] is False else "passes"])
for k in ("G_budget_k10_s1", "G_budget_k10_s2", "G_budget_k10_s3"):
    x = V[k]; vr.append([f"Addendum G {k[2:]}", f"gate {yes(x['gate'])}, E {x['E']}, d7 {f2(x['d7_K16'])}", "pass" if x.get("pass") else "fail"])
x = V["IJ_budget_k4_s3"]; vr.append(["Addendum I/J budget_k4_s3", f"gate {yes(x['gate'])}", "unscored"])
x = VAR["P13_v3_rawfb_s1"]; vr.append(["P13 pre-LN feedback: β toward 1, competition persists",
                                       f"β {x['beta']:.3f} (range 0.62–0.87), competition {yes(x['competition_persists'])}", "**fails**" if not x["pass"] else "passes"])
x = VAR["P12_v2_deep_s1"]; vr.append(["P12 / P17 four layers", f"gate {yes(x['gate'])} ({f2(x['delayed'])})", "unscored"])
x = VAR["H_v1_wide_s2"]; vr.append(["Addendum H wider thought, seed 2", f"gate {yes(x['gate'])} ({f2(x['delayed'])})", "unscored"])
for key, lab in (("P23", "P23 blocking is step-locked"), ("P24", "P24 width changes amplitude, not clock"),
                 ("P25_primary", "P25 budget per frontier edge (primary)"), ("P25_replication_s0_s3", "P25 replication, s0–s3")):
    x = CSV["verdicts"][key]; vr.append([lab, f"{x['n_pass']}/{len(x['scorable'])} models pass", f"**{x['reading']}**"])
table(["Prediction", "Evidence", "Reading"], vr)

h2("5. Extrapolation is not predicted by the manipulated factors or by E")
fig(chart_extrap(), "Competent PROTOCOL4 runs. The finish-early account predicts no point above the line inside the shaded "
                    "band; the halt account predicts every no-step and randk run above the line. Neither pattern appears.")
p("Extrapolating runs differ in step embedding (present / absent), budget (K = 6, 8, 10) and architecture; their "
  "same-recipe siblings mostly do not extrapolate. Among competent runs, 3 of 8 with K ≥ 8 extrapolate versus 1 of 3 "
  "with K = 6 — far too few runs to read, but a cheap pre-registration target. The working reading is that "
  "extrapolation is a seed-level outcome; comparing extrapolating and non-extrapolating siblings "
  "(nostep_k8_s0 vs s1; budget_k10_s3 vs s1/s2) is the natural next analysis.")

h2("6. The sweep's attention budget is split per frontier edge")
p("**Exploratory (before addendum E; disclosed there).** On default graphs, frontier size already varies at fixed W. "
  "A per-depth regression of log excess per-edge attention on log W and log n (frontier edges) gave b_n ≈ −0.8 and "
  "b_W ≈ −0.2 to −0.5 at depths 1–2 in the four sweeping reference models (`latent/frontier_amp.py`). A first pooled "
  "fit on raw attention had pointed the other way (b_W −1.21 on main_s1); the (W, n) table showed a clear 1/n pattern "
  "and the fit was replaced. A controlled test on fork-mode graphs (`--shapes fork`: frontier exactly W in chain "
  "graphs, 2W past the split in fork graphs) gave fork/chain ratios of 0.49–0.68 in s0–s3, with competence kept on fork "
  "graphs (0.87–0.92).")
rows = [[r.replace("runs/", ""), *[f1(v) if v is not None else "–" for v in
         [FAF[r].get(str(d), {}).get(str(w), [None])[0] for d in (1, 2, 3) for w in (2, 4, 8)]]] for r in FAF]
table(["Model", "d1 W2", "d1 W4", "d1 W8", "d2 W2", "d2 W4", "d2 W8", "d3 W2", "d3 W4", "d3 W8"], rows,
      "Exploratory fork/chain per-edge excess ratios (results/frontier_amp_fork.json). Per-edge budget → 0.5; per-branch → 1.0.")
p("**Confirmation (addendum E, held-out confirm families, fresh seeds):**")
fig(chart_fork(), "P25: mean fork/chain ratio at depths 2–3 per model. The pass rule (≤ 0.70) applies to each depth "
               "separately, so budget_k8_s1 (0.74 at depth 2) and nostep_k8_s1 (0.85 at depth 3) fail although their means sit near the line.")

h2("7. The step-locked clock holds in a subset of models")
rows = []
for r, pm in CSV["per_model"].items():
    a, b, c = pm["P23"], pm["P24"], pm.get("P25", {})
    mr = c.get("mean_ratio") or {}
    rows.append([r.replace("runs/", ""), str(pm["clock"]), yes(a["diag"]), yes(a["off"]), yes(a["pass"]),
                 yes(b["same_clock"]), f1(b["edge_W8_W2"]), f1(b["total_W8_W2"]), yes(b["pass"]),
                 f1(mr.get("2")), f1(mr.get("3")), f1(c.get("fork_bal_acc")), yes(c.get("pass"))])
table(["Model", "Clock k*(1..3)", "P23 diag", "P23 off", "P23", "P24 same clock", "edge W8/W2", "total W8/W2", "P24",
       "P25 d2", "P25 d3", "fork acc", "P25"], rows,
      "results/confirm_sweep_verdicts.json. P23 diag: Δ ≤ −0.05 at k*(x); off: |Δ| ≤ 0.02 at |k − k*| ≥ 2. "
      "P24 bands: edge 0.10–0.45, total 0.6–1.6.")
p("The width amplitude part of P24 holds in all 22 models (per-edge W8/W2 0.13–0.32). What fails is the clock: in "
  "13 models blocking two or more steps away from the argmax still moves the outcome (up to 0.24), and in 12 the "
  "argmax step shifts with W. Post hoc (exploratory): models split into regular clocks (one step per depth, the same "
  "at every width: s0, s1, s3, v2_deep_s0, randk_s0, budget_k10_s1, nostep_s4) and irregular ones (clocks like "
  "[1,5,6], [1,3,8], [2,7,10]) that read a depth over several steps. The frozen qualifying criterion (argmax "
  "≥ 1.5× decoy) admitted all 22 models, so single-step predictions were scored on multi-step readers. The readings "
  "stand as scored; a retest needs a stricter criterion fixed in a new addendum.")

h2("8. Training reliability")
fig(chart_stall(), "Stall pilots (TRAINING_TODO Phase 0). Each run stops at its stage-1 advance or at step 4000 with "
                   "the real 30k-step schedule.")
p("Stage-1 stalls come with removing the step embedding: 3/6 at the base learning rate, 1/6 with warmup 1000, 0/6 "
  "with lr 5e-4 (0/12 with either lower-lr config; one-sided Fisher p ≈ 0.025 against base), and 0/12 on the "
  "recipe with a step embedding. The pilot does not address the late-stage stalls behind most gate failures (§3). "
  "Per TRAINING_TODO, a changed recipe is for runs that answer new questions, not replications.")

h2("9. Addendum-J re-score of earlier models (sensitivity)")
rows = [[t, v["run"], f2(v["delayed"]), f2(v["matched5"]), yes(v["gate_old"]), yes(v["gate_J"])] for t, v in RJ["models"].items()]
table(["Tag", "Run", "Delayed", "Matched-5", "Old gate", "J gate"], rows, RJ["note"])

h2("10. Deviations and disclosures")
ul([
    "The frontier-size-versus-width test ran on discovery graphs before its addendum; addendum E discloses it and "
    "treats s0–s3 as replications only.",
    "randk_s1 ended at curriculum stage 4. The first version of `latent/protocol4_confirm.py` passed it on accuracy "
    "alone; the scorer now applies the addendum-D rule (below full depth = gate failure), which removes randk_s1 from "
    "the competent set and leaves P22 unchanged (PROTOCOL5 plan step 1, 2026-09-25).",
    "The first fork design (shape drawn per branch) tied frontier size more tightly to W than default graphs; it was "
    "replaced by a per-graph shape before any use.",
    "Local runs died at 00:42 with the assistant session that launched them; resumed from checkpoints (exact resume, "
    "logs truncated to the checkpoint). Duplicate evaluation queues ran on three pods for ~15 min; results were "
    "unaffected (outputs are skip-if-valid).",
    "Cloud runs used torch 2.14.0+cu126 on driver-550/580.119 hosts; hosts with driver 580.178 failed cuInit and were "
    "discarded unused (addendum C).",
])

h2("11. Next steps, by value")
ul([
    "Fix late-stage stalls before training more seeds: a stage 2–4 restart rule, or the lr 5e-4 recipe for new-question runs.",
    "Treat extrapolation as seed-level: compare extrapolating and non-extrapolating siblings mechanistically.",
    "Re-test P23/P24 under a new addendum with a strict single-step-reader criterion fixed before blocking.",
    "Phase 4 (Coconut GPT-2 on ProsQA) — the only item needing a rented 24 GB card; only after the reliability fix.",
])

h2("12. Reproduce")
ul([
    "Score: `python -m latent.protocol4_confirm`, `PYTHONPATH=. python scratch/score_p4_variants.py`, "
    "`python -m latent.confirm_sweep run && python -m latent.confirm_sweep score`, `PYTHONPATH=. python scratch/rescore_gateJ.py`.",
    "Frontier: `python -m latent.frontier_amp <runs> --out results/frontier_amp.json`, `PYTHONPATH=. python scratch/frontier_fork_ratio.py`.",
    "Pilots: `python scratch/stall_pilot.py <device>` and `... <device> - nostep 10,11,12,13,14,15 <configs> --K 6 --no_step_emb`.",
    "Training queues: `scratch/local_resume.sh` (local, launch detached), `scratch/pod_setup.sh` / `pod_after_deploy.sh` / "
    "`pod_reap.sh` / `pod_fetch.sh` (RunPod over the SSH proxy).",
    "This report: `python report/build_protocol4_report.py`.",
])

# ------------------------------------------------------------------ renderers
def md_inline(t):
    return t


def html_inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


md, hb = [], []
for b in B:
    k = b[0]
    if k in ("h1", "h2", "h3"):
        lvl = int(k[1]); md.append("#" * lvl + " " + b[1] + "\n"); hb.append(f"<{k}>{html_inline(b[1])}</{k}>")
    elif k == "p":
        md.append(b[1] + "\n"); hb.append(f"<p>{html_inline(b[1])}</p>")
    elif k == "ul":
        md.append("\n".join("* " + i for i in b[1]) + "\n"); hb.append("<ul>" + "".join(f"<li>{html_inline(i)}</li>" for i in b[1]) + "</ul>")
    elif k == "table":
        head, rows, note = b[1], b[2], b[3]
        md.append("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")
        if note:
            md.append("_" + note + "_\n")
        hb.append('<div class="tw"><table><thead><tr>' + "".join(f"<th>{html_inline(h)}</th>" for h in head) + "</tr></thead><tbody>"
                  + "".join("<tr>" + "".join(f"<td>{html_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
                  + (f'<p class="note">{html_inline(note)}</p>' if note else ""))
    elif k == "fig":
        md.append(f"_[Figure: {b[2]} — see protocol4_report.html]_\n")
        hb.append(f'<figure>{b[1]}<figcaption>{html_inline(b[2])}</figcaption></figure>')

open(os.path.join(HERE, "protocol4_report.md"), "w", encoding="utf-8").write("\n".join(md))

CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#8a8984;--line:#e4e3de;--card:#f4f3ef;
--s1:#2a78d6;--s2:#eb6834;--band:rgba(42,120,214,.07)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--fg:#fff;--fg2:#c3c2b7;
--muted:#8f8e86;--line:#34332f;--card:#232322;--s1:#3987e5;--s2:#d95926;--band:rgba(57,135,229,.12)}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#8f8e86;--line:#34332f;--card:#232322;
--s1:#3987e5;--s2:#d95926;--band:rgba(57,135,229,.12)}
body{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.7rem;line-height:1.25;margin:0 0 12px}h2{font-size:1.25rem;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--line)}
p,li{color:var(--fg)}code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.88em}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;font-size:.84rem;min-width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
th{color:var(--fg2);font-weight:600;background:var(--card);position:sticky;top:0}
.note{color:var(--fg2);font-size:.82rem;margin-top:4px}
figure{margin:18px 0;background:var(--card);border-radius:10px;padding:14px}figcaption{color:var(--fg2);font-size:.85rem;margin-top:6px}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.axis{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--fg2);font-size:11px}.tick.small{font-size:9.5px}.label{fill:var(--fg2);font-size:12px}
.ref{stroke:var(--muted);stroke-width:1.2}.ref.dash{stroke-dasharray:5 4}.reftext{fill:var(--fg2);font-size:10.5px}
.band{fill:var(--band)}.dlabel{fill:var(--fg);font-size:10.5px}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.ring{stroke:var(--card);stroke-width:2}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:.85rem;color:var(--fg2);margin-bottom:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}.legend svg{width:14px;height:14px;display:inline-block;flex:none}
@media print{h2{break-after:avoid}figure,table{break-inside:avoid}th{position:static}}
"""
page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>PROTOCOL4 Report</title><style>' + CSS + "</style></head><body><main>" + "\n".join(hb) + "</main></body></html>")
out_html = os.path.join(HERE, "protocol4_report.html")
open(out_html, "w", encoding="utf-8").write(page)

for br in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
    if os.path.exists(br):
        pdf = os.path.join(HERE, "protocol4_report.pdf")
        subprocess.run([br, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        "file:///" + out_html.replace("\\", "/")], capture_output=True, timeout=120)
        print("pdf:", os.path.exists(pdf) and os.path.getsize(pdf))
        break
print("wrote", out_html)
