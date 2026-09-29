// ================================================================ Part III (PROTOCOL3)
(() => {
  const P3 = DATA.P3 || {};
  const PR = P3.protocol3 || {models: {}, verdicts: {}};
  const M = PR.models || {}, V = PR.verdicts || {};
  const LBL = {s0: "seed 0", s1: "seed 1", s2: "seed 2", s3: "seed 3 (held out)", s4: "seed 4 (held out)",
               v1: "V1 wide (d = 512)", v2: "V2 deep (4 layers)", v3: "V3 raw feedback",
               v1b: "V1 wide, seed 1", k4: "budget K = 4", k10: "budget K = 10", s5: "seed 5 (held out)", k4b: "budget K = 4, seed 1", k4c: "budget K = 4, seed 2", k8: "budget K = 8", nostep: "no step embedding"};
  const RUN = t => M[t] ? M[t].run : null;
  const TAG_OF = {}; Object.keys(M).forEach(t => { TAG_OF[M[t].run] = t; });
  const ok = t => M[t] && M[t].competence && M[t].competence.delayed >= 0.85;
  const have = t => M[t] && M[t].competence;
  const pend = `<span class="chip">pending</span>`;
  const v2f = v => (v == null || Number.isNaN(v)) ? "–" : v.toFixed(2);
  const col = ["--s1", "--s2", "--s3", "--s4", "--s5"];
  const MCOL = {s0: "--s1", s1: "--s2", s2: "--s3", s3: "--s4", s4: "--s5", v1: "--s6", v2: "--s7", v3: "--s4"};   // fixed per model
  const mcol = t => css(MCOL[t] || "--neutral");
  const mdash = t => t && t.startsWith("v") ? "5 4" : "";

  // ---------------- intro + scorecard
  const scored = ["P11", "P12", "P13", "P15"].filter(p => V[p]);
  P("prose-p3").innerHTML = `<p>Part III asks whether the Part II account survives changes it should be sensitive to. It was pre-registered before any of its models was trained (<code>docs/PROTOCOL3.md</code>). Three architecture variants each change one element the account depends on: a wider thought (V1), a deeper network (V2), and recurrence through the raw residual instead of the final LayerNorm (V3). Three further seeds, trained after the predictions they are scored against were frozen, test whether Part II's predictions hold on seeds that played no part in forming them. Two exploratory findings from the same period get their own sections: in the upfront format queries never read the graph, and one seed learned a search that keeps working when run for more steps than it was trained with. That second finding was turned into a prospective prediction (P15) before the new models were evaluated.</p>
  <p>All Part III analyses reuse the frozen Part II pipeline on fresh graphs (300 per width). Training on this rig proved fragile. Two host crashes killed runs in progress, one GPU dropped off the bus at the end, and about half of all gated training attempts stalled at the first curriculum stage; those were restarted under a rule written before their evaluation (<a href="#integrity">details</a>). Several of the new runs still failed the competence gate and are reported as training failures.</p>`;
  const rows = [];
  const vRow = (id, pred, model, res) => rows.push([id, pred, model, res.txt, res.chip]);
  const P11 = V.P11, P12 = V.P12, P13 = V.P13, P15 = V.P15;
  vRow("P11", "Wide thought (d = 512): width effect persists — targeted W8/W2 ≤ 0.5 and per-branch attention excess W8/W2 ≤ 0.5", LBL.v1,
    P11 ? {txt: `targeted ${v2f(P11.targeted_W8_over_W2)}, attention ${v2f(P11.attention_W8_over_W2)}${P11.competent ? "" : " (failed competence gate)"}`, chip: chip(P11.pass)} : {txt: "training", chip: pend});
  vRow("P12", "Deep network (4 layers): query-time share at W = 4 above the reference minimum; 3-hop nodes mostly query-resolved", LBL.v2,
    P12 ? {txt: `Gq ${pc(P12.Gq_W4)} vs ref. min ${pc(P12.ref_min)}; 3-hop Gq ${pc(Math.abs(P12.Gq_d3_W4) < 0.005 ? 0 : P12.Gq_d3_W4)}${P12.competent ? "" : " (failed competence gate)"}`, chip: chip(P12.pass)} : {txt: "training", chip: pend});
  vRow("P13", "Raw-residual feedback: amplitude exponent β leaves [0.62, 0.87] (toward 1); competition persists", LBL.v3,
    P13 ? {txt: `β = ${v2f(P13.beta)}; boosting one branch lowers others: ${P13.competition_persists ? "yes" : "no"}`, chip: chip(P13.pass)} : {txt: "–", chip: pend});
  ["s3", "s4", "s5"].forEach(s => {
    const H = V[`heldout_${s}`];
    vRow({s3: "C", s4: "D", s5: "I1"}[s], "Held-out seed: every Part II prediction that held for all models (P1, P2, P3, P5, P7, P9, P14) holds again", LBL[s],
      H ? {txt: `${H.expected_scored.filter(p => H.verdicts[p]).length} of ${H.expected_scored.length} hold${H.competent ? "" : " (failed competence gate)"}`, chip: H.competent ? chip(H.expected_all_hold) : `<span class="chip">not scored</span>`} : {txt: "training", chip: pend});
  });
  vRow("P15", "A model extrapolates to unseen chain lengths iff extra steps help its depth-6 accuracy (≥ +0.05, K′ 6 → 8)", "new recurrent models",
    P15 ? {txt: `${Object.values(P15.per_model).filter(r => r.agree).length} of ${P15.n_scored} agree${P15.pending.length ? `; pending: ${P15.pending.map(t => LBL[t]).join(", ")}` : ""}${(P15.failed_gate || []).length ? `; not scored (failed competence gate): ${P15.failed_gate.map(t => LBL[t]).join(", ")}` : ""}`, chip: P15.pending.length ? pend : chip(P15.pass)} : {txt: "awaiting new models", chip: pend});
  const P16 = V.P16;
  vRow("P16", "No new model whose layer-1 sweep reaches depth 4 within six steps extrapolates", "new recurrent models",
    P16 ? {txt: Object.entries(P16.per_model).map(([t, r]) => `${LBL[t]}: E = ${r.E}, ${r.extrapolates ? "extrapolates" : "no extrapolation"}`).join("; ") + (P16.pending.length ? `; pending: ${P16.pending.map(t => LBL[t]).join(", ")}` : "") + ((P16.failed_gate || []).length ? `; not scored (failed competence gate): ${P16.failed_gate.map(t => LBL[t]).join(", ")}` : ""),
           chip: P16.pending.length && P16.pass ? pend : chip(P16.pass)} : {txt: "awaiting new models", chip: pend});
  if (P15 && !P15.pass) rows.find(r => r[0] === "P15")[4] = chip(false);
  const H = V.H_v1b;
  vRow("H", "Second wide-thought seed: targeted W8/W2 > 0.5 again → the wider thought causes the weaker decline; 0.18–0.37 → V1's run", LBL.v1b,
    H ? {txt: H.competent ? `W8/W2 = ${f2(H.targeted_W8_over_W2)} → ${H.reading}` : `failed competence gate (${pct(M.v1b && M.v1b.competence ? M.v1b.competence.delayed : NaN)} after four attempts): unresolved`, chip: H.reading ? `<span class="chip">${H.reading}</span>` : `<span class="chip">not scored</span>`} : {txt: "training", chip: pend});
  const G4 = V.G_k4, G10 = V.G_k10;
  vRow("G", "Training budget (exploratory): K = 10 should not extrapolate; K = 4 should either extrapolate or read the graph at query time", `${LBL.k4}, ${LBL.k10}`,
    {txt: [G10 ? `K = 10: ${G10.competent ? (G10.search.extrapolates ? "extrapolates" : "no extrapolation") : "failed competence gate (not scored)"}` : "K = 10: training",
           G4 ? `K = 4: ${G4.competent ? `${G4.search.extrapolates ? "extrapolates" : "no extrapolation"}, query share W4 ${pc(G4.Gq_W4)}${G4.neither ? " (neither route: against the hypothesis)" : ""}` : "failed competence gate"}` : "K = 4: training"].join("; "),
     chip: G4 && G4.competent ? chip(!G4.neither) : pend});
  const G4b = V.G_k4b;
  vRow("I2", "Second K = 4 seed: if competent, it extrapolates or reads the graph at query time", LBL.k4b,
    G4b ? {txt: G4b.competent ? `${G4b.search.extrapolates ? "extrapolates" : "no extrapolation"} (distance 7 at 12 steps ${f2(G4b.search.d7_K12)}), query share W4 ${pc(G4b.Gq_W4)}` : "failed competence gate",
           chip: G4b.competent ? chip(!G4b.neither) : `<span class="chip">not scored</span>`} : {txt: "training", chip: pend});
  table(P("tbl-p3score"), ["", "prediction", "model", "result", ""], rows, []);

  // ---------------- ablation table
  const order = ["s0", "s1", "s2", "s3", "s4", "s5", "v1", "v1b", "v2", "v3", "k4", "k4b", "k10"].filter(t => M[t] && (M[t].competence || !["v1b", "k4", "k10", "s5", "k4b"].includes(t)));
  const g = (t, f) => { try { const v = f(M[t]); return v == null ? NaN : v; } catch (e) { return NaN; } };
  const aRows = order.map(t => [LBL[t],
    have(t) ? pct(M[t].competence.delayed) : "training",
    pct(g(t, m => m.matched["6"].bal_acc)),
    f2(g(t, m => m.targeted_full["2"])), f2(g(t, m => m.targeted_full["8"])),
    f2(g(t, m => m.targeted_full["8"] / m.targeted_full["2"])),
    pc(g(t, m => m.route_shares["8"].TL)), pc(g(t, m => m.route_shares["4"].Gq)),
    f2(g(t, m => m.beta_fresh)), f2(g(t, m => m.attention.W8_over_W2)),
    (() => { const e = g(t, m => m.attention.energy_W4["probe filters (6)"]), q = g(t, m => m.attention.energy_W4["branch patterns (6)"]);
             return Number.isNaN(e) ? "–" : `${(100 * e).toFixed(1)}% / ${pc(q)}`; })()]);
  table(P("tbl-abl"), ["model", "delayed acc.", "6-hop matched", "targeted W2", "targeted W8", "W8 / W2", "latent share W8", "query share W4", "β", "attention W8/W2", "probe / pattern energy"],
    aRows, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const m3 = M.v3 || {};
  P("prose-abl").innerHTML = `<p>Each variant keeps the seed-0 recipe (scaffold, gated curriculum, 30k steps) and changes one thing. The table shows the quantities the Part II account rests on, next to the three reference seeds.</p>
  <p><b>V3, raw-residual feedback.</b> Coconut feeds the final-LayerNorm output back as the next input. V3 feeds the residual stream before that LayerNorm. It trains to ${have("v3") ? pct(m3.competence.delayed) : "–"} delayed accuracy. The Part II account attributes the per-branch amplitude law to renormalisation of the shared state, so it predicted that β would move toward 1 (per-branch shares of about 1/W no longer rescaled), while competition, which arises in attention, would stay. Both held: β = ${f2(m3.beta_fresh)}, outside the reference range 0.62–0.87, and boosting one reachable branch's attention still lowers the others (P13). Without the renormalisation, V3's branch edits are also larger (targeted ${f2(g("v3", m => m.targeted_full["2"]))} at W = 2 against ${(() => { const r = mm(["s0", "s1", "s2"].map(s => g(s, m => m.targeted_full["2"]))); return `${f2(r[1])}–${f2(r[2])}`; })()} in the reference seeds).</p>
  <p><b>V1 and V2</b> test the other two candidate limits. If the width effect came from superposition capacity, doubling the width of the thought should weaken it (P11). If it came from the prefix's limited depth of computation, extra layers should move more nodes to query-time resolution (P12).</p>
  ${P11 ? `<p><b>V1, a wider thought (d = 512).</b> The normalisation account predicted that the width effect would survive. Half of that held: per-branch attention still falls about as 1/W (W8/W2 = ${f2(P11.attention_W8_over_W2)}). The targeted effect, though, declines much less than in any d = 256 seed (W8/W2 = ${f2(P11.targeted_W8_over_W2)} against ${(() => { const r = mm(["s0", "s1", "s2", "s3"].map(s => g(s, m => m.targeted_full["8"] / m.targeted_full["2"]))); return `${f2(r[1])}–${f2(r[2])}`; })()}), so P11 fails. The direction is the one a capacity account predicted. Its other prediction, lower collateral than seed 0, does not hold (${f2(g("v1", m => m.collateral_full["8"]))} vs ${f2(g("s0", m => m.collateral_full["8"]))} at W = 8). V1 also keeps most of each branch's effect in the latent pathway at every width, like seeds 2 and 3. With one wide run we cannot say whether this is the extra width or the run.</p>` : ""}
  ${P12 ? `<p><b>V2, a deeper network (4 layers).</b> Its first run was lost at step 23 000 in the second host crash; the rerun is the most competent model of the project (${pct(g("v2", m => m.competence.delayed))} delayed, ${pct(g("v2", m => m.matched["6"].bal_acc))} at six hops, ${pct(g("v2", m => m.competence.upfront))} upfront). The prediction was that extra layers would let queries read more hops directly from the graph. The opposite happened. V2's query-time share at W = 4 is ${pc(P12.Gq_W4)}, below every reference seed (minimum ${pc(P12.ref_min)}), and at three hops it is ${pc(Math.abs(P12.Gq_d3_W4) < 0.005 ? 0 : P12.Gq_d3_W4)}. The extra layers went into the latent pathway, which keeps ${pc(g("v2", m => m.route_shares["8"].TL))} of each branch's effect even at eight branches. Its targeted effect also declines less with width than in any d = 256 seed (W8/W2 = ${f2(g("v2", m => m.targeted_full["8"] / m.targeted_full["2"]))}), much like V1. Across all the competent models, the strong route shift of Part II turns out to be specific to seeds 0 and 1. In seed 2, seed 3, V1, V2 and the K = 4 run, the latent pathway still carries 63–78% of a branch's effect at eight branches, and in V3 about half.</p>` :`<p>V2 (four layers) is still training: its first run was lost at step 23 000 in the second host crash.</p>`}`;

  // ---------------- held-out seeds
  const FC = CF || {verdicts: {}};
  const preds = ["P1", "P2", "P3", "P5", "P7", "P9", "P14", "P4", "P6", "P8", "P10"];
  const sRows = preds.filter(p => FC.verdicts[p]).map(p => [p, ["P4", "P6", "P8", "P10"].includes(p) ? "mixed in Part II (no expectation)" : "held for every Part II model",
    chip(FC.verdicts[p].all), ...["s3", "s4", "s5"].map(s => (FC.verdicts[p].heldout || {})[s] === undefined ? pend : chip(FC.verdicts[p].heldout[s]))]);
  table(P("tbl-seed3"), ["prediction", "Part II status", "Part II (all models)", LBL.s3, LBL.s4 + " (failed gate)", LBL.s5], sRows, []);
  P("prose-seed3").innerHTML = `<p>Seed 3 was started after Part II's predictions were frozen, and seed 4 after Part III's were. Neither influenced any threshold. Each is scored with the unchanged Part II scorer. Before evaluation, we expected the seven predictions that held for every Part II model to hold again; the four mixed ones are reported without a directional expectation.</p>
  ${V.heldout_s3 ? `<p><b>Seed 3</b> is the strongest model of the project: ${pct(g("s3", m => m.matched["6"].bal_acc))} depth-matched accuracy at six hops, against 59–66% for seeds 0–2. Its branch edits are large (targeted ${f2(g("s3", m => m.targeted_full["2"]))} at W = 2) and mostly stay in the latent pathway (${pc(g("s3", m => m.route_shares["8"].TL))} at W = 8). ${V.heldout_s3.expected_scored.filter(p => V.heldout_s3.verdicts[p]).length} of the ${V.heldout_s3.expected_scored.length} expected predictions hold. The exception is P7, and it fails in the opposite direction from a rescue: restoring total gain slightly <em>lowers</em> the targeted effect (W = 4: ${f2(CF.recurrent.s3.gain["4"].plain)} → ${f2(CF.recurrent.s3.gain["4"].restored)}), which is outside the frozen tolerance of 0.03. Its per-branch amplitude barely falls with width (β = ${f2(g("s3", m => m.beta_fresh))}), so P10 fails too. Competition inside the thought appears here as well, in three of four seeds now.</p>` : ""}
  ${V.heldout_s5 ? `<p><b>Seed 4</b> failed training (${pct(g("s4", m => m.competence.delayed))} delayed accuracy) and is not scored. A third held-out seed was added in its place. <b>Seed 5</b> clears the gate (${pct(g("s5", m => m.competence.delayed))}) but reaches only ${pct(g("s5", m => m.matched["6"].bal_acc))} at six hops, and it does not use its thought. At every candidate site a branch edit has a targeted effect below 0.0001, so the site rule picked one at random. Branch effects travel through query-time graph reading (${pc(g("s5", m => m.route_shares["4"].Gq))} at W = 4), and its first layer shows no sweep. The thought-edit predictions therefore fail for want of anything to edit: ${V.heldout_s5.expected_scored.filter(p => V.heldout_s5.verdicts[p]).length} of ${V.heldout_s5.expected_scored.length} expected predictions hold. The verdict stands as scored. The lesson is about the gate: overall accuracy admits a model that never uses the latent loop, as the no-latent controls already showed (91–93% overall). A depth-matched gate would have excluded it.</p>` : ""}`;

  // ---------------- upfront format
  const UP = P3.upcache || {};
  const uRows = [];
  ["s0", "s1", "s2"].forEach(s => { const R = UP[s]; if (!R) return;
    W3.forEach(W => { const u = R[W] && R[W].upcache; if (!u) return;
      const dl = CF && CF.recurrent[s] ? CF.recurrent[s].route_shares[W] : null;
      uRows.push([W === "2" ? LBL[s] : "", W, f2(u.clean_gap[0]), pc(u.shares.T), pc(u.shares.TL), pc(u.shares.Gq), dl ? pc(dl.TL) : "–", dl ? pc(dl.Gq) : "–"]); }); });
  table(P("tbl-up3"), ["seed", "W", "clean effect", "thought only", "latent pathway", "query reads graph", "delayed: latent", "delayed: query"], uRows, [1, 2, 3, 4, 5, 6, 7]);
  P("prose-up3").innerHTML = `<p>Part II found that as width grows, delayed queries resolve more nodes by reading the graph tokens themselves, bypassing the thought. In the upfront format the target node comes before the latent steps and one ANSWER token reads the result. Applying the same route decomposition there (exploratory, discovery graphs) gives a clean answer in all three seeds. The ANSWER token's own reading of the graph carries none of the target branch's effect, at every width and every target distance, and the latent pathway (thought plus START, TARGET and early-latent cache) carries all of it. Seed 1's thought alone carries everything; seeds 0 and 2 split it between the thought and the early cache.</p>
  <p>So the route shift that Part II used to explain the width effect is specific to the delayed diagnostic. When the model knows what it is looking for, it computes the answer inside the latent loop. This is also why upfront branch edits were selective in Part I.</p>`;

  // ---------------- test-time search
  const B = P3.budget || {}, L = P3.length || {};
  const bRuns = Object.keys(B).filter(r => (B[r].mode === "coconut") && TAG_OF[r]);
  const ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const bPts = r => ks.map(k => { const row = B[r].rows.find(x => x["K'"] === k && (k <= B[r].K ? x.variant === "train" : x.variant === "repeat"));
    return row ? [k, row.by_depth["6"].bal_acc, 0, 0] : null; }).filter(Boolean);
  const MORDER = ["s0", "s1", "s2", "s3", "s4", "v1", "v2", "v3"];
  bRuns.sort((a, b) => MORDER.indexOf(TAG_OF[a]) - MORDER.indexOf(TAG_OF[b]));
  const bSeries = bRuns.map(r => ({name: LBL[TAG_OF[r]], color: mcol(TAG_OF[r]), pts: bPts(r), dash: mdash(TAG_OF[r])}));
  P("lg-budget").innerHTML = bSeries.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  if (bSeries.length) lineChart(P("fig-budget"), {x: ks, y: [0.45, 0.85], yt: [0.5, 0.6, 0.7, 0.8], yfmt: t => pct(t), tfmt: pct, xname: "K′", xtitle: "latent steps at inference (trained with 6)", series: bSeries, w: 900, h: 280, aria: "Depth-6 accuracy vs latent steps"});
  P("cap-budget").textContent = "Dashed lines: architecture variants. Beyond six steps the model runs its recurrence longer than it ever did in training.";
  const lRuns = Object.keys(L).filter(r => TAG_OF[r] || /pause|k0/.test(r));
  const lk = [0, 2, 4, 6, 8, 10, 12, 16];
  const lname = r => TAG_OF[r] ? LBL[TAG_OF[r]] : r.includes("pause") ? "pause seed 0" : "no-latent seed 2";
  lRuns.sort((a, b) => (TAG_OF[a] ? MORDER.indexOf(TAG_OF[a]) : 99) - (TAG_OF[b] ? MORDER.indexOf(TAG_OF[b]) : 99));
  const lSeries = lRuns.map(r => ({name: lname(r), color: TAG_OF[r] ? mcol(TAG_OF[r]) : css("--neutral"), band: true, dash: TAG_OF[r] ? mdash(TAG_OF[r]) : "2 3",
    pts: lk.map(k => { const row = L[r].rows.find(x => x["K'"] === k); if (!row) return null; const ci = row.far_ci ? row.far_ci["7"] : [row.far["7"], row.far["7"]]; return [k, row.far["7"], ci[0], ci[1]]; }).filter(Boolean)}));
  P("lg-length").innerHTML = lSeries.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  if (lSeries.length) lineChart(P("fig-length"), {x: lk, y: [0.4, 0.8], yt: [0.5, 0.6, 0.7, 0.8], yfmt: t => pct(t), tfmt: pct, xname: "K′", xtitle: "latent steps", series: lSeries, w: 460, h: 260, aria: "Distance-7 accuracy vs latent steps"});
  P("cap-length").textContent = "Band: 95% bootstrap interval over graphs. Positives: a decoy chain re-hung from the leaf of a reachable chain (distances 7–12). Negatives: the same shape re-hung from a decoy leaf.";
  const s2run = RUN("s2"), wdRow = (k, v) => B[s2run] && B[s2run].rows.find(x => x["K'"] === k && x.variant === v);
  const r10 = wdRow(10, "repeat"), r6 = wdRow(6, "train"), dsx = [1, 2, 3, 4, 5, 6];
  if (r10 && r10.by_width_depth) {
    const ser = [];
    ["2", "4", "8"].forEach((W, i) => {
      ser.push({name: `W = ${W}, 10 steps`, color: css(col[i]), pts: dsx.map(d => [d, r10.by_width_depth[W][d], 0, 0])});
      ser.push({name: `W = ${W}, 6 steps`, color: css(col[i]), dash: "4 4", pts: dsx.map(d => [d, r6.by_width_depth[W][d], 0, 0])});
    });
    P("lg-wd").innerHTML = ["2", "4", "8"].map((W, i) => `<span><i style="background:${css(col[i])}"></i>W = ${W}</span>`).join("");
    lineChart(P("fig-wd"), {x: dsx, y: [0.45, 1], yt: [0.5, 0.6, 0.7, 0.8, 0.9, 1], yfmt: t => pct(t), tfmt: pct, xname: "hops", xtitle: "hops from start", series: ser, w: 460, h: 260, aria: "Seed 2 accuracy by depth and width"});
  }
  P("cap-wd").textContent = "Seed 2 only. Ten steps move depth 6 to the width's own ceiling but never raise the ceiling.";
  const s2L = L[s2run] ? L[s2run].rows.find(x => x["K'"] === 12) : null;
  P("prose-search").innerHTML = `<p>Truncating the latent budget below six steps was not a clean ablation (Part I). Extending it is. Each model ran its recurrence for up to ten steps (the step-6 embedding reused for steps 7–10) and was scored with depth-matched decoys on discovery graphs. The seeds turn out to differ in kind. Seed 2 resolves depth one step at a time, a staircase: three hops by step 2, four by step 4, five by step 6. It keeps going past its training budget, with six-hop accuracy rising from ${pct(g("s2", m => m.search.d6_K6))} at six steps to ${pct(g("s2", m => m.search.d6_K8))} at eight. Seeds 0 and 1 have resolved everything they will by step 4–6 and gain nothing from more steps. Pause-token and no-latent models gain nothing either.</p>
  <p>Chains longer than any in training make the difference sharper. Re-hanging a decoy chain from the leaf of a reachable one puts nodes at distances 7–12, which never occur in training. Seed 2 reaches distance 7 with ${s2L ? pct(s2L.far["7"]) : "–"} balanced accuracy after twelve steps (chance before step 8) and distance 8 with ${s2L ? pct(s2L.far["8"]) : "–"}, roughly one hop per two steps. Every other model stays at chance beyond six hops however long it runs. All three seeds sweep forward step by step (previous section), but only seed 2's sweep can simply be run longer. Seeds 0 and 1 stop once their faster sweep has covered the trained range. On the training distribution the two behaviours reach similar accuracy.</p>
  <p>Width limits this search through fidelity, not time. In seed 2 depth 6 resolves at about the same step at every width, but the level it resolves to falls with width. Extra steps bring each width to its own ceiling (depth 6: ${r10 ? pct(r10.by_width_depth["2"]["6"]) : "–"} at W = 2, ${r10 ? pct(r10.by_width_depth["8"]["6"]) : "–"} at W = 8) and never above it. That fits Part II's normalised code: each branch's share of the state shrinks with width, and each hop copies it with more loss. After seeing this we pre-registered P15. On every newly trained recurrent model, in-range gains from extra steps should predict out-of-range extrapolation.${P15 && !P15.pass ? ` It failed. V1 gains ${f3(P15.per_model.v1.gain_6_to_8)} at six hops from two extra steps, just over the bar, yet stays at chance at distance 7. In-range gains are not a reliable sign of an open-ended loop.` : ""}</p>`;

  // ---------------- two search algorithms (attention traces + causal block)
  const TU = P3.trace_up, BL = P3.block, TD = P3.trace_delayed, LU = P3.length_up;
  const v3r = RUN("v3");
  if (TU && TU[v3r]) {
    const tr = TU[v3r].reachable.trace, ks8 = [1, 2, 3, 4, 5, 6, 7, 8], xs6 = [1, 2, 3, 4, 5, 6];
    heatmap(P("fig-walk-attn"), {w: 460, rows: ks8.map(k => `step ${k}`), cols: xs6.map(String), ml: 64, ch: 24, max: 0.25,
      val: (r, c) => tr[`L1_k${r.split(" ")[1]}`].path[Number(c) - 1], colLabel: c => `depth ${c}`,
      tipf: (r, c, v) => `${r}, path depth ${c}<br>attention <b>${v.toFixed(3)}</b>`, aria: "V3 attention along the target path"});
  }
  if (BL) {
    const ks8 = [1, 2, 3, 4, 5, 6, 7, 8], xs6 = [1, 2, 3, 4, 5, 6];
    heatmap(P("fig-walk-block"), {w: 460, rows: ks8.map(k => `step ${k}`), cols: xs6.map(String), ml: 64, ch: 24, max: 0.5,
      val: (r, c) => -BL.path[r.split(" ")[1]][Number(c) - 1], colLabel: c => `depth ${c}`,
      tipf: (r, c, v) => `${r}, block path edge at depth ${c}<br>P(reachable) drops by <b>${v.toFixed(3)}</b>`, aria: "Effect of blocking one path edge at one step"});
    const ctrl = Math.max(...Object.values(BL.control).flat().map(Math.abs));
    P("cap-walk").textContent = `V3 only; ${BL.n} graphs, baseline P(reachable) ${f2(BL.p_base)}. Largest effect of blocking a same-depth edge in another reachable branch: ${ctrl.toFixed(3)}. Seeds 0 and 1 show no step structure in the target-first format; seed 2 sweeps forward instead (attention on all same-depth edges of reachable branches).`;
  }
  const DB = P3.delayed_block;
  [["s1", "fig-sweep-s1"], ["s2", "fig-sweep-s2"]].forEach(([t, id]) => {
    const R_ = DB && DB[RUN(t)]; if (!R_) return;
    heatmap(P(id), {w: 460, rows: [1, 2, 3, 4, 5, 6, 7, 8].map(k => `step ${k}`), cols: ["1", "2", "3", "4"], ml: 64, ch: 24, max: 0.35,
      val: (r, c) => -R_.downstream[r.split(" ")[1]][Number(c)], colLabel: c => `depth ${c}`,
      tipf: (r, c, v) => `${r}, block depth-${c} edges<br>deeper nodes drop by <b>${v.toFixed(3)}</b>`, aria: `Forward-sweep blocking, ${LBL[t]}`});
  });
  const DBn = P3.delayed_block_new || {};
  const s3drop = DBn[RUN("s3")] ? Math.max(...[1, 2, 3, 4].map(k => -DBn[RUN("s3")].downstream[k][k])) : NaN;
  if (DB) P("cap-sweep").textContent = `Seed 0 matches seed 1 shifted by one step (depth x read at step x + 1, nothing after step 5). Held-out seed 3 also advances one depth per step, and blocking the right step lowers the branch's downstream P(reachable) by up to ${f2(s3drop)}; V1 reads depths 1–2 in two passes (steps 1–3, then 4–5). Blocking one branch's edges raises the other reachable branches by at most ${Math.max(...[...Object.values(DB), ...Object.values(DBn)].flatMap(R_ => Object.values(R_.control).flat())).toFixed(3)}: the competition of Part II.`;
  const lu = r => LU && LU[r] ? LU[r] : null;
  const luv = (r, k, d) => lu(r) && lu(r)[k] ? lu(r)[k][d] : NaN;
  const E_ = t => M[t] ? M[t].frontier_E : null;
  P("prose-walk").innerHTML = `<p>Attention traces show what each latent step reads from the graph (exploratory, discovery graphs). In the delayed format every recurrent model sweeps <em>forward</em>. At each step its first layer attends to the edges of one within-branch depth across all reachable branches, leaving decoy edges at baseline, and the depth advances with the step. The models differ in speed. Seed 1 advances one depth per step and reaches the deepest trained depth at step 5, seed 0 one step later, and seed 2 advances one depth every two steps (depth 3 at step 6). The sweep is causal and clocked. Blocking one branch's edges of a given depth at the step that reads them lowers that branch's deeper nodes (by 0.11–0.15 in seed 1, up to 0.34 in seed 2); blocking the same edges at any other step changes nothing. ${(() => { const TW = P3.trace_W || {}, r1 = RUN("s1");
      const v = W => TW[W] && TW[W][r1] ? TW[W][r1].delayed["L1_k1"].reachable[1] : NaN;
      return Number.isNaN(v("2")) ? "" : `Width leaves the clock unchanged: the step at which each depth is read is the same at two, four and eight branches. It changes the amplitude. Seed 1's per-edge attention on the depth-1 frontier at step 1 is ${v("2").toFixed(3)}, ${v("4").toFixed(3)} and ${v("8").toFixed(3)} at W = 2, 4 and 8, so the total on the frontier stays near ${(2 * v("2")).toFixed(2)}: a fixed attention budget shared among the branches. That is the mechanism behind the per-branch amplitude law of Part II, and behind the finding below that width limits the search's fidelity rather than its speed.`; })()}</p>
  <p>Pause-token models show no such sweep. Their latent positions cannot carry graph content into the first layer, and in the second layer two of the three pause seeds attend to the reachable hub edges equally at every step. The stepwise frontier needs the recurrent thought.</p>
  <p>With the target given first, V3 does something else. Its attention starts at the edge into the target's parent and climbs the target's ancestry one edge per latent step until it reaches the hub edge at step 6. Whether that hub edge comes from the start or the decoy hub settles the answer. Blocking the edge the walk should read at a given step lowers the answer by up to ${BL ? f2(-Math.min(...Object.values(BL.path).flat())) : "–"}; blocking it one step early or late, or blocking a same-depth edge elsewhere, does nothing. This backward pointer-chasing follows a single path, so it cannot depend on width. V3's target-first accuracy is 100% at every width from 1 to 8, it never needs to read the graph at answer time, and with more steps it follows chains to distance 7 (${pct(luv(v3r, "16", "7"))} after sixteen steps) even though it does not extrapolate in the delayed format.</p>
  <p>The forward sweeps suggest why seeds differ at test time. Seeds 0 and 1 finish the trained depth inside the six-step budget, and seed 0 at least then enters a state that stops the search (next sections). Seed 2's slower sweep is still running at step 6. This led to a second prospective prediction (P16): no new model whose sweep reaches depth 4 or more within six steps will extrapolate. Frontier depth reached by step 6 so far: seed 0 ${E_("s0")}, seed 1 ${E_("s1")}, seed 2 ${E_("s2")}, V3 ${E_("v3")}.</p>`;

  // ---------------- rewind
  const RW = P3.rewind || {};
  const farv = d => d ? mean(["7", "8", "9"].map(x => d[x])) : NaN;
  if (RW.s0) {
    const base = farv(RW.s0.baseline["12"]);
    const js = [3, 4, 5, 6, 7, 8, 9, 10, 11], ss = [0, 1, 2, 3, 4, 5, 6, 7];
    const cell = (j, s) => { const r = j - 1 - s; const v = RW.s0.grid[`K12_j${j}_r${r}_rewind`]; return v ? farv(v) - base : NaN; };
    heatmap(P("fig-rewind"), {rows: js.map(j => `inject at step ${j}`), cols: ss.map(String), ml: 118, ch: 26, max: 0.13,
      val: (r, c) => cell(Number(r.split(" ").pop()), Number(c)), colLabel: c => `s = ${c}`,
      tipf: (r, c, v) => `${r}, thought from step ${c}<br>far accuracy change <b>${Number.isNaN(v) ? "–" : f3(v)}</b>`, aria: "Rewind heatmap for seed 0"});
    const best = t => { const R_ = RW[t]; if (!R_) return null; const b = farv(R_.baseline["12"]);
      return Math.max(...Object.entries(R_.grid).filter(([k]) => k.startsWith("K12_") && k.endsWith("_rewind")).map(([, v]) => farv(v))) - b; };
    const ctrlMax = suf => Math.max(...Object.entries(RW.s0.grid).filter(([k]) => k.endsWith(suf)).map(([k, v]) => farv(v) - farv(RW.s0.baseline[k.startsWith("K16") ? "16" : "12"])));
    P("cap-rewind").textContent = `Largest gain from any rewind in the other models: seed 1 ${f3(best("s1"))}, seed 2 ${f3(best("s2"))}, V3 ${f3(best("v3"))}. Seed 0 controls (12 and 16 steps): the thought from a different graph gives at most ${f3(ctrlMax("_foreign"))} (only for the encoder output, s = 0), a zero vector at most ${f3(ctrlMax("_zero"))}.`;
  }
  P("prose-rewind").innerHTML = `<p>Why do seeds 0 and 1 stop at six hops? Either their recurrence cannot propagate further, or something stops it. A rewind separates the two. During a run, the thought fed into step j is replaced by the model's own earlier thought while the key/value cache, which holds everything computed so far, is kept (exploratory, long-chain graphs).</p>
  <p>In seed 2, rewinds only cost reach: the thought carries its search state, and an old thought does not fit the newer cache. In seed 0 a single rewind does the opposite. Feeding a thought from steps 1–3 into step 6 or later lifts accuracy at distances 7–9 from chance to about 62%, with no cost at shorter distances, and in a twenty-step run it reaches distance 12. Thoughts from step 4 onward do nothing. So does the same step's thought taken from a different graph. The gain is largest at two branches (balanced accuracy 0.49 → 0.66) and small at six.</p>
  <p>Seed 0's propagation machinery therefore works past its trained depth. A first guess was that a state its thought enters around step 4 stops it. That state is not a simple, graph-independent direction, and the next section shows that the stop is keyed to the step embedding instead: shifting a late thought by the average difference between early and late thoughts does not unlock the search (far accuracy at most ${(() => { const PS = P3.phase; if (!PS) return "0.52"; return f2(Math.max(...PS.rows.filter(r => r.kind === "phase").map(r => mean(["7", "8", "9"].map(x => r.chain[x]))))); })()}). Only an early thought from the same graph, combined with the propagated cache, restarts it. No rewind helps seed 1, seed 2 or V3.${RW.s3 ? ` Held-out seed 3, whose sweep also finishes early, shows the same signature more weakly. Its best rewinds all feed back its step-3 thought and lift far accuracy from ${f2(farv(RW.s3.baseline["12"]))} to ${f2(Math.max(...Object.entries(RW.s3.grid).filter(([k]) => k.startsWith("K12_") && k.endsWith("_rewind")).map(([, v]) => farv(v))))}.` : ""}</p>`;

  // ---------------- clock: step embedding carries the stop
  const CFz = P3.clock_freeze || {}, CSw = P3.clock_swap || {};
  const clockModels = ["s3", "v2", "v1"].filter(t => CFz[RUN(t)]);
  const dx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const cSer = [];
  clockModels.forEach(t => {
    const R_ = CFz[RUN(t)];
    cSer.push({name: `${LBL[t]}, step clock held at 2`, color: mcol(t), pts: dx.map(d => [d, R_["freeze_2_K20"].acc[d], 0, 0])});
    cSer.push({name: `${LBL[t]}, normal`, color: mcol(t), dash: "4 4", pts: dx.map(d => [d, R_["normal_K20"].acc[d], 0, 0])});
  });
  P("lg-clock").innerHTML = clockModels.map(t => `<span><i style="background:${mcol(t)}"></i>${LBL[t]}</span>`).join("");
  if (cSer.length) lineChart(P("fig-clock"), {x: dx, y: [0.45, 1], yt: [0.5, 0.6, 0.7, 0.8, 0.9, 1], yfmt: t => pct(t), tfmt: pct, xname: "distance", xtitle: "hops from start", series: cSer, w: 900, h: 300, aria: "Frozen step clock vs normal, accuracy by distance"});
  const fz = (t, d) => CFz[RUN(t)] ? CFz[RUN(t)]["freeze_2_K20"].acc[d] : NaN;
  const sw = (t, cnd, d) => CSw[RUN(t)] ? CSw[RUN(t)][cnd].matched[d] : NaN;
  const v1c = CFz[RUN("v1")] ? CFz[RUN("v1")]["freeze_2_K12"] : null;
  P("cap-clock").textContent = `Seeds 3 and V2 show the effect most clearly; V1 is the prospective test (P17), which failed. With the clock held at 2, seeds 0 and 1 reach ${pct(fz("s0", "9"))} and ${pct(fz("s1", "9"))} at distance 9 (normal: chance). Seed 2, which never stops, is unaffected and not shown.`;
  P("prose-clock").innerHTML = `<p>The rewinds changed the thought. A cleaner test changes only the step embedding each latent step receives, the model's sole explicit signal of how far into the loop it is, and leaves the thought and the cache alone (exploratory). Two things separate. First, the depth a step reads does not follow the step embedding. Hold the embedding at an early step, or delay it, and every model still reads depth 1, then 2, then 3, at the same steps as before. The frontier is carried by the thought. Second, the <em>stop</em> does follow the step embedding. Advance it by two steps and seed 3's sweep ends early (six-hop accuracy ${f2(sw("s3", "normal", "6"))} → ${f2(sw("s3", "shift_+2", "6"))}). Hold it at step 2 and the models that normally stop at six hops keep going: seed 3 reaches ${pct(fz("s3", "7"))} at distance 7 and ${pct(fz("s3", "9"))} at 9, and V2 ${pct(fz("v2", "7"))} at 7 and ${pct(fz("v2", "10"))} at 10, where their normal runs are at chance.</p>
  <p>So the propagation step these models learned works well past the depth they were trained on, and what stops them is a halt keyed to the step index, learned because every training episode ended by step 6. The attention trace below shows that the halt acts on the sweep itself. With normal step embeddings V2's first layer stops reading edges at step 6. With the embedding held at step 2 it walks the long chain one distance per step, through distances 7–12, and stops only at the chain's end. Seed 2's slower sweep never finished early, never learned the halt, and extrapolates without help. This supersedes the reading of the rewind results as a halting state in the thought: a rewind restores early thought content but keeps the late step embeddings, and it unlocks much less. ${V.P18 && V.P18.pass !== undefined ? `A training-level test agrees. A model trained with no step embedding at all (P18, pre-registered) has no step index to learn a halt from, and it extrapolates: ${pct(V.P18.search.d7_K12)} at distance 7 after twelve steps, still above 60% at distance 12 after sixteen, gaining about one hop per extra step. ` : ""}The effect of holding the clock is not universal. Seeds 0 and 1 gain less (about 60–65% out to distance 12). The two models tested only after it was pre-registered as P17, V1 and the K = 8 run, gain too little to count. Their traces show why: their sweeps do not halt sharply at a step but fade hop by hop, reaching distance 7 too weakly to matter, and a held clock cannot fix that. So there are two kinds of limit, a step-keyed halt (seed 3, V2) and per-hop attenuation (K = 8, V1), and P17 had assumed the first everywhere${v1c ? ` (distance 7: ${pct(v1c.acc["7"])}, bar 60%)` : ""}.</p>`;
  const CT = P3.clock_trace;
  if (CT && CT[RUN("v2")]) {
    const steps14 = Array.from({length: 14}, (_, i) => i + 1), dist12 = Array.from({length: 12}, (_, i) => String(i + 1));
    [["normal", "fig-ctrace-n"], ["freeze_2", "fig-ctrace-f"]].forEach(([cnd, id]) => {
      const tr = CT[RUN("v2")][cnd];
      heatmap(P(id), {w: 460, rows: steps14.map(k => `step ${k}`), cols: dist12, ml: 60, ch: 16, max: 0.2, fmt: v => v >= 0.05 ? v.toFixed(2).replace(/^0/, "") : "",
        val: (r, c) => tr[Number(r.split(" ")[1]) - 1][Number(c) - 1], colLabel: c => c,
        tipf: (r, c, v) => `${r}, edge into distance ${c}<br>attention <b>${v.toFixed(3)}</b>`, aria: `V2 attention trace, ${cnd}`});
    });
  }
  const P17v1 = v1c ? {d7: v1c.acc["7"], ci: v1c.ci["7"]} : null;
  const k8c = CFz[RUN("k8")] ? CFz[RUN("k8")]["freeze_2_K12"] : null;
  const p17row = ["P17", "Holding the step embedding at step 2 lets a non-extrapolating model reach distance 7 (≥ 60%, lower bound > 55%)", `${LBL.v1}; ${LBL.k8 || "budget K = 8"}`,
    [P17v1 ? `V1: ${pct(P17v1.d7)} [${pct(P17v1.ci[0])}, ${pct(P17v1.ci[1])}]` : "V1: –", k8c ? `K = 8: ${pct(k8c.acc["7"])} [${pct(k8c.ci["7"][0])}, ${pct(k8c.ci["7"][1])}]` : "K = 8: training"].join("; "),
    P17v1 ? chip(P17v1.d7 >= 0.6 && P17v1.ci[0] > 0.55 && (!k8c || (k8c.acc["7"] >= 0.6 && k8c.ci["7"][0] > 0.55))) : pend];
  const st_ = P("tbl-p3score").querySelector("tbody");
  if (st_) st_.insertAdjacentHTML("beforeend", "<tr>" + p17row.map(c => `<td>${c}</td>`).join("") + "</tr>");
  const jRow = (id, pred, model, v, txt) => st_ && st_.insertAdjacentHTML("beforeend", "<tr>" + [id, pred, model, v ? txt(v) : "training",
    !v ? pend : v.gate === false ? `<span class="chip">not scored</span>` : v.pass === undefined ? pend : chip(v.pass)].map(c => `<td>${c}</td>`).join("") + "</tr>");
  jRow("J1", "K = 4, seed 2 (stricter gate): extrapolates or reads the graph at query time", "budget K = 4, seed 2", V.J_k4c,
    v => v.gate === false ? "failed competence gate" : `distance 7 at 12 steps ${f2(v.search.d7_K12)}; query share W4 ${pc(v.Gq_W4)}`);
  jRow("J2", "K = 8: sweep reaches depth 4 within six steps and no extrapolation at 16 steps", "budget K = 8", V.J_k8,
    v => v.gate === false ? "failed competence gate" : `E = ${v.E}; distance 7 at 16 steps ${f2(v.d7_K16)}`);
  jRow("P18", "No step embedding: the model extrapolates (distance 7 ≥ 60% at 12 steps)", "no step embedding", V.P18,
    v => v.gate === false ? "failed competence gate" : `distance 7 at 12 steps ${f2(v.search.d7_K12)} [${v.search.d7_K12_ci ? v.search.d7_K12_ci.map(f2).join(", ") : "–"}]; E = ${v.E}`);

  // ---------------- development
  const D = P3.development;
  if (D && Object.keys(D).length > 1) {
    const st = Object.keys(D).map(Number).sort((a, b) => a - b);
    const host = P("fig-dev-host");
    host.innerHTML = `<div class="grid2"><figure><p class="fig-title">Routes during training, W = 4</p><p class="fig-sub">Share of the clean counterfactual effect of one branch removal carried by the latent pathway and by query-time graph reading (seed 4, discovery graphs, k = 1).</p><div class="legend"><span><i style="background:${css("--s1")}"></i>latent pathway</span><span><i style="background:${css("--s2")}"></i>query reads graph</span><span><i style="background:${css("--s3")}"></i>thought only</span></div><div id="fig-dev-route"></div></figure>
      <figure><p class="fig-title">Competence during training</p><p class="fig-sub">Depth-matched balanced accuracy by hops from start, full budget.</p><div class="legend">${["2", "4", "6"].map((d, i) => `<span><i style="background:${css(col[i])}"></i>${d} hops</span>`).join("")}</div><div id="fig-dev-comp"></div></figure></div>`;
    const sh = (s, c) => { const x = D[s].cache["4"]; return x && x.shares ? x.shares[c] : null; };
    const xs = st;
    lineChart(P("fig-dev-route"), {x: xs, y: [-0.1, 1.1], yt: [0, 0.25, 0.5, 0.75, 1], yfmt: t => pct(t), tfmt: pc, xname: "step", xlab: v => v % 5000 === 0 ? (v / 1000) + "k" : "", xtitle: "training step",
      series: [["TL", "--s1"], ["Gq", "--s2"], ["T", "--s3"]].map(([c, cl]) => ({name: c, color: css(cl), pts: xs.map(s => [s, sh(String(s), c), 0, 0])})), w: 460, h: 250, aria: "Route shares over training"});
    lineChart(P("fig-dev-comp"), {x: xs, y: [0.45, 1], yt: [0.5, 0.75, 1], yfmt: t => pct(t), tfmt: pct, xname: "step", xlab: v => v % 5000 === 0 ? (v / 1000) + "k" : "", xtitle: "training step",
      series: ["2", "4", "6"].map((d, i) => ({name: `${d} hops`, color: css(col[i]), pts: xs.map(s => [s, D[String(s)].matched[d], 0, 0])})), w: 460, h: 250, aria: "Competence over training"});
    const withSched = st.filter(s => D[String(s)].frontier_schedule);
    if (withSched.length) {
      host.insertAdjacentHTML("beforeend", `<figure><p class="fig-title">Seed 4: the sweep never advances past depth 1</p><p class="fig-sub">Depth whose edges layer 1 reads at each latent step (delayed format, W = 4); blank = no depth stands out. Rows: checkpoints.</p><div id="fig-dev-sched"></div></figure>`);
      heatmap(P("fig-dev-sched"), {rows: withSched.map(s => s >= 1000 ? `${s / 1000}k` : String(s)), cols: ["1", "2", "3", "4", "5", "6"], ml: 60, ch: 18, max: 5.5, fmt: v => `d${v}`,
        val: (r, c) => { const s = withSched.find(x => (x >= 1000 ? `${x / 1000}k` : String(x)) === r); const v = D[String(s)].frontier_schedule[Number(c) - 1]; return v == null ? NaN : v; },
        colLabel: c => `step ${c}`, tipf: (r, c, v) => `checkpoint ${r}, latent step ${c}<br>${Number.isNaN(v) ? "no clear frontier" : `reads depth ${v}`}`, aria: "Frontier schedule over training"});
    }
  }
  P("prose-dev").innerHTML = D && Object.keys(D).length > 1
    ? `<p>Seed 4 saved its weights every 1 000 steps and at each curriculum advance. Each checkpoint was run through the route decomposition and the attention trace (exploratory, discovery graphs, k = 1). Seed 4 turned out to be a training failure (${pct(g("s4", m => m.competence.delayed))} delayed accuracy, below the 85% gate), which makes its trajectory informative for a different reason. It shows what happens when the clocked sweep does not form.</p>
    <p>At the first checkpoint, on star-shaped branches, a branch's effect travels almost entirely through query-time graph reading, and already more so at higher width (latent share ${pc(D["1000"].cache["2"].shares.TL)} at W = 2, ${pc(D["1000"].cache["8"].shares.TL)} at W = 8). As two-hop nodes are learned, the latent pathway grows to about half of the effect. But the first layer's sweep never advances: from step 4 000 to the end it reads depth-1 edges at every step. Deep nodes stay at chance. When three-hop nodes finally improve after step 17 500, they do so through query-time reading, whose share at W = 4 rises from ${pc(D["8000"].cache["4"].shares.Gq)} to ${pc(D["30000"].cache["4"].shares.Gq)}. That route never reaches past three hops. Width then separates the routes further (latent share at W = 8 ${pc(D["8000"].cache["8"].shares.TL)} → ${pc(D["30000"].cache["8"].shares.TL)}).</p>`
    : `<p>Seed 4 is training with checkpoints every 1 000 steps. This section fills in when it finishes.</p>`;

  // ---------------- incident + integrity
  const I = P3.integrity;
  P("prose-integrity").innerHTML = `<p>During the night of 24 September the host crashed with a memory-management bugcheck, and Windows then retired bad RAM regions. Shortly before the crash an analysis job failed on a resampling index of exactly 2<sup>32</sup> + 11, a single flipped bit in host memory. One GPU (the RTX 3070) had also logged about 940 context-switch exceptions an hour earlier, which explains a training stall first attributed to memory pressure. The card later passed a bit-exact compute test and a full VRAM pattern test.</p>
  <p>To check whether stored results had been silently corrupted, seed 1's fresh-confirmation cache factorial and dose experiments were recomputed from scratch on a different GPU and compared number by number with the stored summary. ${I ? `All ${I.deterministic_compared.toLocaleString()} deterministic quantities agree within 10<sup>−4</sup>. The only differences (${I["random_control_over_1e-4"]} numbers) are in the random-subspace null control, whose basis is redrawn from a device RNG.` : "(pending)"}</p>
  <p>A second crash followed at about 07:20 (bugcheck 0x133, a DPC watchdog timeout), and at the end of the session the RTX 3070 dropped off the PCI bus ("GPU is lost"), taking the last evaluation job with it; that model had already failed the competence gate. Runs launched after the second crash keep a full training state and resume after a crash.</p>
  <p>Training on this rig is not deterministic even with a fixed seed and device, so a seed pins the data stream and initialisation, not the trajectory. About half of all gated training attempts, in both precisions, stalled at the first curriculum stage, with delayed loss on a plateau near 0.565 and upfront loss at chance. Such runs were stopped at step 4 000 and relaunched under a rule written before any of them was evaluated. Every attempt is kept on disk and listed in the research log.</p>`;

  // ---------------- abstract + status additions
  const ab = P("prose-abstract");
  if (ab) ab.insertAdjacentHTML("beforeend", `<p><b>Part III</b> looks at what the latent loop computes and how that depends on architecture and seed. In the delayed format every recurrent model runs a clocked breadth-first sweep: at step k its first layer reads the edges of one depth across all reachable branches, and blocking those edges at that step, and only then, cuts the branch's deeper nodes. The per-edge attention on the frontier halves each time the width doubles while the clock stays fixed, which is Part II's shared budget seen directly. Seeds differ in clock rate. Those that finish the trained depth inside the budget stop there, and the stop is keyed to the step embedding: holding it at an early step lets two of them follow chains well past the trained depth (up to 88% at seven hops). The seed whose sweep is still running at the last step follows chains beyond any trained length when given more steps, and so do models trained with only four steps, too few for the sweep to finish (both competent seeds), and a model trained with no step embedding at all; a model trained with eight steps does not. A wider and a deeper variant both kept more of each branch in the latent pathway, contrary to both pre-registered predictions for them. With the target given first, the raw-feedback variant instead walks backward from the target along its parent chain, a single-path search whose accuracy does not depend on width. Of the pre-registered Part III tests, ${["P11", "P12", "P13", "P15", "P16"].filter(p => V[p] && V[p].pass === true && !(V[p].pending || []).length).length} passed, ${["P11", "P12", "P13", "P15", "P16"].filter(p => V[p] && V[p].pass === false).length} failed, and the rest are pending. See the <a href="#part3h">Part III scorecard</a>.</p>`);
  const stt = P("prose-status");
  if (stt) stt.insertAdjacentHTML("beforeend", ` Part III (<code>docs/PROTOCOL3.md</code>, addenda B–G) adds three architecture variants, two held-out seeds, test-time search tests and causal attention-blocking tests, several of them exploratory and marked as such.`);

  // ---------------- findings, claims, limits additions
  const fl = document.getElementById("findings-list");
  if (fl) {
    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: one seed learned a search that can run longer.</b> Given more latent steps than it was trained with, seed 2 keeps resolving deeper nodes and follows chains to distances never seen in training (distance 7: ${s2L ? pct(s2L.far["7"]) : "–"} after twelve steps). Seeds 0 and 1, the pause and no-latent controls, and the raw-feedback variant stay at chance beyond six hops. Width caps the accuracy the search reaches, not how fast it proceeds.</li>`);
    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: the thought carries the frontier; the step embedding carries the stop.</b> Holding the step embedding at step 2 leaves the depth each step reads unchanged but removes the halt, and models that never extrapolate follow chains past six hops (seed 3 and V2 at 87–88% at distance 7). The propagation step generalises; the halt was learned from the step index. The prospective test of this on V1 (P17) failed, so it is not universal.</li>`);
    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: the route shift is a property of the delayed diagnostic.</b> With the target given first, the answer token never reads the graph (0% query-time share in all three seeds), and the latent pathway carries the whole effect at every width.</li>`);
    if (P13) fl.insertAdjacentHTML("beforeend", `<li><b>Part III: removing the thought's renormalisation changes the amplitude law as predicted.</b> With raw-residual feedback β = ${f2(P13.beta)} (reference 0.62–0.87), and competition between branches persists (P13, pre-registered).</li>`);
  }
  const ct = document.querySelector("#tbl-claims table tbody");
  if (ct) {
    const add = cells => ct.insertAdjacentHTML("beforeend", "<tr>" + cells.map(c => `<td>${c}</td>`).join("") + "</tr>");
    add(["Some recurrent seeds implement an extendable iterative search", "Budget sweep staircase; length test to distances 7–9 (seed 2 only); P15 prospective", `<span class="chip">exploratory</span>`, "1 of 3 original seeds; mechanism of the loop not dissected"]);
    add(["The stop is keyed to the step embedding, not the search state", "Clock freeze/shift on s0, s1, s3, V2 (exploratory); P17 prospective on V1 failed", `<span class="chip">exploratory</span>`, "Strong in 2 of 5 models tested, weak in V1"]);
    add(["Width limits search fidelity, not speed", "Width-resolved budget sweep, seed 2", `<span class="chip">exploratory</span>`, "One seed; ceilings measured on discovery graphs"]);
    add(["The delayed-format route shift does not occur when the target is known", "Upfront route decomposition, 3 seeds, Gq ≈ 0", `<span class="chip yes">3/3 seeds</span>`, "Discovery graphs; not pre-registered"]);
    if (P13) add(["Final-LayerNorm renormalisation sets the per-branch amplitude law", `V3: β = ${f2(P13.beta)} vs 0.62–0.87 (P13)`, chip(P13.pass), "One variant seed"]);
  }
  P("prose-limits").insertAdjacentHTML("beforeend", `<p><b>Part III specifics.</b> Each architecture variant is a single training run, so a variant result can reflect that run's trajectory. The held-out seeds address this for the reference recipe only. The length test uses all-chain graphs, whose structure (every branch a chain) essentially never occurs in training; failure to extrapolate could partly reflect that distribution shift rather than the absence of an iterative loop. Stage-1 training failures were handled by a restart rule. This conditions the reported models on competence, as the Part I and II gates already did. Roughly half of all gated training attempts stalled at stage 1, in both precisions. The competence gate is overall balanced accuracy, and one held-out seed that passed it (seed 5) turned out not to use its latent loop at all; a depth-matched gate would have been the better choice.</p>`);
})();
