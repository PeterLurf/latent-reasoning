// ================================================================ Part II (follow-up)
const FU = S.followup || null;
const CF = FU ? FU.confirm : null;
const RECS = ["s0", "s1", "s2"];
const recOK = CF ? RECS.filter(s => CF.recurrent[s]) : [];
const W3 = ["2", "4", "8"];
const mm = a => { const v = a.filter(x => x != null && !Number.isNaN(x)); return v.length ? [mean(v), Math.min(...v), Math.max(...v)] : [NaN, NaN, NaN]; };
const chip = ok => ok === true ? `<span class="chip yes">pass</span>` : ok === false ? `<span class="chip no">fail</span>` : `<span class="chip">n/a</span>`;
const pc = v => (v == null || Number.isNaN(v)) ? "–" : (100 * v).toFixed(0) + "%";

// ---------------- controls: groups of seeds
const grp = names => names.filter(n => C[n]);
const PAUSE_N = grp(["pause", "pause1", "pause2"]), K0_N = grp(["k0", "k0_1", "k0_2"]);
const accG = ns => mm(ns.map(n => balD(C[n])));
const depthG = (ns, d) => mm(ns.map(n => byDepth(C[n], d)));

if (CF) {
  // ---------------- revised abstract, status, findings
  const sh = (s, W, c) => CF.recurrent[s].route_shares[W][c];
  const latW2 = mm(recOK.map(s => sh(s, "2", "TL"))), latW8 = mm(recOK.map(s => sh(s, "8", "TL")));
  const qW2 = mm(recOK.map(s => sh(s, "2", "Gq"))), qW8 = mm(recOK.map(s => sh(s, "8", "Gq")));
  const pauseQ = mm(Object.values(CF.pause).flatMap(p => W3.map(W => p.route_shares[W].Gq)));
  const nPause = Object.keys(CF.pause).length;
  const pauseAcc = accG(PAUSE_N), k0Acc = accG(K0_N);
  const pd5 = depthG(PAUSE_N, "5"), pd3 = depthG(PAUSE_N, "3");
  const RD = FU.route_depth || {};
  const deepT = W => mm(recOK.filter(s => RD[s]).flatMap(s => ["5","6"].map(d => RD[s][W][d] && RD[s][W][d].clean_gap > 0.05 ? RD[s][W][d].T : NaN)));
  const gapD = (W, d) => mm(recOK.filter(s => RD[s]).map(s => RD[s][W][d].clean_gap))[0];

  P("prose-abstract").innerHTML = `<p>A continuous-thought transformer can hold several graph-search branches in one latent state, and edits to one branch move that branch more than the others. The edits also pass a semantic test: an inserted branch continues with the recipient's graph rather than carrying the donor's answers. But each edit weakens as more branches are active. The follow-up study (Part II) explains why. Scaling the edit does not help, and nor does restoring attention gain. Two things change instead. First, branches share a normalised, relative code: boosting one lowers the others, and the per-branch signal falls roughly as W<sup>−0.6…0.9</sup>. With it, the model's own sensitivity to a branch's deep nodes collapses as width grows. Second, nodes two or three hops away move from the latent pathway to queries reading the graph tokens directly, where a thought edit cannot reach. Averaged over the branch, the latent pathway's share falls from ${pc(latW2[0])} to ${pc(latW8[0])}. For the deepest nodes, though, a thought edit reproduces a similar share of the model's own counterfactual at every width. The directions a linear probe reads from are almost causally inert; the activation patterns for the same node ids carry the effect.</p>`;
  P("prose-status").innerHTML = `<strong>Status.</strong> Two completed studies on one architecture and one synthetic task. Part I was frozen from discovery data and confirmed on held-out families. Part II followed a second pre-registration (<code>docs/PROTOCOL2.md</code>, <code>docs/protocol2_frozen.json</code>) and was confirmed on <em>fresh</em> graphs that exclude every family any earlier analysis touched. It uses 300 graphs per width, a bootstrap over graphs within each seed, and three recurrent seeds shown separately. Controls: ${nPause} pause-token and ${K0_N.length} no-latent seeds. Training used a documented lookup scaffold (see <a href="#limits">Limits</a>).`;

  const F2 = [];
  const CMf = S.competence_matched || {};
  const mat = (ns, d) => mm(ns.filter(n => CMf[n]).map(n => CMf[n][d].bal_acc));
  const ctrlN = ["pause", "pause1", "pause2", "k0", "k0_1", "k0_2"].filter(n => CMf[n]);
  F2.push(`<b>Recurrence buys reliability on the deepest queries, not a broad capability.</b> With depth-matched decoys (chance 50%), recurrent seeds score ${pct(mat(seeds, "6")[1])}–${pct(mat(seeds, "6")[2])} at six hops. All ${ctrlN.length} pause and no-latent control seeds score ${pct(mat(ctrlN, "6")[1])}–${pct(mat(ctrlN, "6")[2])}. Below that, controls vary enormously between seeds. Some never pass two hops. One no-latent seed, with no latent steps at all, reaches ${pct(mat(ctrlN, "4")[2])} at four hops and ${pct(mat(ctrlN, "5")[2])} at five, matching the weakest recurrent seed. Both of Part I's single-seed control conclusions were seed luck.`);

  F2.push(`<b>Readable ≠ usable.</b> Editing along the probe's own read-out directions does nothing, even at a matched perturbation size where the branch's activation patterns move the answer 6–57× more. Probe filters hold 0.2–0.3% of the donor difference; patterns hold 35–53%.`);
  F2.push(`<b>Branch edits are real but weaken with width, and dose does not rescue them.</b> Raising the edit to 1.5× adds little at any width; at eight branches no seed reaches its two-branch effect (P2). The fraction of the clean counterfactual effect a thought edit recovers falls with width in every seed.`);
  F2.push(`<b>Why the branch average weakens.</b> Per deep node the edit does not weaken: at 5–6 hops a thought edit reproduces ${pc(deepT("2")[1])}–${pc(deepT("2")[2])} of the model's own counterfactual at W = 2 and ${pc(deepT("8")[1])}–${pc(deepT("8")[2])} at W = 8. The average falls for two reasons. The model's own counterfactual on deep nodes collapses with width (at four hops ${gapD("2","4").toFixed(2)} → ${gapD("8","4").toFixed(2)}). And two- and three-hop nodes move to query-time graph reading, so the latent-pathway share of the whole branch falls ${pc(latW2[0])} → ${pc(latW8[0])} while query reading rises ${pc(qW2[0])} → ${pc(qW8[0])}. The two routes together reproduce the whole clean effect. Pause-token models split it differently by seed: query-time graph reading carries ${pc(pauseQ[1])}–${pc(pauseQ[2])} across pause seeds and widths, with the rest in START's cache.`);
  F2.push(`<b>Edits have the right semantics.</b> An inserted branch continues with the recipient's graph (follow index up to +0.53, CI above zero at every frozen site and width), not the donor's endpoints. Wrong-branch donors do nothing. Two branches edited together each follow their own recipient continuation, slightly sub-additively.`);
  F2.push(`<b>Branches share a normalised, relative code.</b> Boosting one reachable branch's attention lowers the others; boosting a decoy or every edge barely matters; restoring total gain changes nothing. Per-branch amplitude falls as W<sup>−β</sup> with β = 0.62–0.87, between the 1/√W expected from a LayerNorm-normalised superposition and 1/W.`);
  F2.push(`<b>Scorecard.</b> Of eleven pre-registered predictions, ${Object.keys(CF.verdicts).filter(p => CF.verdicts[p].all === true).length} held for every model tested (${Object.keys(CF.verdicts).filter(p => CF.verdicts[p].all === true).join(", ")}). The rest held only partially or were falsified in part. Notably, local geometry <em>does</em> add some held-out predictive value for targeted effects (P8), one seed prefers 1/W over 1/√W (P10), and a two-hop-limited pause seed keeps branches in START's cache (P1b). See the scorecard.`);
  document.getElementById("findings-list").innerHTML = F2.map(x => `<li>${x}</li>`).join("");

  // ---------------- competence figure rebuilt: depth-matched balanced accuracy, all seeds
  (() => {
    const CMx = S.competence_matched || {};
    const host = P("fig-comp"); host.innerHTML = "";
    const ds = ["1","2","3","4","5","6"], xs = [1,2,3,4,5,6];
    const have = ns => ns.filter(n => CMx[n]);
    const band = (ns, col, name) => ({name, color: css(col), band: ns.length > 1,
      pts: ds.map((d, i) => { const v = mm(ns.map(n => CMx[n][d].bal_acc)); return [xs[i], v[0], v[1], v[2]]; })});
    const R_ = have(seeds), Pz = have(["pause", "pause1", "pause2"]), Kz = have(["k0", "k0_1", "k0_2"]);
    const series = [band(R_, "--s1", `Recurrent (${R_.length} seeds)`)];
    if (Pz.length) series.push(band(Pz, "--s2", `Pause tokens (${Pz.length} seed${Pz.length > 1 ? "s" : ""})`));
    if (Kz.length) series.push(band(Kz, "--s3", `No latent steps (${Kz.length} seed${Kz.length > 1 ? "s" : ""})`));
    P("lg-comp").innerHTML = series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
    lineChart(host, {x: xs, y:[0.4,1], yt:[0.5,.6,.7,.8,.9,1], yfmt: t => pct(t), tfmt: pct,
      xname:"distance", xtitle:"hops from start node", series, w:900, h:300, aria:"Depth-matched balanced accuracy, recurrent vs control seeds"});
    document.querySelector("#competence + .prose + figure .fig-title").textContent = "Depth-matched accuracy by distance from the start node";
    document.querySelector("#competence + .prose + figure .fig-sub").textContent = "At each distance: balanced accuracy of reachable-branch nodes vs decoy-branch nodes at the same within-branch depth (chance = 50%). Delayed format, held-out families, full latent budget.";
    P("cap-comp").textContent = `Mean over seeds with the seed range as a band. Controls replay the matched recurrent seed's realised curriculum schedule, with the same data and scaffold. Per-distance recall on reachable nodes alone (the Part I metric) can be inflated by a bias toward answering "reachable"; this matched metric cannot.`;
    const lo = (ns, d) => mm(ns.map(n => CMx[n][d].bal_acc));
    P("prose-competence").innerHTML = `<p>Every recurrent seed clears the 85% competence bar on held-out families (${pct(accMain[1])}–${pct(accMain[2])} overall balanced accuracy). With depth-matched decoys as negatives, they score ${pct(lo(R_, "5")[1])}–${pct(lo(R_, "5")[2])} at five hops and ${pct(lo(R_, "6")[1])}–${pct(lo(R_, "6")[2])} at six. At six hops every control seed is at or near chance (pause ${pct(lo(Pz, "6")[1])}–${pct(lo(Pz, "6")[2])}, no-latent ${pct(lo(Kz, "6")[1])}–${pct(lo(Kz, "6")[2])}).</p>
    <p>Below six hops the controls differ enormously between seeds. Two of three pause seeds and one no-latent seed never pass two hops. Others solve 3–4 hops through computation in the prefix, and one no-latent seed reaches ${pct(lo(Kz, "5")[2])} at five hops, as good as the weakest recurrent seed, with no latent steps at all. Part I's single pause seed happened to be a strong one and its single no-latent seed a weak one, so both control conclusions there were seed luck. What replicates is narrower: recurrent training reached the deepest queries in every seed, and no control seed reached six hops.</p>
    <p>Truncating the latent budget at inference is not a clean ablation. All recurrent seeds answer one-hop queries from the last latent position, so dropping any step collapses those queries. That is why the matched controls carry the latent-use claim.</p>`;
  })();
  // ---------------- Part II prose + scorecard
  P("prose-why").innerHTML = `<p>Part I left the mechanism behind the width effect open. Your handoff plan named three candidate explanations. <em>Dilution</em>: a smaller per-branch signal. <em>Interference</em>: overlapping branch directions. <em>Recovery</em>: the edited branch being recomputed from retained information. The follow-up tested each with a discriminating experiment, after a preflight. The preflight reproduced all 862 published point estimates exactly from the saved checkpoint and found no family overlap between splits. It also flagged that any W = 1 edit involves an untrained width-0 state, so W = 1 is now treated as a separate distribution-shift test. Predictions were written down before any fresh graph was drawn; the scorecard below shows which held.</p>`;
  const V = CF.verdicts;
  const desc = {P1: "Route shift: latent-pathway share falls and query-graph share rises from W = 2 to 8", P1b: "Pause models resolve ≥ 85% at query time",
    P2: "Scaling the edit to 1.5× at W = 8 does not reach the W = 2 effect", P3: "At matched size, pattern directions beat probe filters",
    P4: "Swaps cancel signed collateral but not its magnitude", P5: "Inserted branch follows the recipient continuation (CI > 0)",
    P6: "Boosting one reachable branch lowers the others; decoy/all-edge boosts ≪", P7: "Restoring total gain does not restore edits",
    P8: "Local geometry adds < 0.05 held-out R²", P9: "Composed edits: both follow, single edits specific, sub-additive", P10: "Per-branch amplitude ∝ 1/√W fits best", P14: "Erasing / doubling a branch's own component in the thought lowers / raises its deep nodes"};
  const models = ["s0", "s0@k0", "s1", "s2", "p0", "p1", "p2"];
  const rows = Object.keys(desc).filter(p => V[p]).map(p => [`<b>${p}</b>`, desc[p], ...models.map(m => chip(V[p].per_model[m]))]);
  table(P("tbl-score"), ["", "prediction", "seed 0", "seed 0 @k=0", "seed 1", "seed 2", "pause 0", "pause 1", "pause 2"], rows, []);

  // ---------------- dose
  const DC = FU.dose_curves;
  P("prose-dose").innerHTML = `<p>If edits weakened only because the per-branch signal is smaller, turning the edit up should restore them. It does not. At every width the full-vector effect saturates or grows slowly past the natural dose (1.0), and at eight branches no seed reaches its two-branch effect even at 1.5× (P2). As a fraction of the clean counterfactual effect (actually removing the branch), a thought edit recovers ${pc(mm(recOK.map(s => CF.recurrent[s].recovery_ratio["2"].full))[0])} at W = 2 and ${pc(mm(recOK.map(s => CF.recurrent[s].recovery_ratio["8"].full))[0])} at W = 8 (mean over seeds).</p>
  <p>The probe-filter subspace holds only 4–6% of the donor difference's norm, so a tiny effect alone would prove nothing. The comparison that matters is at matched perturbation size. The branch-pattern edit at dose 0.1 and the filter edit at dose 1.5 have similar norms, and the pattern edit moves the branch 6–57× more in every seed (P3). Direction, not size, is what the probe gets wrong.</p>`;
  (() => {
    const host = P("fig-dose");
    const doses = ["0.0","0.1","0.25","0.5","1.0","1.5"];
    W3.forEach(W => {
      const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
      const ser = (kind, col) => ({name: kind, color: css(col), band: true, pts: doses.map(l => { const v = mm(recOK.map(s => DC[s][W].remove[kind][l].targeted[0])); return [parseFloat(l), v[0], v[1], v[2]]; })});
      lineChart(d, {x: doses.map(parseFloat), y:[0, 0.45], yt:[0,.1,.2,.3,.4], xname:"dose", xtitle:"edit dose (× natural difference)", w:340, h:230,
        series:[ser("full","--s1"), ser("pattern","--s2"), ser("filter","--neutral")], aria:`Dose response W=${W}`});
    });
    P("cap-dose").textContent = `Clean counterfactual gap (actual removal) for reference: ${W3.map(W => `W = ${W}: ${mm(recOK.map(s => CF.recurrent[s].clean_gap[W][0]))[0].toFixed(2)}`).join(", ")}. Insertion edits show a threshold instead: almost nothing below dose 0.5, then a jump (tables in results/).`;
  })();

  // ---------------- routes
  P("prose-routes").innerHTML = `<p>The state an edit can touch is more than one vector. For a single-branch removal we patched donor values into one route at a time. The routes were: the thought; the START and early-latent cache entries that later positions read; the edge-token cache as read by later latent steps; and the same edge-token cache as read by the queries. The shares add up. Editing the latent pathway plus the queries' view of the graph reproduces the whole clean effect at every width, in every seed.</p>
  <p>The shares move with width (P1). At two branches most of a branch's effect travels through the latent pathway; at eight, most of it is resolved by queries reading the graph tokens directly. The breakdown by node distance below shows what that average hides. One-hop nodes are always resolved at query time, and nodes four or more hops away always through the latent pathway. What moves is the boundary, since two- and three-hop nodes switch from latent to query-time resolution at higher width. The amount of effect on deep nodes also shrinks, because the model itself becomes less decisive about them. The one competent pause seed resolves it that way at every width; a pause seed that never learned multi-hop reachability keeps about half of it in START's cache instead (P1b, partial). Editing the latent cache alongside the thought also stops the edited branch from reappearing in later thoughts. That is the recovery the plan asked to test causally, and it runs through the retained START and early-latent entries.</p>`;
  (() => {
    const Ws = [2,4,8];
    const mk = (c, col, name) => recOK.map((s, i) => ({name: `${name} ${s}`, color: css(col), pts: Ws.map(W => [W, CF.recurrent[s].route_shares[W][c], 0, 0]), label: s}));
    const pauseSer = Object.entries(CF.pause).map(([p, v]) => ({name: `pause ${p.slice(1)}`, color: css("--neutral"), pts: Ws.map(W => [W, v.route_shares[W].Gq, 0, 0]), label: `p${p.slice(1)}`}));
    lineChart(P("fig-route-lat"), {x: Ws, y:[0,1.05], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, xname:"W", xtitle:"width W", r:30,
      series: mk("TL", "--s1", "latent"), aria:"Latent pathway share by width"});
    lineChart(P("fig-route-q"), {x: Ws, y:[0,1.05], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, xname:"W", xtitle:"width W", r:30,
      series: [...mk("Gq", "--s2", "query"), ...pauseSer], aria:"Query-time graph share by width"});
    P("cap-routes").textContent = `One line per seed (labelled at the right). Latent pathway = thought edit plus the donor's START..START+k cache entries. Query-time graph = donor edge-token cache visible to the queries only. Patching graph tokens changes the effective input, so these shares locate where the difference is read, not where it is stored.`;
  })();
  (() => {
    const host = P("fig-depthroute");
    const ds = [1,2,3,4,5,6];
    W3.forEach(W => {
      const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
      const ser = (c, col, name) => ({name, color: css(col), band: true, pts: ds.map(dd => { const v = mm(recOK.filter(s => RD[s]).map(s => { const x = RD[s][W][dd]; return x && x.clean_gap > 0.05 ? x[c] : NaN; })); return [dd, v[0], v[1], v[2]]; })});
      lineChart(d, {x: ds, y:[-0.15, 1.25], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, zero:true, xname:"distance", xtitle:"hops from start", w:340, h:230,
        series:[ser("T","--s1","Thought only"), ser("TL","--s3","Latent pathway"), ser("Gq","--s2","Query-time graph")], aria:`Route share by distance W=${W}`});
    });
    const cg = W => ({name: `W = ${W}`, color: css({"2":"--s1","4":"--s2","8":"--s3"}[W]), band: true, pts: ds.map(dd => { const v = mm(recOK.filter(s => RD[s]).map(s => RD[s][W][dd].clean_gap)); return [dd, v[0], v[1], v[2]]; })});
    lineChart(P("fig-cleangap"), {x: ds, y:[0, 1.05], yt:[0,.25,.5,.75,1], xname:"distance", xtitle:"hops from start: clean counterfactual effect of removing the branch (ΔP)", w:900, h:220,
      series: W3.map(cg), aria:"Clean counterfactual effect by distance"});
    P("cap-depthroute").textContent = `Shares are shown only where the clean effect exceeds 0.05. Shares can exceed 100% or fall below 0 when routes overlap. This breakdown was not pre-registered; it re-analyses the confirmation graphs (the aggregate numbers reproduce exactly).`;
  })();
  (() => {
    const host = P("fig-margin");
    const M = FU.cache_margins.s0;
    W3.forEach(W => {
      const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
      const ks = [0,1,2,3,4,5,6];
      const ser = (c, col, name) => ({name, color: css(col), pts: ks.map(k => [k, M[W][c][k], 0, 0])});
      lineChart(d, {x: ks, y:[-0.1, 0.5], yt:[0,.2,.4], zero:true, xname:"thought", xtitle:"thought k", w:340, h:230, tfmt: v => v.toFixed(2),
        series:[ser("base","--neutral","Unedited"), ser("T","--s1","Thought edit"), ser("TL","--s2","Thought + latent cache"), ser("clamp","--s3","Clamped")], aria:`Margin trace W=${W}`});
    });
    P("cap-margin").textContent = `Seed 0 (k* = 1). After a thought-only edit the branch disappears from thoughts 2–3 and returns by thoughts 4–6. Adding the donor's START and LAT₁ cache entries prevents the return. Clamping the donor thought at every later step suppresses it until the final, unclamped thought. Other seeds: see results/followup_confirm.json.`;
  })();

  // ---------------- swaps
  P("prose-swaps").innerHTML = `<p>A removal changes the number of active branches; a swap does not. If collateral came from a shared normaliser that depends on the active count, swaps should remove it. They remove its <em>sign</em>: removal makes other branches look more reachable, insertion less, and a swap sits at zero in every seed and width. But unsigned collateral on the untouched branches stays about the size of a removal's at W = 4 and 8, and only shrinks at W = 2 in some seeds (P4, partial). So active-count normalisation explains the direction of the collateral, not most of its size. The clean counterfactuals show similar cross-branch changes: actually removing a branch moves the model's answers about the other branches by 0.03–0.08. Some interference is part of how the model computes, not an artefact of the edit.</p>`;
  (() => {
    const groups = [];
    W3.forEach(W => {
      [["swap","Swap"],["remove","Remove A"],["insert","Insert B"]].forEach(([c, n]) => {
        const u = mm(recOK.map(s => CF.recurrent[s].swap[W][c][0])), sg = mm(recOK.map(s => -CF.recurrent[s].swap[W][c][1]));
        groups.push({name: `${n}\nW = ${W}`, bars:[{name:"Unsigned", color: css("--s1"), v: u}, {name:"Signed", color: css("--s2"), v: sg}]});
      });
    });
    barChart(P("fig-swap"), {groups, y:[-0.08, 0.1], yt:[-0.06,-0.03,0,0.03,0.06,0.09], h:280, aria:"Swap vs removal collateral"});
    P("cap-swap").textContent = `Signed collateral plotted as (patched − unpatched) P(reachable) on untouched branches: positive = they became more reachable (after a removal), negative = less (after an insertion). Insertion at W = 8 creates a width-9 donor, outside the training range.`;
  })();

  // ---------------- semantics
  P("prose-sem").innerHTML = `<p>This test asks what an edit transfers. The recipient graph and the donor graph agree about branch B's upper part but disagree about what lies below depth two. There, a subtree is swapped with one from a decoy branch. We insert B's state from the donor and ask which future the model computes. If the edit transfers a branch state, the recipient's own continuation should become reachable. If it transfers the donor's pre-computed answers, the donor's endpoints should light up instead. In every seed, at every frozen site and width, the recipient continuation wins (P5): its CI lies above zero and donor endpoints barely move. A wrong-branch donor does nothing. Two branches edited at once each follow their own recipient continuation, and a single edit leaves the other branch alone (P9). The joint edit is slightly sub-additive, as competition predicts.</p>
  <p>Two limits. The branch-pattern subspace, which carries removals well, transfers insertions poorly, so the "on" and "off" states of a branch are not mirror images in the state. And at seed 0's non-frozen common site (k = 0), edits at W ≥ 4 are too weak for the test to discriminate.</p>`;
  (() => {
    const groups = [];
    W3.forEach(W => {
      const g2 = k => mm(recOK.map(s => CF.recurrent[s].continuation[`W${W}_j2`][k]));
      groups.push({name: `True donor\nW = ${W}`, bars:[{name:"Recipient continuation", color: css("--s1"), v: g2("edit_C_rec")}, {name:"Donor endpoints", color: css("--s2"), v: g2("edit_C_don")}]});
      const wr = mm(recOK.map(s => CF.recurrent[s].continuation[`W${W}_j2`].follow_wrong[0]));
      groups.push({name: `Wrong donor\nW = ${W}`, bars:[{name:"Follow index (wrong donor)", color: css("--neutral"), v: wr}]});
    });
    barChart(P("fig-sem"), {groups, y:[0, 0.6], yt:[0,.1,.2,.3,.4,.5,.6], h:260, aria:"Continuation test"});
    P("cap-sem").textContent = `Depth j = 2 shown; j = 3 agrees (tables in results/). For the wrong-branch donor the bar is the follow index (recipient − donor endpoints). Natural insertion raises the recipient continuation by ${W3.map(W => mm(recOK.map(s => CF.recurrent[s].continuation[`W${W}_j2`].natural_C_rec))[0].toFixed(2)).join(" / ")} at W = 2 / 4 / 8.`;
    const rows = [];
    recOK.forEach(s => { const c = CF.recurrent[s].composition; if (!c) return;
      W3.forEach(W => rows.push([`seed ${s.slice(1)}`, W, f3(c[W].both_f1[0]) + " / " + f3(c[W].both_f2[0]), f3(c[W].only1_f2) + " / " + f3(c[W].only2_f1), f3(c[W].additivity[0]) + " / " + f3(c[W].additivity[1])])); });
    table(P("tbl-pair"), ["seed", "W", "joint edit: follow B1 / B2", "single edit: other branch", "joint − single (B1 / B2)"], rows, [1,2,3,4]);
  })();

  // ---------------- mechanism
  const gm = s => CF.recurrent[s].gain_model;
  P("prose-mech").innerHTML = `<p>Two direct manipulations locate the competition. Adding a logit bias to the layer-1 attention onto one reachable branch's edges raises that branch and lowers the others, in proportion to the boost. The same bias on a decoy branch's edges, or on every edge at once, barely moves the reachable branches (P6 holds in two seeds; in seed 1 a uniform boost also lifts them slightly). What matters is a branch's share <em>relative to the other reachable branches</em>, not how much attention reaches edges overall. Consistently, restoring the reachable branches' total attention at W = 4 and 8 to its W = 2 level leaves edit effects unchanged (P7).</p>
  <p>The natural branch signal in the thought, |h(donor) − h(recipient)|, falls with width as W<sup>−β</sup>, with β = ${recOK.map(s => gm(s) ? gm(s).free_beta_fresh.toFixed(2) : "–").join(", ")} on fresh graphs. A LayerNorm-normalised superposition of roughly orthogonal branch codes, each taking about a 1/W attention share, predicts 1/√W. That model has the lowest held-out error in two seeds; seed 0 sits closer to 1/W (P10, partial). A shared, normalised code explains why each branch's signal weakens with width, and with it the model's own counterfactual sensitivity on deep nodes. Together with the inward move of the latent/query boundary, that accounts for the weaker branch-averaged edits.</p>`;
  (() => {
    const AD = FU.attention_doses;
    const bs = ["0.5","1.0","2.0"];
    const ser = (key, col, name) => ({name, color: css(col), band: true, pts: bs.map(b => { const v = mm(recOK.map(s => AD[s]["4"][b][key])); return [parseFloat(b), v[0], v[1], v[2]]; })});
    lineChart(P("fig-boost"), {x: bs.map(parseFloat), y:[-0.5, 0.05], yt:[-0.4,-0.3,-0.2,-0.1,0], zero:true, xname:"bias", xtitle:"logit bias b", h:250,
      series:[ser("att_boost_other_attached","--s1","reachable boost"), ser("dec_boost_attached","--s2","decoy boost"), ser("all_boost_attached","--s3","all-edge boost")], aria:"Attention boost effects"});
    P("cap-boost").textContent = `ΔP(reachable) on the untouched reachable branches. The boosted reachable branch itself rises by ${mm(recOK.map(s => AD[s]["4"]["1.0"].att_boost_self))[0].toFixed(3)} at b = 1.`;
    const Wg = [1,2,3,4,5,6,7,8];
    const cols = ["--s1","--s2","--s3"];
    const series = recOK.filter(s => gm(s)).map((s, i) => ({name: `seed ${s.slice(1)}`, color: css(cols[i]), pts: Wg.map(W => [Math.log2(W), Math.log2(gm(s).mean_diff_by_W[W]), 0, 0])}));
    const ref = recOK.filter(s => gm(s)).flatMap(s => { const a = Math.log2(gm(s).mean_diff_by_W["2"]);
      return [{name:"1/√W", color: css("--neutral-2"), pts: Wg.map(W => [Math.log2(W), a - 0.5 * (Math.log2(W) - 1), 0, 0])},
              {name:"1/W", color: css("--neutral-2"), pts: Wg.map(W => [Math.log2(W), a - (Math.log2(W) - 1), 0, 0])}]; });
    lineChart(P("fig-gain"), {x: Wg.map(W => Math.log2(W)), y:[1, 5], yt:[1,2,3,4,5], yfmt: t => (2 ** t).toFixed(0), tfmt: v => (2 ** v).toFixed(1),
      xlab: v => String(Math.round(2 ** v)), xname:"W", xtitle:"width W (log scale)", h:250, series:[...ref, ...series], aria:"Branch signal amplitude vs width"});
    P("cap-gain").textContent = `Held-out MSE on log amplitude (fresh graphs, intercept fitted on discovery): ${recOK.filter(s => gm(s)).map(s => `seed ${s.slice(1)}: 1/√W ${gm(s)["beta=0.5"].toFixed(3)}, 1/W ${gm(s)["beta=1.0"].toFixed(3)}, constant ${gm(s)["beta=0.0"].toFixed(3)}`).join("; ")}. Seed 0's jump between W = 2 and 3 reflects its later edit site (k* = 1).`;
  })();

  // ---------------- amplification (P14)
  (() => {
    const rows = recOK.filter(s => CF.recurrent[s].amplify).flatMap(s => ["4","8"].map(W => { const a = CF.recurrent[s].amplify[W];
      return [`seed ${s.slice(1)}`, W, f3(a.erase_own_deep[0]), f3(a.double_own_deep[0]), f3(a.double_others[0]), a.random_abs.toFixed(3)]; }));
    P("prose-mech").insertAdjacentHTML("beforeend", `<p>The same logic applies inside the thought itself (P14, pre-registered as addendum B). With no donor at all, we rescaled one reachable branch's own pattern component of the saved thought. Erasing it lowers that branch's nodes three or more hops away, and doubling it raises them, in every seed. The same operation on a random subspace of equal rank does almost nothing. Doubling one branch also lowers the other reachable branches in ${CF.P14_competition_W4_seeds} of 3 seeds at W = 4, so competition is visible in the thought, not only in the attention that builds it.</p>`);
    P("prose-mech").insertAdjacentHTML("afterend", `<div class="tbl" id="tbl-amp"></div>`);
    table(P("tbl-amp"), ["seed", "W", "erase own: ΔP (≥ 3 hops)", "double own: ΔP (≥ 3 hops)", "double own: ΔP other branches", "random subspace |ΔP|"], rows, [1,2,3,4,5]);
  })();

  // ---------------- controllability
  P("prose-ctrl").innerHTML = `<p>Following the plan's second approach, we linearised the forked query logits around the saved thought (Jacobian J, projected onto the shared 20-dimensional donor-difference subspace). For each branch we computed its response gain and the fraction of its response direction not shared with the other branches (ρ = ‖p‖/‖a‖, the part a perfectly selective edit could use). We then asked whether this geometry predicts held-out edit outcomes better than width, donor norm and clean confidence alone. For targeted effects it does, modestly, in two seeds (+0.07 and +0.11 R²; P8 falsified there). For collateral it adds nothing. The linearisation itself is poor at natural edit sizes, with predicted-vs-observed correlations of 0.15–0.31 and magnitudes overestimated. So the geometry is a weak, partial predictor, not a mechanism.</p>`;
  (() => {
    const rows = recOK.filter(s => CF.recurrent[s].controllability).map(s => { const c = CF.recurrent[s].controllability;
      return [`seed ${s.slice(1)}`, c.obs_target_prob.baseline.toFixed(2), c.obs_target_prob.geometry.toFixed(2), c.obs_col_prob.baseline.toFixed(2), c.obs_col_prob.geometry.toFixed(2),
        c.linear_validity_corr_target.toFixed(2), W3.map(W => c.resid_ratio_by_W[W].toFixed(2)).join(" / ")]; });
    table(P("tbl-ctrl"), ["seed", "targeted R²: baseline", "+ geometry", "collateral R²: baseline", "+ geometry", "linear validity r", "ρ at W = 2 / 4 / 8"], rows, [1,2,3,4,5,6]);
  })();

  // ---------------- replication table
  P("prose-rep").innerHTML = `<p>Part I's controls were single seeds. The replications replay each recurrent seed's exact curriculum schedule. They overturn both of Part I's single-seed control conclusions. Some control seeds never pass two hops, while one no-latent seed solves up to five hops through the prefix alone; only six-hop queries separate every recurrent seed from every control. At the cache interface, where a pause model keeps a branch depends on what it learned. The competent pause seed resolves branches at query time from the graph tokens; a pause seed limited to two hops keeps about half of the effect in START's cache.</p>`;
  (() => {
    const CMx = S.competence_matched || {};
    const lab_ = n => ({s0: "recurrent seed 0", s1: "recurrent seed 1", s2: "recurrent seed 2", pause: "pause seed 0", pause1: "pause seed 1", pause2: "pause seed 2", k0: "no-latent seed 0", k0_1: "no-latent seed 1", k0_2: "no-latent seed 2"})[n] || n;
    const rows = [...seeds, ...PAUSE_N, ...K0_N].map(n => [lab_(n), n]).map(([lab, n]) =>
      [lab, pct(balD(C[n])), pct(balU(C[n])), ...["2","3","4","5","6"].map(d => CMx[n] ? pct(CMx[n][d].bal_acc) : "–")]);
    table(P("tbl-rep"), ["model", "delayed bal. acc", "upfront", "2 hops*", "3*", "4*", "5*", "6*"], rows, [1,2,3,4,5,6,7]);
    P("tbl-rep").insertAdjacentHTML("beforeend", `<p class="fig-sub">* depth-matched balanced accuracy (reachable vs same-depth decoy nodes; chance 50%).</p>`);
  })();

  // ---------------- claims table rebuilt
  (() => {
    const host = P("tbl-claims"); host.innerHTML = "";
    const yes = t => `<span class="chip yes">${t}</span>`, no = t => `<span class="chip no">${t}</span>`, midc = t => `<span class="chip">${t}</span>`;
    const rows = [
      ["Recurrent latent computation is needed for the deepest queries", "Six hops: every recurrent seed above, every control seed at or near chance (depth-matched); five hops: one no-latent seed matches the weakest recurrent seed", midc("supported at six hops, partial at five"), "Three seeds per model type; controls are 2-layer like the recurrent models"],
      ["Extra positions or feedback are needed for 3–5 hop queries", "Some no-latent seeds solve 3–5 hops through prefix computation; other control seeds fail beyond two", no("not supported"), "Competence of controls is strongly seed-dependent"],
      ["Several branches in one saved state can be edited separately", "Effect matrices; targeted > collateral at W ≥ 3; composed edits branch-specific", yes("supported, partial"), "Pre-set selectivity bar met by one seed at W = 2 only"],
      ["Edits transfer a branch state, not the donor's answers", "Conflicting-continuation test, all seeds and widths at frozen sites", yes("supported"), "Pattern subspace transfers insertions poorly"],
      ["Probe read-out directions are not the causally used ones", "Filter vs pattern at matched size; energy fractions", yes("supported"), "Linear probes only"],
      ["Branch-averaged edits weaken because deep-node counterfactual sensitivity collapses and 2–3-hop nodes move to query-time resolution; per deep node, edit efficacy does not decline", "Route decomposition by node distance (post-hoc, fresh graphs), all seeds", yes("supported (post-hoc)"), "Depth analysis not pre-registered; graph-route patches change the effective input"],
      ["Dilution by magnitude explains the width effect", "Dose and gain-restoration tests", no("not supported"), "Tested edit family only"],
      ["Branches share a normalised, relative code", "Relative attention boosts; signed collateral; W^−β amplitude", midc("supported, partly quantitative"), "β between 0.5 and 1; one seed prefers 1/W"],
      ["Retained cache entries reconstruct an edited branch", "Thought-only edit: branch returns; + latent-cache edit: no return", yes("supported"), "Shown most clearly for seed 0"],
      ["Local linear controllability predicts selective-edit outcomes", "Held-out R², linear validity", midc("weak, targeted only"), "Linearisation poor at natural edit sizes"],
      ["The mechanism emerges without supervision of intermediate steps", "Lookup scaffold + depth curriculum were required", no("not claimed"), "Scaffold may shape the solution"],
    ];
    table(host, ["claim", "evidence", "status", "main remaining limitation"], rows, []);
  })();

  // ---------------- limits (append)
  P("prose-limits").insertAdjacentHTML("beforeend", `<p><b>Part II specifics.</b> Graph-route patches change what the model reads as input, so route shares say where the difference is used, not where it is stored. The attention tests bias one layer (all heads) rather than a single identified head. The W = 2 composed-edit recipients have width 0, outside the training range. Insertion at W = 8 creates a width-9 donor. The discovery-derived predictions (P1, P3, P4, P6–P9) confirm discovery observations on fresh graphs rather than testing new hypotheses.</p>`);
}
