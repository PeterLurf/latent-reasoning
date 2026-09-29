// ---------------------------------------------------------------- data helpers
const RUNS = {s0: "runs/r3_pc_gated_gpt2", s1: "runs/main_s1", s2: "runs/main_s2"};
const C = S.competence;
const compMain = seeds.map(s => C[s]).filter(Boolean);
const topK = c => c[String(Math.max(...Object.keys(c).map(Number)))];
const byDepth = (c, d) => topK(c).delayed.by_depth[d];
const balD = c => topK(c).delayed.bal_acc[0], balU = c => topK(c).upfront.bal_acc[0];
const rng3 = a => [mean(a), Math.min(...a), Math.max(...a)];
const kstar = s => S.seeds[s].k_star;
const energy = (s, key) => X[RUNS[s]] ? X[RUNS[s]].energy_W4[key] : NaN;
const eW = (c, W, m) => eff(c, String(W), m);
const pr = v => (v[1] === v[2]) ? f3(v[0]) : `${f3(v[0])} (seeds ${f3(v[1])} to ${f3(v[2])})`;
const pr2 = v => `${v[0].toFixed(2)} (range ${v[1].toFixed(2)}–${v[2].toFixed(2)})`;
const has = n => !!C[n];
document.getElementById("m-seeds").textContent = `${seeds.length} training seeds + pause and no-latent controls`;

// ---------------------------------------------------------------- prose
const P = id => document.getElementById(id);
const accMain = rng3(compMain.map(balD));
const accK0 = has("k0") ? balD(C.k0) : NaN, accPause = has("pause") ? balD(C.pause) : NaN;
const deep = (c) => mean(["5","6"].map(d => byDepth(c, d)));
const deepMain = rng3(compMain.map(deep));
const t2 = eW("full|remove|hub", 2, "targeted"), t8 = eW("full|remove|hub", 8, "targeted");
const c2 = eW("full|remove|hub", 2, "collateral_attached"), c8 = eW("full|remove|hub", 8, "collateral_attached");
const eP = rng3(seeds.map(s => energy(s, "branch patterns (6)"))), eF = rng3(seeds.map(s => energy(s, "probe filters (6)"))),
      eR = rng3(seeds.map(s => energy(s, "random (6)")));
const probeMax = Math.max(...seeds.flatMap(s => WS.map(W => { const e = S.seeds[s].effects["probe|remove|hub"][W]; return e ? Math.abs(e.targeted[0]) : 0; })));

P("prose-abstract").innerHTML = `<p>A continuous-thought transformer can hold several graph-search branches in one latent state. We asked whether each held branch can also be <em>edited on its own</em>: whether swapping in a single-edge counterfactual for one branch changes answers about that branch while sparing the others. First, a control: most of the task (up to four hops) is solved just as well with extra pause positions and no hidden-state feedback, so the recurrence matters mainly for the deepest queries. On the branch question the answer is: partly. In most conditions an edit moves the edited branch more than the others, and it acts only on nodes that the latent steps still have to reach. But the effect of each edit shrinks steadily as more branches share the state, and branches compete: removing one makes the others look <em>more</em> reachable. The directions a linear probe reads the branches from are almost causally inert; the directions the branches actually vary along carry the effect.</p>`;

P("prose-status").innerHTML = `<strong>Status.</strong> Completed study, one architecture, one synthetic task. Protocol frozen from discovery-split data before confirmation; all effects below are on held-out graph families (confirm split), 300 graphs per width, with percentile bootstrap over graphs inside each seed and the seed range shown across ${seeds.length} independently trained models. Numbers are probability changes on the model's P(reachable). Training used a documented lookup scaffold (see <a href="#limits">Limits</a>), so emergence claims are restricted accordingly.`;

const F = [];
F.push(`<b>Extra positions are needed; continuous feedback helps only at the deepest queries.</b> All ${seeds.length} recurrent seeds reach ${pct(accMain[1])}–${pct(accMain[2])} balanced accuracy on held-out families; a matched no-latent model reaches ${pct(accK0)} and almost always calls nodes 3+ hops away unreachable. A matched pause-token model, with the same extra positions but no hidden-state feedback, reaches ${pct(accPause)} overall and is as good or better up to four hops. It falls to ${pct(byDepth(C.pause,"5"))}–${pct(byDepth(C.pause,"6"))} at five and six hops, where the recurrent seeds reach ${pct(deepMain[1])}–${pct(deepMain[2])}.`);
F.push(`<b>Readable ≠ usable.</b> Replacing a branch along the probe's own read-out directions changes nothing (largest effect across all seeds and widths ${f3(probeMax)}). Those six directions hold ${(100*eF[0]).toFixed(1)}% of the donor difference, <em>less</em> than a random six-dimensional subspace (${(100*eR[0]).toFixed(1)}%). The branch's activation patterns, the same six node ids read through the activation covariance, hold ${(100*eP[1]).toFixed(0)}–${(100*eP[2]).toFixed(0)}% and reproduce most of the full-vector effect.`);
F.push(`<b>Branch edits work, but less as width grows.</b> A full-vector single-branch removal moves the edited branch by ${pr(t2)} at W = 2 and ${pr(t8)} at W = 8. Probe decodability declines too, but more slowly: readability outlasts separate control.`);
F.push(`<b>Edits act only on what the latent steps still have to compute.</b> Nodes 1–2 hops from the start are untouched (≤ 0.05) by an edit at the frozen site; the effect concentrates on nodes 3–6 hops away, up to 0.6 in probability at W = 2.`);
F.push(`<b>Branches compete for a shared budget.</b> Removing one branch makes the other reachable branches look <em>more</em> reachable and inserting one makes them look less, a signed pattern predicted before confirmation. Attention from the latent positions onto each branch's edges falls roughly as 1/W.`);
F.push(`<b>Pre-set selectivity bar met once.</b> The proposal's bar (≥ 0.10 targeted with ≤ 0.03 collateral, pooled over a branch) is met only by one seed at W = 2. Effect sizes vary about 4× between seeds; the qualitative pattern is what replicates.`);
document.getElementById("findings-list").innerHTML = F.map(x => `<li>${x}</li>`).join("");

P("prose-setup").innerHTML = `<p><b>Task.</b> Directed reachability on fixed-budget graphs: 10 disjoint branches, each a random six-node tree, 62 nodes and 60 edge tokens per graph in shuffled order, with ids drawn fresh from 80 per graph. W ∈ {1…8} branches hang off the start node; the rest hang off a decoy hub, so every unreachable node has a shape-matched reachable counterpart. Structural families are hashed into train / discovery / confirm splits so a renamed or reordered copy can never cross splits.</p>
<p><b>Model.</b> Two-layer pre-LN decoder, d = 256, trained from scratch. Coconut-style recurrence: the final-layer output at each of K = 6 latent positions becomes the next position's input. In the <em>delayed</em> format the graph and start are processed, six latent steps run, and only then is a target node supplied; many targets are asked as independent forks of the identical saved state, which is what makes "several branches in one state" testable.</p>
<p><b>Intervention.</b> For one reachable branch, the donor is the same graph with that branch's single hub edge moved to the decoy hub. At thought h<sub>k</sub> we set h ← h + UUᵀ(h<sub>donor</sub> − h) and recompute everything after position k, keeping the cache before it. U is the full space, a branch-specific pattern subspace (six directions indexed by the edited branch's node ids), the probe's own read-out filters for those ids, a shared PCA subspace of discovery donor differences, or matched random controls. The frozen site k* is chosen per seed on discovery graphs by a pre-set rule (seed 0: k* = ${kstar("s0")}; seeds 1–2: k* = ${kstar("s1")}, ${kstar("s2")}).</p>`;

P("prose-competence").innerHTML = `<p>Every trained seed clears the proposal's 85% competence bar on held-out families (${pct(accMain[1])}–${pct(accMain[2])}). Latent computation matters where the task needs depth: the separately trained no-latent model is perfect at one hop but answers queries three or more hops away at ${pct(byDepth(C.k0,"3"))}–${pct(byDepth(C.k0,"6"))}${has("pause") ? `. The pause-token control, which gets the same extra positions but no hidden-state feedback, reaches ${pct(accPause)} overall. It is strong to four hops and falls off at five and six (${pct(byDepth(C.pause,"5"))}, ${pct(byDepth(C.pause,"6"))} vs ${pct(deepMain[1])}–${pct(deepMain[2])} averaged over those depths for the recurrent seeds). So continuous feedback helps most for the deepest queries rather than across the board` : ""}.</p>
<p>Truncating the latent budget at inference is <em>not</em> a clean ablation here. All seeds answer one-hop queries from the last latent position, so dropping any step collapses those queries. That is why the matched controls, not truncation, carry the latent-use claim.</p>`;
P("cap-comp").textContent = `Recurrent seeds: mean with the seed range as a band. Controls are single seeds with the same data, scaffold and curriculum schedule. Queries are balanced 50/50 reachable/unreachable; "unr." is accuracy on unreachable queries.`;

P("prose-readable").innerHTML = `<p>A linear probe reads each node's reachability from the thought above chance, and above label-shuffled and untrained-model controls, which sit at 0.49–0.52 AUC. So the branches are <em>readable</em>. The obvious next step, editing a branch along the probe's read-out directions, fails completely. The probe's weights are <em>filters</em> that separate the classes against structured noise. They are nearly orthogonal to the <em>pattern</em> along which the thought actually moves when the branch is detached, which is the familiar filter/pattern distinction from neuroimaging decoders (Haufe et al., 2014). Reading the same six node ids through the activation covariance gives branch-specific pattern directions, and those carry most of the full-vector effect.</p>
<p>The effect is also not generic perturbation. A random direction with the full difference's norm moves the targeted branch by at most ${f3(Math.max(...seeds.map(s => S.seeds[s].effects["normrand|remove|hub"]["4"].targeted[0])))} while still producing collateral. Rank-matched random subspaces capture about 2% of the difference and do nothing. A shared 20-dimensional PCA subspace of discovery differences reproduces the full effect.</p>`;
P("cap-sub").textContent = `Each seed at its own frozen site k*. "Branch patterns" and "probe filters" use the same six node ids and the same rank; they differ only in whether the covariance is applied. Isotropic = random direction with the norm of the full donor difference.`;

P("prose-width").innerHTML = `<p>Across widths, the targeted effect of a single-branch edit falls monotonically from W = 2 to W = 8 in every seed. Collateral on the untouched branches falls too, but not as fast, so the targeted share of the change shrinks. Per-branch probe AUC also declines with width, from 0.58–0.62 at W = 2 to about 0.55 at W = 8. Indexed to W = 2, readability keeps ${pct(mean(seeds.map(s => (S.seeds[s].decode.branch_auc_by_width_at_kstar["8"]-0.5)/(S.seeds[s].decode.branch_auc_by_width_at_kstar["2"]-0.5))))} of its above-chance margin at W = 8, while the targeted effect keeps ${pct(mean(seeds.map(s => S.seeds[s].effects["full|remove|hub"]["8"].targeted[0]/S.seeds[s].effects["full|remove|hub"]["2"].targeted[0])))}. This is the pattern the proposal set out to test: decodability outlasts demonstrated separate use.</p>`;
P("cap-w").textContent = `Collateral is undefined at W = 1 (no other reachable branch). Seed ranges are wide: seed 2's effects are 3–4× seed 0's at every width, but all three decline.`;
P("cap-index").textContent = `Indexed at W = 2 because W = 1 has no competing branch (and one seed's W = 1 effect is anomalously small). Per-seed index, then mean across seeds; band = seed range.`;

P("prose-depth").innerHTML = `<p>Averaging over a whole branch hides most of the effect. At the frozen site, nodes one hop from the start are untouched and nodes two hops away nearly so (≤ 0.05), because they are answered from positions the edit leaves in the cache. The effect sits on nodes three or more hops away, the ones the remaining latent steps must still reach. Timing agrees: edits at the earliest thoughts matter most, and edits at k ≥ 4 do almost nothing in two of three seeds. That fits a frontier that is expanded over the first few steps and then read, and it gives a precise answer to "which alternatives are in play at step k".</p>`;
P("cap-heat").textContent = `Mean over seeds, each at its own k*. Distance 1–2 cells are ≈ 0 in every seed. Darker = larger drop in P(reachable) toward the counterfactual "unreachable".`;
P("cap-steps").textContent = `k = 0 is the START output, before any latent position. Mean over seeds; band = seed range.`;

P("prose-competition").innerHTML = `<p>If branches shared the thought as independent slots, removing one should leave the others alone. Instead the others move in a consistent direction. After a removal they become more reachable; after an insertion, less. The signed shift is largest at small width and shrinks roughly like the change in a 1/W share would. This sign pattern was written down as a prediction before any confirmation graph was drawn, and it holds in two seeds at every width and in the third at small and large widths. The identity control is exactly zero and a same-status control (one decoy edge moved, nothing's reachability changed) stays at 0.022 or less, so this is not generic damage.</p>
<p>A direct look at attention agrees (exploratory, discovery graphs). The first layer's attention from the latent positions onto each reachable branch's edges, above the decoy baseline, is split among the branches. Each branch's share falls at least as fast as 1/W. A fixed attention budget divided among the alternatives is a simple mechanism for the interference.</p>`;
P("cap-signed").textContent = `Mean over seeds; band = seed range. Plotted: patched − unpatched P(reachable), averaged over the other reachable branches' nodes. Above zero = the untouched branches became more reachable.`;
P("cap-attn").textContent = `Per-branch excess attention (reachable − decoy branch), layer 1, averaged over latent positions 1–3. Grey line: 1/W scaled to the W = 1 value. Mean over seeds; band = seed range.`;

P("prose-recovery").innerHTML = `<p>Tracing the edited branch through later thoughts shows why single-site edits are modest. Immediately after a full-vector edit, the probe no longer separates the edited branch from decoys. In two seeds the branch then partly reappears over the next steps, because the graph and the cached early positions are still there to recompute it from. In the third seed it stays suppressed. A null effect at one interface therefore would not have shown the branch was unused, and the positive effects here are lower bounds on what the state carries.</p>`;
P("cap-trace").textContent = `Margin = mean probe logit of the edited branch's nodes minus decoy nodes, per later step. Seeds with k* = 0 contribute from step 1, the seed with k* = 1 from step 2.`;

P("prose-upfront").innerHTML = `<p>When the target is given before the latent steps, the thoughts become target-conditioned. Editing the target's own branch moves the answer strongly (0.14–0.74 depending on seed) while edits to the other branches barely move it (≤ 0.03 at all but one seed-width cell). The delayed-format pattern subspace is nearly inert here, so a target-conditioned state encodes the branches along different directions. This is conventional reachability, and it shows branch-level edits matter in ordinary use. It does not by itself show simultaneous alternatives, since only one branch is relevant to the answer.</p>`;

P("prose-limits").innerHTML = `<p><b>Training scaffold.</b> Plain end-to-end training never left chance (every variant, 3–5k steps). The cause was diagnosed as a saddle: two identity-matching circuits that each need the other before the loss moves. Training therefore used auxiliary PARENT(v) and CHILDREN(v) lookup queries, read against the prefix only and never asked about the start node, annealed to zero by step 15k, plus an accuracy-gated depth curriculum. No loss ever touched latent thoughts or the frontier. The scaffold may favour query-time lookup, and claims about <em>emergence</em> of the mechanism should be read with that in mind.</p>
<p><b>Discovery-driven conditions.</b> The pattern and PCA subspaces were added after discovery showed the probe-filter condition was inert. They were frozen, with the site rule, before confirmation. Two pipeline smoke tests on <em>intermediate</em> checkpoints touched confirm-split graphs before sampling was routed through the protocol; no choice was changed as a result, and the event is logged.</p>
<p><b>Scope.</b> One two-layer architecture, disjoint branches, one intervention interface (the thought fed to the next latent position). The frozen site differs between seeds. Deep-query competence is partial (0.4–0.8 at 5–6 hops), so effects on those nodes act on uncertain baseline predictions. The pause control is a single seed.</p>
<p><b>What would change the conclusion.</b> (i) A decoder whose weights, used as edit directions, controlled the branch, which would undercut the filter/pattern account. (ii) Selectivity that stayed flat with width at a different interface, such as edits applied to all later positions or to cache entries, which would make the decline an interface artifact rather than interference in the state. (iii) Signed collateral vanishing in more seeds, which would weaken the shared-budget account.</p>
<p><b>Next.</b> Multi-site edits (thought plus cache) to bound recovery; a formal model of the 1/W attention budget and its predicted selectivity curve; merging branches; a second architecture.</p>`;

P("prose-repro").innerHTML = `<p>Everything is in the project directory. <code>python -m latent.tests</code> validates the instrument: generator vs exact BFS, single-token counterfactual locality, cache vs full recompute, forked queries vs separate sequences, identity patch exactness. <code>docs/PROTOCOL.md</code> and <code>docs/protocol_frozen.json</code> hold the pre-registration, <code>docs/RESEARCH_LOG.md</code> records every decision including the failed pilots, and <code>results/summary.json</code> holds the numbers behind every figure here. Hardware: RTX 4060, RTX 3070, RTX 2070 (8 GB each); about 2 GB and roughly 30–65 minutes per 30k-step run.</p>`;

// ---------------------------------------------------------------- figures
// diagram
P("fig-diagram").innerHTML = `
<svg viewBox="0 0 900 230" class="chart diagram" role="img" aria-label="Task and intervention diagram">
  <g font-family="var(--sans)">
    ${[0,1,2].map(i => `<line x1="60" y1="115" x2="150" y2="${50+i*65}" stroke="${i===1 ? "var(--s2)" : "var(--axis)"}" stroke-width="${i===1?2:1.5}"/>`).join("")}
    <line x1="60" y1="200" x2="150" y2="${50+65}" stroke="var(--s2)" stroke-width="2" stroke-dasharray="4 3"/>
    ${[0,1,2].map(i => [0,1,2].map(j => `<circle cx="${150+j*55}" cy="${50+i*65}" r="9" fill="${i===1 ? "var(--s2)" : "var(--s1)"}" stroke="var(--surface)" stroke-width="2"/>` + (j<2 ? `<line x1="${159+j*55}" y1="${50+i*65}" x2="${196+j*55}" y2="${50+i*65}" stroke="var(--axis)" stroke-width="1.5"/>` : "")).join("")).join("")}
    <circle cx="60" cy="115" r="13" fill="var(--ink)"/><text x="60" y="119" text-anchor="middle" style="fill:var(--ground);font-weight:600">s</text>
    <circle cx="60" cy="200" r="11" fill="var(--neutral)"/><text x="60" y="204" text-anchor="middle" style="fill:var(--ground)">s′</text>
    <text x="150" y="22" class="t-ink">W reachable branches</text>
    <text x="300" y="119" style="font-size:11px">edited branch i</text>
    <text x="86" y="224" style="font-size:11px">donor: s→rootᵢ becomes s′→rootᵢ</text>
  </g>
  <g transform="translate(440,40)" font-family="var(--sans)">
    ${["edges","S", "L1","L2","L3","L4","L5","L6"].map((t,i) => { const x = i===0 ? 0 : 60 + (i-1)*46, w = i===0 ? 52 : 38;
      const hot = i >= 3; return `<rect x="${x}" y="40" width="${w}" height="30" rx="4" fill="${i===0 ? "var(--grid)" : (hot ? "var(--accent-wash)" : "var(--grid)")}" stroke="${i===2 ? "var(--s2)" : "none"}" stroke-width="2"/><text x="${x+w/2}" y="59" text-anchor="middle" class="t-ink">${t}</text>`; }).join("")}
    ${[0,1,2].map(i => `<rect x="${60+7*46}" y="${20+i*26}" width="60" height="20" rx="4" fill="var(--surface)" stroke="var(--axis)"/><text x="${60+7*46+30}" y="${34+i*26}" text-anchor="middle" style="font-size:11px">query ${["a","b","c"][i]}</text>`).join("")}
    <path d="M ${60+46+19} 80 C ${60+46+19} 110, ${60+2*46+19} 110, ${60+2*46+19} 74" fill="none" stroke="var(--s2)" stroke-width="2"/>
    <text x="${60+46-6}" y="128" style="font-size:11px" class="t-ink">h₁ ← h₁ + UUᵀ(h₁ᵈᵒⁿ − h₁)</text>
    <text x="0" y="22" style="font-size:11px">kept in cache</text>
    <text x="${60+2*46}" y="22" style="font-size:11px">recomputed</text>
    <text x="0" y="160" style="font-size:11px">every query is an independent fork of the same saved state</text>
  </g>
</svg>`;

// competence
(() => {
  const ds = ["1","2","3","4","5","6","99"];
  const xs = [1,2,3,4,5,6,7];
  const series = [{name:"Recurrent (3 seeds)", color:css("--s1"), band:true,
    pts: ds.map((d,i) => { const v = rng3(compMain.map(c => byDepth(c, d))); return [xs[i], v[0], v[1], v[2]]; })}];
  if (has("pause")) series.push({name:"Pause tokens", color:css("--s2"), pts: ds.map((d,i) => [xs[i], byDepth(C.pause, d), 0, 0])});
  if (has("k0")) series.push({name:"No latent steps", color:css("--s3"), pts: ds.map((d,i) => [xs[i], byDepth(C.k0, d), 0, 0])});
  P("lg-comp").innerHTML = series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  lineChart(P("fig-comp"), {x: xs, y:[0,1], yt:[0,.25,.5,.75,1], yfmt: t => pct(t), tfmt: pct, xlab: v => v===7 ? "unr." : v,
    xname:"distance", xtitle:"hops from start node", series, w:900, h:300, aria:"Accuracy by distance for recurrent, pause and no-latent models"});
  const rows = [];
  for (const [name, c] of Object.entries(C)) for (const [Kp, b] of Object.entries(c))
    rows.push([name.startsWith("k0") ? `no-latent seed ${name === "k0" ? 0 : name.slice(3)}` : name.startsWith("pause") ? `pause seed ${name === "pause" ? 0 : name.slice(5)}` : "recurrent seed " + name.slice(1), Kp, pct(b.delayed.bal_acc[0]) + ` [${pct(b.delayed.bal_acc[1])}, ${pct(b.delayed.bal_acc[2])}]`, pct(b.upfront.bal_acc[0])]);
  table(P("tbl-comp"), ["model", "latent budget at inference", "delayed bal. acc [95% CI]", "upfront bal. acc"], rows, [1,2,3]);
})();

// subspace bars (W = 4)
(() => {
  const conds = [["full|remove|hub","Full vector"],["pattern|remove|hub","Branch patterns\n6 dims"],["pca20|remove|hub","Shared PCA\n20 dims"],
    ["pca6|remove|hub","Shared PCA\n6 dims"],["probe|remove|hub","Probe filters\n6 dims"],["random6|remove|hub","Random\n6 dims"],
    ["random20|remove|hub","Random\n20 dims"],["normrand|remove|hub","Isotropic\nnorm-matched"]];
  const groups = conds.map(([c, n]) => ({name:n, bars:[{name:"Targeted", color:css("--s1"), v: eW(c,4,"targeted")},
    {name:"Collateral", color:css("--s2"), v: eW(c,4,"collateral_attached")}]}));
  const ymax = Math.max(...groups.flatMap(g => g.bars.map(b => b.v[2]))) * 1.1;
  const step = ymax > 0.2 ? 0.05 : 0.02;
  barChart(P("fig-sub"), {groups, y:[0, Math.ceil(ymax/step)*step], yt: Array.from({length: Math.ceil(ymax/step)+1}, (_,i) => i*step), aria:"Targeted and collateral effects by subspace at width 4"});
  const rows = conds.map(([c,n]) => [n.replace("\n"," "), ...seeds.map(s => { const e = S.seeds[s].effects[c]["4"]; return `${f3(e.targeted[0])} / ${e.collateral_attached[0].toFixed(3)}`; }),
    seeds.map(s => ({"full|remove|hub":null, "pattern|remove|hub":"branch patterns (6)", "probe|remove|hub":"probe filters (6)", "pca6|remove|hub":"PCA (6)", "pca20|remove|hub":"PCA (20)", "random6|remove|hub":"random (6)"})[c] ? pct(energy(s, ({"pattern|remove|hub":"branch patterns (6)", "probe|remove|hub":"probe filters (6)", "pca6|remove|hub":"PCA (6)", "pca20|remove|hub":"PCA (20)", "random6|remove|hub":"random (6)"})[c])) : "–").join(" · ")]);
  table(P("tbl-sub"), ["condition", ...seeds.map(s => `seed ${s.slice(1)}: targeted / collateral`), "energy of donor difference captured (s0 · s1 · s2)"], rows, [1,2,3]);
})();

// width
(() => {
  const mk = (c, m, ws) => ws.map(W => { const v = eW(c, W, m); return [W, v[0], v[1], v[2]]; });
  lineChart(P("fig-w-tgt"), {x: WS, y:[0, 0.35], yt:[0,.1,.2,.3], xname:"W", xtitle:"width W", aria:"Targeted effect vs width",
    series:[{name:"Full vector", color:css("--s1"), band:true, pts: mk("full|remove|hub","targeted",WS)},
            {name:"Branch patterns", color:css("--s2"), band:true, pts: mk("pattern|remove|hub","targeted",WS)}]});
  const W2 = WS.slice(1);
  lineChart(P("fig-w-col"), {x: W2, y:[0, 0.1], yt:[0,.03,.06,.09], xname:"W", xtitle:"width W", aria:"Collateral vs width",
    series:[{name:"Full vector", color:css("--s1"), band:true, pts: mk("full|remove|hub","collateral_attached",W2)},
            {name:"Branch patterns", color:css("--s2"), band:true, pts: mk("pattern|remove|hub","collateral_attached",W2)}]});
  const idx = (fn) => W2.map(W => { const v = rng3(seeds.map(s => fn(s, W))); return [W, v[0], v[1], v[2]]; });
  lineChart(P("fig-index"), {w:900, h:280, x: W2, y:[0, 1.3], yt:[0,.25,.5,.75,1,1.25], yfmt: t => t.toFixed(2), tfmt: v => v.toFixed(2), xname:"W", xtitle:"width W",
    aria:"Readability vs usability indexed to W=2",
    series:[{name:"Readable (AUC − 0.5)", color:css("--s3"), band:true, pts: idx((s,W) => (S.seeds[s].decode.branch_auc_by_width_at_kstar[W]-0.5)/(S.seeds[s].decode.branch_auc_by_width_at_kstar["2"]-0.5))},
            {name:"Usable (targeted)", color:css("--s1"), band:true, pts: idx((s,W) => S.seeds[s].effects["full|remove|hub"][W].targeted[0]/S.seeds[s].effects["full|remove|hub"]["2"].targeted[0])}]});
  const rows = WS.map(W => [W, ...seeds.map(s => { const e = S.seeds[s].effects["full|remove|hub"][W], p = S.seeds[s].effects["pattern|remove|hub"][W];
    return `${f3(e.targeted[0])} / ${isNaN(e.collateral_attached[0]) || e.collateral_attached[0]==null ? "–" : e.collateral_attached[0].toFixed(3)} · ${f3(p.targeted[0])}`; }),
    seeds.map(s => S.seeds[s].decode.branch_auc_by_width_at_kstar[W].toFixed(3)).join(" · ")]);
  table(P("tbl-w"), ["W", ...seeds.map(s => `seed ${s.slice(1)}: full tgt / col · pattern tgt`), "branch AUC (s0 · s1 · s2)"], rows, [0,1,2,3,4]);
})();

// depth heatmap
(() => {
  const rows = ["1 hop","2 hops","3 hops","4 hops","5 hops","6 hops"];
  heatmap(P("fig-heat"), {rows, cols: WS, max: 0.6, aria:"Targeted effect by distance and width",
    val: (r, W) => { const j = String(rows.indexOf(r)); const v = seeds.map(s => S.seeds[s].depth["full|remove|hub"][W][j][0]).filter(x => x != null && !Number.isNaN(x)); return v.length ? mean(v) : NaN; }});
})();

// step profile
(() => {
  const host = P("fig-steps");
  [2,4,8].forEach(W => {
    const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
    const pts = m => [0,1,2,3,4].map(k => { const v = rng3(seeds.map(s => S.seeds[s].steps["full|remove|hub"][W][k][m][0])); return [k, v[0], v[1], v[2]]; });
    lineChart(d, {x:[0,1,2,3,4], y:[0, 0.35], yt:[0,.1,.2,.3], xname:"k", xtitle:"edited thought k", w:340, h:230, aria:`Step profile at W=${W}`,
      series:[{name:"Targeted", color:css("--s1"), band:true, pts: pts("targeted")}, {name:"Collateral", color:css("--s2"), band:true, pts: pts("collateral_attached")}]});
  });
})();

// signed collateral
(() => {
  const W2 = WS.slice(1);
  const mk = c => W2.map(W => { const v = eW(c, W, "collateral_signed"); return [W, -v[0], -v[2], -v[1]]; });
  // plotted as (patched - base): positive = others less reachable
  lineChart(P("fig-signed"), {w:900, h:280, x: W2, y:[-0.1, 0.08], yt:[-0.09,-0.06,-0.03,0,0.03,0.06], zero:true, xname:"W", xtitle:"width W", aria:"Signed collateral vs width",
    tfmt: v => (v>=0?"+":"") + v.toFixed(3),
    series:[{name:"Remove one branch", color:css("--s1"), band:true, pts: mk("full|remove|hub")},
            {name:"Insert one branch", color:css("--s2"), band:true, pts: mk("full|insert|hub")}]});
})();

// attention budget
(() => {
  const Wa = [1,2,4,8];
  const runs = seeds.map(s => X[RUNS[s]]).filter(Boolean);
  const v = W => rng3(runs.map(r => mean(["LAT1","LAT2","LAT3"].map(L => r.attention.layer1[L][W].excess))));
  const pts = Wa.map(W => { const a = v(W); return [W, a[0], a[1], a[2]]; });
  const ref = Wa.map(W => [W, pts[0][1]/W, 0, 0]);
  lineChart(P("fig-attn"), {w:900, h:260, x: Wa, y:[0, 0.3], yt:[0,.1,.2,.3], xname:"W", xtitle:"width W", aria:"Attention per branch vs width",
    series:[{name:"1/W reference", color:css("--neutral"), pts: ref, label:"∝ 1/W"}, {name:"Excess attention per branch", color:css("--s1"), band:true, pts}]});
})();

// downstream trace
(() => {
  const host = P("fig-trace");
  [2,4,8].forEach(W => {
    const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
    const ks = [1,2,3,4,5,6];
    const pts = (kind, fld) => ks.map(k => { const v = seeds.map(s => { const t = S.seeds[s].downstream[`W${W}|${kind}`]; return t && t[k] ? t[k][fld][0] : NaN; }).filter(x => x != null && !Number.isNaN(x));
      return v.length ? [k, mean(v), Math.min(...v), Math.max(...v)] : [k, NaN, NaN, NaN]; });
    lineChart(d, {x: ks, y:[-0.1, 0.8], yt:[0,.2,.4,.6,.8], zero:true, xname:"step", xtitle:"later thought k′", w:340, h:230, tfmt: v => v.toFixed(2), aria:`Probe margin trace at W=${W}`,
      series:[{name:"Unpatched", color:css("--neutral"), pts: pts("full","recipient_margin")},
              {name:"Full edit", color:css("--s1"), pts: pts("full","patched_margin")},
              {name:"Pattern edit", color:css("--s2"), pts: pts("pattern","patched_margin")}]});
  });
})();

// upfront table
(() => {
  const rows = WS.map(W => [W, ...seeds.map(s => { const u = S.seeds[s].upfront.full[W], p = S.seeds[s].upfront.pattern[W];
    return `${f3(u.targeted[0])} · ${u.collateral ? u.collateral[0].toFixed(3) : "–"} · ${f3(p.targeted[0])}`; })]);
  table(P("tbl-upfront"), ["W", ...seeds.map(s => `seed ${s.slice(1)}: target-branch ΔP · other-branch |ΔP| · pattern ΔP`)], rows, [0,1,2,3]);
})();

// claims table
(() => {
  const yes = t => `<span class="chip yes">${t}</span>`, no = t => `<span class="chip no">${t}</span>`, mid = t => `<span class="chip">${t}</span>`;
  const rows = [
    ["Extra latent positions are needed for queries 3+ hops away", "Matched no-latent control (10–12% correct on reachable 3+ hop nodes vs 38–97% for the recurrent seeds)", yes("supported"), "Single-seed control"],
    ["Continuous hidden-state feedback, specifically, improves reachability", "Matched pause-token control: equal overall (91%), better at ≤ 4 hops, worse at 5–6 hops (27–29% vs 38–70%) and on the upfront format", mid("only for the deepest queries"), "Single-seed pause control; overall accuracy does not separate them"],
    ["Several branches in one saved state can be edited separately, with the edited branch moving most", "Effect matrices from one saved prefix; targeted > collateral at W ≥ 3 in all seeds; controls ≈ 0", yes("supported, partial"), "Pre-set selectivity bar met by one seed at W = 2 only"],
    ["Probe read-out directions are not the causally used directions", "Filter vs pattern subspaces with identical ids and rank; energy fractions", yes("supported"), "Linear probes only; one probe family"],
    ["Separate usability declines faster with width than decodability", "Indexed curves W = 2…8, all seeds", yes("supported"), "Index base (W = 2) chosen because W = 1 has no competitor"],
    ["Branches compete for a normalised budget", "Signed collateral sign flips with edit direction (pre-registered); attention share ∝ 1/W", mid("supported in 2/3 seeds fully"), "Attention analysis exploratory; no causal test of the budget itself"],
    ["Edits affect only nodes the remaining latent steps must compute", "Depth-resolved effects; step profile", yes("supported"), "Deep-node baselines are uncertain (0.4–0.8 accuracy)"],
    ["Information edited at one step is partly recomputed later", "Probe-margin traces after the edit", mid("2 of 3 seeds"), "Recovery not causally blocked (cache edits not run)"],
    ["The mechanism emerges without supervision of intermediate steps", "Lookup scaffold + depth curriculum were required to train", no("not claimed"), "Scaffold may shape the solution"],
  ];
  table(P("tbl-claims"), ["claim", "evidence", "status", "main remaining limitation"], rows, []);
})();

// effect matrices
(() => {
  const host = P("fig-matrix");
  [2,4,8].forEach(W => {
    const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
    const M = seeds.map(s => S.seeds[s].matrix["full|remove|hub"][W]);
    const rows = Array.from({length: W}, (_, i) => `edit ${i+1}`), cols = Array.from({length: W}, (_, j) => j+1);
    heatmap(d, {rows, cols, max: 0.35, w: 300, ml: 52, ch: Math.max(18, Math.min(40, 240 / W)), colLabel: c => `q${c}`,
      aria: `Effect matrix at W=${W}`,
      val: (r, c) => mean(M.map(m => m[rows.indexOf(r)][c-1])),
      tipf: (r, c, v) => `${r} → query branch ${c}<br>${rows.indexOf(r) === c-1 ? "targeted" : "collateral |ΔP|"} <b>${v.toFixed(3)}</b>`});
  });
  const dg = [2,4,8].map(W => mean(seeds.map(s => { const m = S.seeds[s].matrix["full|remove|hub"][W]; return mean(m.map((r,i) => r[i])); })));
  const od = [2,4,8].map(W => mean(seeds.map(s => { const m = S.seeds[s].matrix["full|remove|hub"][W]; const v = []; m.forEach((r,i) => r.forEach((x,j) => { if (i!==j) v.push(x); })); return mean(v); })));
  P("cap-matrix").textContent = `Branch slots are in arbitrary order, so only the diagonal/off-diagonal contrast is meaningful. Mean diagonal ${dg.map(v => v.toFixed(3)).join(" / ")} vs mean off-diagonal ${od.map(v => v.toFixed(3)).join(" / ")} at W = 2 / 4 / 8.${window.PRINT ? "" : " Hover a cell for its value."}`;
})();
