import pathlib

p = pathlib.Path(r"D:\Latent\report\render2.js")
r = p.read_text(encoding="utf-8")
R = []

R.append(("Scaling the edit does not help, and nor does restoring attention gain. Instead the branch's causal effect <em>moves out of the latent state</em>: the thought and early-latent cache carry ${pc(latW2[0])} of it at two branches but ${pc(latW8[0])} at eight, while queries reading the graph tokens directly take over (${pc(qW2[0])} → ${pc(qW8[0])}). Within the state, branches share a normalised, relative code. Boosting one lowers the others, and the per-branch signal falls roughly as W<sup>−0.6…0.9</sup>.",
          "Scaling the edit does not help, and nor does restoring attention gain. Two things change instead. First, branches share a normalised, relative code: boosting one lowers the others, and the per-branch signal falls roughly as W<sup>−0.6…0.9</sup>. With it, the model's own sensitivity to a branch's deep nodes collapses as width grows. Second, nodes two or three hops away move from the latent pathway to queries reading the graph tokens directly, where a thought edit cannot reach. Averaged over the branch, the latent pathway's share falls from ${pc(latW2[0])} to ${pc(latW8[0])}. For the deepest nodes, though, a thought edit reproduces a similar share of the model's own counterfactual at every width."))

R.append(("  F2.push(`<b>The branch's effect moves out of the latent state as width grows.</b> Latent-pathway share ${pc(latW2[0])} → ${pc(latW8[0])} from W = 2 to 8; query-time graph reading ${pc(qW2[0])} → ${pc(qW8[0])}.",
          "  F2.push(`<b>Why the branch average weakens.</b> Per deep node the edit does not weaken: at 5–6 hops a thought edit reproduces ${pc(deepT(\"2\")[1])}–${pc(deepT(\"2\")[2])} of the model's own counterfactual at W = 2 and ${pc(deepT(\"8\")[1])}–${pc(deepT(\"8\")[2])} at W = 8. The average falls for two reasons. The model's own counterfactual on deep nodes collapses with width (at four hops ${gapD(\"2\",\"4\").toFixed(2)} → ${gapD(\"8\",\"4\").toFixed(2)}). And two- and three-hop nodes move to query-time graph reading, so the latent-pathway share of the whole branch falls ${pc(latW2[0])} → ${pc(latW8[0])} while query reading rises ${pc(qW2[0])} → ${pc(qW8[0])}."))

R.append(('  const pd5 = depthG(PAUSE_N, "5"), pd3 = depthG(PAUSE_N, "3");',
          '  const pd5 = depthG(PAUSE_N, "5"), pd3 = depthG(PAUSE_N, "3");\n'
          '  const RD = FU.route_depth || {};\n'
          '  const deepT = W => mm(recOK.filter(s => RD[s]).flatMap(s => ["5","6"].map(d => RD[s][W][d] && RD[s][W][d].clean_gap > 0.05 ? RD[s][W][d].T : NaN)));\n'
          '  const gapD = (W, d) => mm(recOK.filter(s => RD[s]).map(s => RD[s][W][d].clean_gap))[0];'))

R.append(("The shares move with width (P1). At two branches most of a branch's effect travels through the latent pathway. At eight, most of it is resolved by queries reading the graph tokens directly.",
          "The shares move with width (P1). At two branches most of a branch's effect travels through the latent pathway; at eight, most of it is resolved by queries reading the graph tokens directly. The breakdown by node distance below shows what that average hides. One-hop nodes are always resolved at query time, and nodes four or more hops away always through the latent pathway. What moves is the boundary, since two- and three-hop nodes switch from latent to query-time resolution at higher width. The amount of effect on deep nodes also shrinks, because the model itself becomes less decisive about them."))
R.append((" This route shift, more than interference within the state, is why a thought edit loses its grip as width grows: the queries increasingly read the graph instead.", ""))

R.append(("A shared, normalised code is enough to explain why each branch's signal weakens. It does not explain the route shift above, which is the larger effect.",
          "A shared, normalised code explains why each branch's signal weakens with width, and with it the model's own counterfactual sensitivity on deep nodes. Together with the inward move of the latent/query boundary, that accounts for the weaker branch-averaged edits."))

R.append(('      ["Edits weaken with width mainly because the effect moves out of the latent state", "Route decomposition: latent share ↓, query-graph share ↑ with W, all seeds", yes("supported"), "Graph-route patches change the effective input"],',
          '      ["Branch-averaged edits weaken because deep-node counterfactual sensitivity collapses and 2–3-hop nodes move to query-time resolution; per deep node, edit efficacy does not decline", "Route decomposition by node distance (post-hoc, fresh graphs), all seeds", yes("supported (post-hoc)"), "Depth analysis not pre-registered; graph-route patches change the effective input"],'))

FIG = '''  (() => {
    const host = P("fig-depthroute");
    const ds = [1,2,3,4,5,6];
    W3.forEach(W => {
      const d = document.createElement("div"); d.innerHTML = `<p class="panel-title">W = ${W}</p>`; host.appendChild(d);
      const ser = (c, col, name) => ({name, color: css(col), band: true, pts: ds.map(dd => { const v = mm(recOK.filter(s => RD[s]).map(s => { const x = RD[s][W][dd]; return x && x.clean_gap > 0.05 ? x[c] : NaN; })); return [dd, v[0], v[1], v[2]]; })});
      lineChart(d, {x: ds, y:[-0.05, 1.1], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, zero:true, xname:"distance", xtitle:"hops from start", w:340, h:230,
        series:[ser("T","--s1","Thought only"), ser("TL","--s3","Latent pathway"), ser("Gq","--s2","Query-time graph")], aria:`Route share by distance W=${W}`});
    });
    const cg = W => ({name: `W = ${W}`, color: css({"2":"--s1","4":"--s2","8":"--s3"}[W]), band: true, pts: ds.map(dd => { const v = mm(recOK.filter(s => RD[s]).map(s => RD[s][W][dd].clean_gap)); return [dd, v[0], v[1], v[2]]; })});
    lineChart(P("fig-cleangap"), {x: ds, y:[0, 1.05], yt:[0,.25,.5,.75,1], xname:"distance", xtitle:"hops from start: clean counterfactual effect of removing the branch (ΔP)", w:900, h:220,
      series: W3.map(cg), aria:"Clean counterfactual effect by distance"});
    P("cap-depthroute").textContent = `Shares are shown only where the clean effect exceeds 0.05. Shares can exceed 100% or fall below 0 when routes overlap. This breakdown was not pre-registered; it re-analyses the confirmation graphs (the aggregate numbers reproduce exactly).`;
  })();
'''
R.append(('  (() => {\n    const host = P("fig-margin");', FIG + '  (() => {\n    const host = P("fig-margin");'))

for old, new in R:
    assert r.count(old) == 1, ("missing or duplicate", old[:70])
    r = r.replace(old, new)
p.write_text(r, encoding="utf-8")
print("applied", len(R), "edits")
