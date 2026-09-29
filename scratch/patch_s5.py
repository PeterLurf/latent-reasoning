import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = '''...["s3", "s4"].map(s => (FC.verdicts[p].heldout || {})[s] === undefined ? pend : chip(FC.verdicts[p].heldout[s]))]);
  table(P("tbl-seed3"), ["prediction", "Part II status", "Part II (all models)", LBL.s3, LBL.s4], sRows, []);'''
b = '''...["s3", "s4", "s5"].map(s => (FC.verdicts[p].heldout || {})[s] === undefined ? pend : chip(FC.verdicts[p].heldout[s]))]);
  table(P("tbl-seed3"), ["prediction", "Part II status", "Part II (all models)", LBL.s3, LBL.s4 + " (failed gate)", LBL.s5], sRows, []);'''
assert s.count(a) == 1, s.count(a); s = s.replace(a, b)
a = '''Competition inside the thought appears here as well, in three of four seeds now.</p>` : ""}`;'''
b = '''Competition inside the thought appears here as well, in three of four seeds now.</p>` : ""}
  ${V.heldout_s5 ? `<p><b>Seed 4</b> failed training (${pct(g("s4", m => m.competence.delayed))} delayed accuracy) and is not scored. A third held-out seed was added in its place. <b>Seed 5</b> clears the gate (${pct(g("s5", m => m.competence.delayed))}) but reaches only ${pct(g("s5", m => m.matched["6"].bal_acc))} at six hops, and it does not use its thought. At every candidate site a branch edit has a targeted effect below 0.0001, so the site rule picked one at random. Branch effects travel through query-time graph reading (${pc(g("s5", m => m.route_shares["4"].Gq))} at W = 4), and its first layer shows no sweep. The thought-edit predictions therefore fail for want of anything to edit: ${V.heldout_s5.expected_scored.filter(p => V.heldout_s5.verdicts[p]).length} of ${V.heldout_s5.expected_scored.length} expected predictions hold. The verdict stands as scored. The lesson is about the gate: overall accuracy admits a model that never uses the latent loop, as the no-latent controls already showed (91–93% overall). A depth-matched gate would have excluded it.</p>` : ""}`;'''
assert s.count(a) == 1, s.count(a); s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
