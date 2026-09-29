import pathlib

p = pathlib.Path(r"D:\Latent\report\render2.js")
r = p.read_text(encoding="utf-8")
old = 'P10: "Per-branch amplitude ∝ 1/√W fits best"};'
new = 'P10: "Per-branch amplitude ∝ 1/√W fits best", P14: "Erasing / doubling a branch\'s own component in the thought lowers / raises its deep nodes"};'
assert r.count(old) == 1
r = r.replace(old, new)
old2 = "  // ---------------- controllability\n"
new2 = '''  // ---------------- amplification (P14)
  (() => {
    const rows = recOK.filter(s => CF.recurrent[s].amplify).flatMap(s => ["4","8"].map(W => { const a = CF.recurrent[s].amplify[W];
      return [`seed ${s.slice(1)}`, W, f3(a.erase_own_deep[0]), f3(a.double_own_deep[0]), f3(a.double_others[0]), a.random_abs.toFixed(3)]; }));
    P("prose-mech").insertAdjacentHTML("beforeend", `<p>The same logic applies inside the thought itself (P14, pre-registered as addendum B). With no donor at all, we rescaled one reachable branch's own pattern component of the saved thought. Erasing it lowers that branch's nodes three or more hops away, and doubling it raises them, in every seed. The same operation on a random subspace of equal rank does almost nothing. Doubling one branch also lowers the other reachable branches in ${CF.P14_competition_W4_seeds} of 3 seeds at W = 4, so competition is visible in the thought, not only in the attention that builds it.</p><div class="tbl" id="tbl-amp"></div>`);
    table(P("tbl-amp"), ["seed", "W", "erase own: ΔP (≥ 3 hops)", "double own: ΔP (≥ 3 hops)", "double own: ΔP other branches", "random subspace |ΔP|"], rows, [1,2,3,4,5]);
  })();

  // ---------------- controllability
'''
assert r.count(old2) == 1
r = r.replace(old2, new2)
r = r.replace('const models = ["s0", "s0@k0", "s1", "s2", "p0", "p1", "p2"];', 'const models = ["s0", "s0@k0", "s1", "s2", "p0", "p1", "p2"];')
p.write_text(r, encoding="utf-8")
print("ok")
