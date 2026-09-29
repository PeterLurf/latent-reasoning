import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = '''  if (st_) st_.insertAdjacentHTML("beforeend", "<tr>" + p17row.map(c => `<td>${c}</td>`).join("") + "</tr>");'''
b = '''  if (st_) st_.insertAdjacentHTML("beforeend", "<tr>" + p17row.map(c => `<td>${c}</td>`).join("") + "</tr>");
  const jRow = (id, pred, model, v, txt) => st_ && st_.insertAdjacentHTML("beforeend", "<tr>" + [id, pred, model, v ? txt(v) : "training",
    !v ? pend : v.gate === false ? `<span class="chip">not scored</span>` : v.pass === undefined ? pend : chip(v.pass)].map(c => `<td>${c}</td>`).join("") + "</tr>");
  jRow("J1", "K = 4, seed 2 (stricter gate): extrapolates or reads the graph at query time", "budget K = 4, seed 2", V.J_k4c,
    v => v.gate === false ? "failed competence gate" : `distance 7 at 12 steps ${f2(v.search.d7_K12)}; query share W4 ${pc(v.Gq_W4)}`);
  jRow("J2", "K = 8: sweep reaches depth 4 within six steps and no extrapolation at 16 steps", "budget K = 8", V.J_k8,
    v => v.gate === false ? "failed competence gate" : `E = ${v.E}; distance 7 at 16 steps ${f2(v.d7_K16)}`);
  jRow("P18", "No step embedding: the model extrapolates (distance 7 ≥ 60% at 12 steps)", "no step embedding", V.P18,
    v => v.gate === false ? "failed competence gate" : `distance 7 at 12 steps ${f2(v.search.d7_K12)} [${v.search.d7_K12_ci ? v.search.d7_K12_ci.map(f2).join(", ") : "–"}]; E = ${v.E}`);'''
assert s.count(a) == 1; s = s.replace(a, b)
a = 'k4c: "budget K = 4, seed 2", k8: "budget K = 8"};'
b = 'k4c: "budget K = 4, seed 2", k8: "budget K = 8", nostep: "no step embedding"};'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
