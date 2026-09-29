import pathlib
p = pathlib.Path(r"D:\Latent\latent\followup_confirm.py")
s = p.read_text(encoding="utf-8")
a = '''       "s4": ("runs/dev_s4", "")}
HELDOUT = ("s3", "s4")'''
b = '''       "s4": ("runs/dev_s4", ""), "s5": ("runs/main_s5", "")}
HELDOUT = ("s3", "s4", "s5")'''
assert s.count(a) == 1; s = s.replace(a, b); p.write_text(s, encoding="utf-8")

p = pathlib.Path(r"D:\Latent\latent\protocol3_confirm.py")
s = p.read_text(encoding="utf-8")
a = 'HELD = {"s3": "runs/main_s3", "s4": "runs/dev_s4"}'
b = 'HELD = {"s3": "runs/main_s3", "s4": "runs/dev_s4", "s5": "runs/main_s5"}'
assert s.count(a) == 1; s = s.replace(a, b)
a = 'EXTRA = {"v1b": "runs/v1_wide_s1", "k4": "runs/budget_k4_s0", "k10": "runs/budget_k10_s0"}   # addenda G, H'
b = 'EXTRA = {"v1b": "runs/v1_wide_s1", "k4": "runs/budget_k4_s0", "k10": "runs/budget_k10_s0", "k4b": "runs/budget_k4_s1"}   # addenda G, H, I'
assert s.count(a) == 1; s = s.replace(a, b)
a = '    for t in ("k4", "k10"):'
b = '    for t in ("k4", "k10", "k4b"):'
assert s.count(a) == 1; s = s.replace(a, b)
a = '        if t == "k4" and gate(t) and m["search"] and o["Gq_W4"] is not None:'
b = '        if t in ("k4", "k4b") and gate(t) and m["search"] and o["Gq_W4"] is not None:'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = 'v1b: "V1 wide, seed 1", k4: "budget K = 4", k10: "budget K = 10"};'
b = 'v1b: "V1 wide, seed 1", k4: "budget K = 4", k10: "budget K = 10", s5: "seed 5 (held out)", k4b: "budget K = 4, seed 1"};'
assert s.count(a) == 1; s = s.replace(a, b)
a = '  ["s3", "s4"].forEach(s => {\n    const H = V[`heldout_${s}`];\n    vRow(s === "s3" ? "C" : "D",'
b = '  ["s3", "s4", "s5"].forEach(s => {\n    const H = V[`heldout_${s}`];\n    vRow({s3: "C", s4: "D", s5: "I"}[s],'
assert s.count(a) == 1, s.count(a); s = s.replace(a, b)
a = '  table(P("tbl-p3score"), ["", "prediction", "model", "result", ""], rows, []);'
b = '''  const G4b = V.G_k4b;
  vRow("I", "Second K = 4 seed: if competent, it extrapolates or reads the graph at query time", LBL.k4b,
    G4b ? {txt: G4b.competent ? `${G4b.search.extrapolates ? "extrapolates" : "no extrapolation"} (distance 7 at 12 steps ${f2(G4b.search.d7_K12)}), query share W4 ${pc(G4b.Gq_W4)}` : "failed competence gate",
           chip: G4b.competent ? chip(!G4b.neither) : `<span class="chip">not scored</span>`} : {txt: "training", chip: pend});
  table(P("tbl-p3score"), ["", "prediction", "model", "result", ""], rows, []);'''
assert s.count(a) == 1; s = s.replace(a, b)
a = 'const order = ["s0", "s1", "s2", "s3", "s4", "v1", "v1b", "v2", "v3", "k4", "k10"].filter(t => M[t] && (M[t].competence || !["v1b", "k4", "k10"].includes(t)));'
b = 'const order = ["s0", "s1", "s2", "s3", "s4", "s5", "v1", "v1b", "v2", "v3", "k4", "k4b", "k10"].filter(t => M[t] && (M[t].competence || !["v1b", "k4", "k10", "s5", "k4b"].includes(t)));'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
