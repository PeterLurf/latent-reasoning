import pathlib

p = pathlib.Path(r"D:\Latent\report\render2.js")
r = p.read_text(encoding="utf-8")
R = [
    ('<p>The same logic applies inside the thought itself (P14, pre-registered as addendum B).',
     '<p>The same logic applies inside the thought itself (P14, pre-registered as addendum B).'),
    ('competition is visible in the thought, not only in the attention that builds it.</p><div class="tbl" id="tbl-amp"></div>`);',
     'competition is visible in the thought, not only in the attention that builds it.</p>`);\n    P("prose-mech").insertAdjacentHTML("afterend", `<div class="tbl" id="tbl-amp"></div>`);'),
    ('lineChart(d, {x: ds, y:[-0.05, 1.1], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, zero:true, xname:"distance", xtitle:"hops from start", w:340, h:230,',
     'lineChart(d, {x: ds, y:[-0.15, 1.25], yt:[0,.25,.5,.75,1], yfmt: t => pc(t), tfmt: pc, zero:true, xname:"distance", xtitle:"hops from start", w:340, h:230,'),
    ('    const rows = [...seeds.map(n => ["recurrent " + n, n]), ...PAUSE_N.map(n => ["pause " + n, n]), ...K0_N.map(n => ["no latent " + n, n])].map(([lab, n]) =>',
     '    const lab_ = n => ({s0: "recurrent seed 0", s1: "recurrent seed 1", s2: "recurrent seed 2", pause: "pause seed 0", pause1: "pause seed 1", pause2: "pause seed 2", k0: "no-latent seed 0", k0_1: "no-latent seed 1", k0_2: "no-latent seed 2"})[n] || n;\n'
     '    const rows = [...seeds, ...PAUSE_N, ...K0_N].map(n => [lab_(n), n]).map(([lab, n]) =>'),
    ("They confirm that extra positions are needed beyond two hops, and they overturn the single pause seed's apparent parity up to four hops.",
     "They overturn both of Part I's single-seed control conclusions. Some control seeds never pass two hops, while one no-latent seed solves up to five hops through the prefix alone; only six-hop queries separate every recurrent seed from every control."),
]
for old, new in R:
    assert r.count(old) == 1, old[:60]
    r = r.replace(old, new)
p.write_text(r, encoding="utf-8")
t = pathlib.Path(r"D:\Latent\report\template.html")
s = t.read_text(encoding="utf-8")
old = "td.n,th.n{"
assert s.count(old) == 1
s = s.replace(old, "td:first-child{white-space:nowrap}\ntd.n,th.n{")
t.write_text(s, encoding="utf-8")
print("ok")
