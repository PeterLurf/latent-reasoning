import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = 'H ? {txt: `W8/W2 = ${f2(H.targeted_W8_over_W2)} → ${H.reading || "failed competence gate"}`, chip: H.reading ? `<span class="chip">${H.reading}</span>` : chip(false)} : {txt: "training", chip: pend});'
b = 'H ? {txt: H.competent ? `W8/W2 = ${f2(H.targeted_W8_over_W2)} → ${H.reading}` : `failed competence gate (${pct(g("v1b", m => m.competence.delayed))} after four attempts): unresolved`, chip: H.reading ? `<span class="chip">${H.reading}</span>` : `<span class="chip">not scored</span>`} : {txt: "training", chip: pend});'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
