import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = 'chip: chip(H.competent && H.expected_all_hold)}'
b = 'chip: H.competent ? chip(H.expected_all_hold) : `<span class="chip">not scored</span>`}'
assert s.count(a) == 1; s = s.replace(a, b)
a = '3-hop Gq ${pc(P12.Gq_d3_W4)}'
b = '3-hop Gq ${pc(Math.abs(P12.Gq_d3_W4) < 0.005 ? 0 : P12.Gq_d3_W4)}'
assert s.count(a) == 1; s = s.replace(a, b)
a = 'and at three hops it is ${pc(P12.Gq_d3_W4)}.'
b = 'and at three hops it is ${pc(Math.abs(P12.Gq_d3_W4) < 0.005 ? 0 : P12.Gq_d3_W4)}.'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
