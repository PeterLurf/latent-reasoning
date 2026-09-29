import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
old = 'xlab: v => v >= 1000 ? (v / 1000) + "k" : v'
assert s.count(old) == 2, s.count(old)
s = s.replace(old, 'xlab: v => v % 5000 === 0 ? (v / 1000) + "k" : ""')
old2 = '<p class="fig-title">When the sweep appears</p>'
assert s.count(old2) == 1
s = s.replace(old2, '<p class="fig-title">Seed 4: the sweep never advances past depth 1</p>')
p.write_text(s, encoding="utf-8")
print("ok")
