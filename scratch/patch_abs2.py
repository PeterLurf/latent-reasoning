import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "and so does a model trained with only four steps, too few for its sweep to finish."
b = "and so do models trained with only four steps, too few for the sweep to finish (both competent seeds), and a model trained with no step embedding at all; a model trained with eight steps does not."
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
