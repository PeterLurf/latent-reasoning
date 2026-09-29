import pathlib

p = pathlib.Path(r"D:\Latent\latent\train.py")
s = p.read_text(encoding="utf-8")
old = """        for i in range(0, n, 512):
            sub = g.subset(np.arange(i, min(n, i + 512)))"""
new = """        for i in range(0, n, 128):          # small eval chunks: lower peak memory next to the captured graph
            sub = g.subset(np.arange(i, min(n, i + 128)))"""
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched")
