import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "After the crash, several bf16 runs stalled at the first curriculum stage, with delayed loss on a plateau near 0.565 and upfront loss at chance."
b = "About half of all gated training attempts, in both precisions, stalled at the first curriculum stage, with delayed loss on a plateau near 0.565 and upfront loss at chance."
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
