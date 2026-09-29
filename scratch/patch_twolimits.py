import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "The effect of holding the clock is not universal. Seeds 0 and 1 gain less (about 60–65% out to distance 12), and V1, the one model tested only after this was pre-registered as P17, gains too little to count"
b = "The effect of holding the clock is not universal. Seeds 0 and 1 gain less (about 60–65% out to distance 12). The two models tested only after it was pre-registered as P17, V1 and the K = 8 run, gain too little to count. Their traces show why: their sweeps do not halt sharply at a step but fade hop by hop, reaching distance 7 too weakly to matter, and a held clock cannot fix that. So there are two kinds of limit, a step-keyed halt (seed 3, V2) and per-hop attenuation (K = 8, V1), and P17 had assumed the first everywhere"
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
