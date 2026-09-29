import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "So the propagation step these models learned works well past the depth they were trained on, and what stops them is a halt keyed to the step index, learned because every training episode ended by step 6."
b = "So the propagation step these models learned works well past the depth they were trained on, and what stops them is a halt keyed to the step index, learned because every training episode ended by step 6. The attention trace below shows that the halt acts on the sweep itself. With normal step embeddings V2's first layer stops reading edges at step 6. With the embedding held at step 2 it walks the long chain one distance per step, through distances 7–12, and stops only at the chain's end."
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
