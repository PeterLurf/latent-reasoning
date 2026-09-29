import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "Seed 0's propagation machinery therefore works past its trained depth. What stops it is a state its thought enters around step 4. That state is not a simple, graph-independent direction:"
b = "Seed 0's propagation machinery therefore works past its trained depth. A first guess was that a state its thought enters around step 4 stops it. That state is not a simple, graph-independent direction, and the next section shows that the stop is keyed to the step embedding instead:"
assert s.count(a) == 1, "rewind"
s = s.replace(a, b)
a = 's5: "seed 5 (held out)", k4b: "budget K = 4, seed 1"};'
b = 's5: "seed 5 (held out)", k4b: "budget K = 4, seed 1", k4c: "budget K = 4, seed 2", k8: "budget K = 8"};'
assert s.count(a) == 1, "lbl"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
p = pathlib.Path(r"D:\Latent\latent\protocol3_confirm.py")
s = p.read_text(encoding="utf-8")
a = '"k4b": "runs/budget_k4_s1"}   # addenda G, H, I'
b = '"k4b": "runs/budget_k4_s1", "k4c": "runs/budget_k4_s2", "k8": "runs/budget_k8_s0"}   # addenda G-J'
assert s.count(a) == 1, "extra"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
