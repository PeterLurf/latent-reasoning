import pathlib

p = pathlib.Path(r"D:\Latent\docs\RESEARCH_LOG.md")
r = p.read_text(encoding="utf-8-sig")
old = "What replicates: only recurrent seeds are clearly above chance at 5"
i = r.index(old)
j = r.index("\n", i)
new = ("Final depth-matched result (all 9 models): 6 hops — recurrent 0.59–0.66 vs all six controls\n"
       "  0.50–0.56; 5 hops — no-latent seed 2 reaches 0.67 (= weakest recurrent seed); 4 hops — no-latent\n"
       "  s1/s2 0.86/0.88 (> recurrent 0.75–0.83). Recurrence buys cross-seed reliability and the deepest queries.")
r = r[:i] + new + r[j:]
p.write_text(r, encoding="utf-8")
print("patched")
