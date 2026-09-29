import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "The effect is not universal. Seeds 0 and 1 gain less (about 60–65% out to distance 12), and V1, the one model tested only after this was pre-registered as P17, gains too little to count"
b = "${V.P18 && V.P18.pass !== undefined ? `A training-level test agrees. A model trained with no step embedding at all (P18, pre-registered) has no step index to learn a halt from, and it extrapolates: ${pct(V.P18.search.d7_K12)} at distance 7 after twelve steps, still above 60% at distance 12 after sixteen, gaining about one hop per extra step. ` : \"\"}The effect of holding the clock is not universal. Seeds 0 and 1 gain less (about 60–65% out to distance 12), and V1, the one model tested only after this was pre-registered as P17, gains too little to count"
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
