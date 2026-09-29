import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = 'const clockModels = ["s3", "v2", "s1", "s0", "v1"].filter(t => CFz[RUN(t)]);'
b = 'const clockModels = ["s3", "v2", "v1"].filter(t => CFz[RUN(t)]);'
assert s.count(a) == 1; s = s.replace(a, b)
a = 'P("cap-clock").textContent = "Line colours per model; dashed lines lie at chance beyond six hops. Seed 2, which never stops, is unaffected by the step embedding and is not shown.";'
b = 'P("cap-clock").textContent = `Seeds 3 and V2 show the effect most clearly; V1 is the prospective test (P17), which failed. With the clock held at 2, seeds 0 and 1 reach ${pct(fz("s0", "9"))} and ${pct(fz("s1", "9"))} at distance 9 (normal: chance). Seed 2, which never stops, is unaffected and not shown.`;'
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
