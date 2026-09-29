import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = "Those that finish the trained depth inside the budget stop there, two of them in a thought state that a rewind can partly undo."
b = "Those that finish the trained depth inside the budget stop there, and the stop is keyed to the step embedding: holding it at an early step lets two of them follow chains well past the trained depth (up to 88% at seven hops)."
assert s.count(a) == 1, "abstract"
s = s.replace(a, b)
a = '''    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: the route shift is a property of the delayed diagnostic.</b>'''
b = '''    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: the thought carries the frontier; the step embedding carries the stop.</b> Holding the step embedding at step 2 leaves the depth each step reads unchanged but removes the halt, and models that never extrapolate follow chains past six hops (seed 3 and V2 at 87–88% at distance 7). The propagation step generalises; the halt was learned from the step index. The prospective test of this on V1 (P17) failed, so it is not universal.</li>`);
    fl.insertAdjacentHTML("beforeend", `<li><b>Part III: the route shift is a property of the delayed diagnostic.</b>'''
assert s.count(a) == 1, "findings"
s = s.replace(a, b)
a = '''    add(["Width limits search fidelity, not speed",'''
b = '''    add(["The stop is keyed to the step embedding, not the search state", "Clock freeze/shift on s0, s1, s3, V2 (exploratory); P17 prospective on V1 failed", `<span class="chip">exploratory</span>`, "Strong in 2 of 5 models tested, weak in V1"]);
    add(["Width limits search fidelity, not speed",'''
assert s.count(a) == 1, "claims"
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
