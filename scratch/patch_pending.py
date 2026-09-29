import pathlib
p = pathlib.Path(r"D:\Latent\latent\protocol3_confirm.py")
s = p.read_text(encoding="utf-8")
old15 = '''        V["P15"] = {"per_model": rows, "n_scored": len(rows), "pending": [t for t in NEW_RECURRENT if t not in rows],'''
new15 = '''        V["P15"] = {"per_model": rows, "n_scored": len(rows), "pending": [t for t in NEW_RECURRENT if t not in rows and M[t]["competence"] is None],
                    "failed_gate": [t for t in NEW_RECURRENT if M[t]["competence"] is not None and not gate(t)],'''
old16 = '''        V["P16"] = {"per_model": rows16, "pending": [t for t in NEW_RECURRENT if t not in rows16],'''
new16 = '''        V["P16"] = {"per_model": rows16, "pending": [t for t in NEW_RECURRENT if t not in rows16 and M[t]["competence"] is None],
                    "failed_gate": [t for t in NEW_RECURRENT if M[t]["competence"] is not None and not gate(t)],'''
for a, b in ((old15, new15), (old16, new16)):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
a = '''`${Object.values(P15.per_model).filter(r => r.agree).length} of ${P15.n_scored} agree${P15.pending.length ? `; pending: ${P15.pending.map(t => LBL[t]).join(", ")}` : ""}`'''
b = '''`${Object.values(P15.per_model).filter(r => r.agree).length} of ${P15.n_scored} agree${P15.pending.length ? `; pending: ${P15.pending.map(t => LBL[t]).join(", ")}` : ""}${(P15.failed_gate || []).length ? `; not scored (failed competence gate): ${P15.failed_gate.map(t => LBL[t]).join(", ")}` : ""}`'''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''(P16.pending.length ? `; pending: ${P16.pending.map(t => LBL[t]).join(", ")}` : ""),'''
b = '''(P16.pending.length ? `; pending: ${P16.pending.map(t => LBL[t]).join(", ")}` : "") + ((P16.failed_gate || []).length ? `; not scored (failed competence gate): ${P16.failed_gate.map(t => LBL[t]).join(", ")}` : ""),'''
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
