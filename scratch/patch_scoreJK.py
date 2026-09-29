import pathlib
p = pathlib.Path(r"D:\Latent\latent\protocol3_confirm.py")
s = p.read_text(encoding="utf-8")
a = '"k4c": "runs/budget_k4_s2", "k8": "runs/budget_k8_s0"}   # addenda G-J'
b = '"k4c": "runs/budget_k4_s2", "k8": "runs/budget_k8_s0", "nostep": "runs/nostep_s0"}   # addenda G-K'
assert s.count(a) == 1; s = s.replace(a, b)
a = '    # held-out seeds (addenda C/D)'
b = '''    # addenda J/K: stricter gate (overall >= 0.85 and depth-matched 5-hop >= 0.60)
    gateJ = lambda t: gate(t) and M[t]["matched"] is not None and M[t]["matched"]["5"]["bal_acc"] >= 0.60

    def len_row(t, k):
        L = jload(f"results/length_gen_{t}.json") or {}
        r = L.get(ALL[t])
        return next((x for x in r["rows"] if x["K'"] == k), None) if r else None

    if M["k4c"]["competence"] is not None:
        m = M["k4c"]
        o = {"gate": gateJ("k4c"), "search": m["search"], "Gq_W4": (m["route_shares"] or {}).get("4", {}).get("Gq")}
        if o["gate"] and o["Gq_W4"] is not None:
            ref = max(M[s_]["route_shares"]["4"]["Gq"] for s_ in REF if M[s_]["route_shares"])
            o["pass"] = bool(m["search"].get("extrapolates") or o["Gq_W4"] > ref)
        V["J_k4c"] = o
    if M["k8"]["competence"] is not None:
        m = M["k8"]
        r16 = len_row("k8", 16)
        o = {"gate": gateJ("k8"), "E": m["frontier_E"], "d7_K16": r16["far"]["7"] if r16 else None}
        if o["gate"] and o["E"] is not None and o["d7_K16"] is not None:
            o["pass"] = bool(o["E"] >= 4 and o["d7_K16"] < 0.60)
        CFk = jload("results/clock_freeze_k8.json") or {}
        if ALL["k8"] in CFk and o["gate"] and not m["search"].get("extrapolates"):
            f = CFk[ALL["k8"]]["freeze_2_K12"]
            o["P17"] = {"d7": f["acc"]["7"], "ci": f["ci"]["7"], "pass": f["acc"]["7"] >= 0.60 and f["ci"]["7"][0] > 0.55}
        V["J_k8"] = o
    if M["nostep"]["competence"] is not None:
        m = M["nostep"]
        o = {"gate": gateJ("nostep"), "search": m["search"], "E": m["frontier_E"]}
        if o["gate"] and "extrapolates" in m["search"]:
            o["pass"] = bool(m["search"]["extrapolates"])
        V["P18"] = o
    # held-out seeds (addenda C/D)'''
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
