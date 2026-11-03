def edit(p, E):
    s = open(p, encoding="utf-8").read()
    for o, n in E:
        assert s.count(o) == 1, ("§261 edit", p, s.count(o), o[:90]); s = s.replace(o, n)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
