import pathlib, re
p = pathlib.Path(r"D:\Latent\docs\PROTOCOL3.md")
s = p.read_text(encoding="utf-8")
rep = [
    ("(written 05:50, before any Part III model other than V3 was evaluated)", "(written before any Part III model other than V3 was evaluated)"),
    ("(written 05:40, V1 still training, never evaluated)", "(written while V1 was still training, never evaluated)"),
    ("* Note (added 2026-09-24 05:10): the first", "* Note (added 2026-09-24, after the crash): the first"),
    ("* Note (05:25): the first restart", "* Note (added later the same night): the first restart"),
    ("## Addendum D (written 2026-09-24 05:10, before runs/dev_s4 started training)", "## Addendum D (written 2026-09-24, before runs/dev_s4 started training)"),
    ("## Addendum E (written 2026-09-24 05:30, after", "## Addendum E (written 2026-09-24, after"),
    ("## Addendum F (written 2026-09-24 06:05, before", "## Addendum F (written 2026-09-24, before"),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
s = s.rstrip("\n") + "\n\n(Clock times were removed from these notes on 2026-09-24 ~06:00 because several had been estimated\nrather than read from the clock; the order of the notes and of the runs is what the file vouches for,\nand file modification times plus the research log give the actual sequence.)\n"
p.write_text(s, encoding="utf-8")
print("ok")
