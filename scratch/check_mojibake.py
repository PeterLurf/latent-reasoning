"""Find (and optionally repair) cp1252-mojibake in UTF-8 text files.

Each maximal run of non-ASCII characters is re-encoded as cp1252 and decoded as UTF-8 (repeatedly, for
multiply-encoded text); the replacement is kept only if every round succeeds. Correct characters such as
an arrow are not cp1252-encodable, and a lone en dash is not valid UTF-8, so they are left alone.
"""
import pathlib
import re
import sys

fix = "--fix" in sys.argv
files = [a for a in sys.argv[1:] if a != "--fix"]
RUN = re.compile(r"[^\x00-\x7f]+")


def repair(run):
    cur = run
    for _ in range(4):
        try:
            nxt = cur.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        cur = nxt
    return cur


for f in files:
    p = pathlib.Path(f)
    s = p.read_text(encoding="utf-8")
    t = RUN.sub(lambda m: repair(m.group()), s)
    changed = [i for i, (a, b) in enumerate(zip(s.splitlines(), t.splitlines()), 1) if a != b]
    print(f, "lines needing repair:", len(changed), changed[:40])
    if fix and changed:
        p.write_text(t, encoding="utf-8")
        print("  written")
