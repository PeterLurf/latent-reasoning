"""auto_ship in one process (2026-10-12: gen_deploy_120's bash auto_ship spawned grep / sed / sort per file over 3 fixed-point passes and took
10+ min under load, so offers went stale before the create - b273's four CREATE FAILEDs). Same rules as the bash function it replaces:
  - scratch/hostcheck.py always;
  - each queue line with >= 2 fields: its module (field 2, dots -> slashes, + .py) if the file exists, plus every boxes.X / boxes/X named in it;
  - every token starting boxes/ bench/ scratch/ that is an existing file (file arguments);
  - boxes/nested170.py when the queue has '# HOSTPROBE';
  - bench/data/mult_<t>_test.txt and bench/data/mult_train/<t>_train.txt for each '--task AxB';
  - 3 passes: every shipped boxes/ or scratch/ .py adds the boxes modules it names.
Prints the sorted unique list space-separated (as the bash version's final xargs).
    python scratch/auto_ship.py <queue file>"""
import os
import re
import sys

BOX = re.compile(rb"boxes[./][a-z0-9_]+")


def named_boxes(f):
    try:
        t = open(f, "rb").read()
    except OSError:
        return []
    return sorted({"boxes/" + m.decode()[6:] + ".py" for m in BOX.findall(t)})


def auto_ship(q):
    a = ["scratch/hostcheck.py"]
    lines = [l for l in open(q, encoding="utf-8").read().split("\n") if not l.startswith("#")]
    mods = sorted({l.split()[1] for l in lines if len(l.split()) >= 2})
    for m in mods:
        f = m.replace(".", "/") + ".py"
        if os.path.isfile(f):
            a.append(f)
        if f.startswith(("scratch/", "boxes/")):
            a += [b for b in named_boxes(f) if os.path.isfile(b)]
    a += [t for t in sorted({t for l in lines for t in l.split(" ")}) if re.match(r"^(boxes|bench|scratch)/", t) and os.path.isfile(t)]
    if any(l.startswith("# HOSTPROBE") for l in open(q, encoding="utf-8").read().split("\n")):
        a.append("boxes/nested170.py")
    for t in sorted({m for l in lines for m in re.findall(r"--task ([0-9]x[0-9])", l)}):
        a += [f for f in ("bench/data/mult_%s_test.txt" % t, "bench/data/mult_train/%s_train.txt" % t) if os.path.isfile(f)]
    for _ in range(3):
        cur = [f for f in a if re.match(r"^(boxes|scratch)/.*[.]py$", f)]
        for f in cur:
            a += [b for b in named_boxes(f) if os.path.isfile(b)]
    return sorted(set(a))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(" ".join(auto_ship(sys.argv[1])))
