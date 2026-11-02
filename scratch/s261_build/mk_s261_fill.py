"""Note 257-2 b395 filler family s261 = the s260 template v3 files (sed s260 -> s261: own ledger / guards / logs, so the $0.6 family cap is not mixed with §258's).
The watch tails the b395 drain log. Each edit asserted.
    python -B scratch/s261_build/mk_s261_fill.py"""
import subprocess

for f in ("fill", "post_admit", "chain", "watch2"):
    s = open("scratch/s260_%s.sh" % f, encoding="utf-8").read().replace("s260", "s261")
    if f == "watch2":
        for o, n in (("logs/auto_drain_gen_b396.out logs/auto_drain_gen_b397.out logs/auto_drain_gen_b398.out logs/auto_drain_gen_b399.out", "logs/auto_drain_gen_b395.out"),
                     ("if (a[1] >= 396)", "if (a[1] >= 395)"), ("the b396-b399 drain logs", "the b395 drain log")):
            assert s.count(o) == 1, (f, o); s = s.replace(o, n)
    g = "scratch/s261_%s.sh" % ("watch" if f == "watch2" else f)
    if f == "watch2":
        s = s.replace("s261_watch2", "s261_watch")
    open(g, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", g]).returncode == 0, g
print("s261 family written, bash -n OK")
