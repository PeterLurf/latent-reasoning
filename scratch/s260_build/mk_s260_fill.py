"""§258 filler family s260 = the s258 template files (sed s258 -> s260; Audit's 11-01 guard fixes) + PMAX default 0.21 (Audit 11-01) + EVAL_H from env in the chain.
The watch tails the b396+ drain logs. Each edit asserted.
    python -B scratch/s260_build/mk_s260_fill.py   -> scratch/s260_fill.sh / s260_post_admit.sh / s260_chain.sh / s260_watch.sh"""
import subprocess

for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s258_%s.sh" % f, encoding="utf-8").read().replace("s258", "s260")
    if f == "fill":
        o = "PMAX=${PMAX:-0.25};"; assert s.count(o) == 1; s = s.replace(o, "PMAX=${PMAX:-0.21};")
    if f == "chain":
        assert "EVAL_H=${EVAL_H:-0.5}" in s
    if f == "watch":
        for o, n in (("logs/auto_drain_gen_b393.out logs/auto_drain_gen_b394.out", "logs/auto_drain_gen_b396.out logs/auto_drain_gen_b397.out logs/auto_drain_gen_b398.out"),
                     ("if (a[1] >= 393)", "if (a[1] >= 396)"), ("the b393-b394 drain logs", "the b396-b398 drain logs"), ("n >= 393", "n >= 396")):
            assert s.count(o) == 1, (f, o); s = s.replace(o, n)
    open("scratch/s260_%s.sh" % f, "w", encoding="utf-8", newline="\n").write(s)
    assert subprocess.run(["bash", "-n", "scratch/s260_%s.sh" % f]).returncode == 0, f
print("s260 family written, bash -n OK")
