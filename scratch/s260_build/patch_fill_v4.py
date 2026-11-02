"""s260 / s261 template v4 (Audit 11-02): REFUSE a cell whose queue has a '--model' / '--init' '<path>=<sha>' value not also shipped as a plain token
(scratch/check_queue_ships.py), right after the WALLW-count refusal. Applied only while no filler of that family runs; each edit asserted once.
    python -B scratch/s260_build/patch_fill_v4.py"""
import subprocess

for fam in ("s260", "s261"):
    p = "scratch/%s_fill.sh" % fam; s = open(p, encoding="utf-8").read(); assert "template v4" not in s, (p, "already applied")
    o = '  [ "$nlq" = "$nw" ] || { log "REFUSED b$n: WALLW has $nw weights, the queue $nlq lines"; exit 2; }\n'
    n = (o + '  sc_=$(python scratch/check_queue_ships.py "scratch/gen_q_b$n.txt") || { log "REFUSED b$n: $sc_ (template v4, Audit 11-02)"; exit 2; }   # template v4\n')
    assert s.count(o) == 1, (p, s.count(o)); s = s.replace(o, n)
    open(p, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", p]).returncode == 0, p
print("template v4 applied to s260 / s261, bash -n OK")
