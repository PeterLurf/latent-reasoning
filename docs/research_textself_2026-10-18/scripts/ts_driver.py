"""runs the TEXT-SELF CPU label checks sequentially (started at Idle priority; children inherit it)."""
import subprocess, sys, os, time
H = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
jobs = [["--ns", "11,12", "--per_n", "40", "--protocols", "full", "--gen_check", "40", "--tag", "_gen"],
        ["--ns", "9,10,11,12,13,14,15,16", "--per_n", "200", "--protocols", "full,reroot", "--bs", "25", "--tag", "_s1"],
        ["--ns", "17,18,19,20", "--per_n", "100", "--protocols", "reroot", "--bs", "25", "--tag", "_s1_deep"]]
for j in jobs:
    t = time.time(); r = subprocess.run([py, "-B", os.path.join(H, "ts_label_check.py")] + j, cwd=H, stdout=open(os.path.join(H, "ts%s.out" % j[-1]), "w"), stderr=subprocess.STDOUT)
    open(os.path.join(H, "ts_driver.log"), "a").write("%s rc %d %.0f s\n" % (j[-1], r.returncode, time.time() - t))
open(os.path.join(H, "ts_driver.done"), "w").write("done\n")
