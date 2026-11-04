"""§271 N1 S1 (build serial s274): host wrappers = s271_guarded / s268_stopland with the trainer / eval modules swapped to boxes.inject_so / boxes.inject_so_eval
(each replacement asserted; nothing else changed); filler family s274 = s271 template v6 renamed. bash -n / ast checked.
    python -B scratch/s274_build/mk_wrappers.py"""
import ast
import subprocess

s = open("scratch/s271_guarded.py", encoding="utf-8").read()
for o, n, c in (('"-m", "boxes.inject_aw"]', '"-m", "boxes.inject_so"]', 1), ('"-m", "boxes.inject_aw_eval", "--run", out, "--parts"', '"-m", "boxes.inject_so_eval", "--run", out, "--parts"', 1)):
    assert s.count(o) == c, (o, s.count(o)); s = s.replace(o, n)
s = '"""s274 copy of scratch/s271_guarded.py: trainer boxes.inject_so, eval boxes.inject_so_eval (§271 N1); otherwise unchanged. Original doc follows."""\n' + s
ast.parse(s); open("scratch/s274_guarded.py", "w", encoding="utf-8", newline="\n").write(s)
s = open("scratch/s268_stopland.py", encoding="utf-8").read(); o = '"-m", "boxes.inject_aw_eval", "--run", out, "--parts"'; assert s.count(o) == 1
s = '"""s274 copy of scratch/s268_stopland.py: eval boxes.inject_so_eval (§271 N1 LOOP-SI stop at 4000); otherwise unchanged. Original doc follows."""\n' + s.replace(o, '"-m", "boxes.inject_so_eval", "--run", out, "--parts"')
ast.parse(s); open("scratch/s274_stopland.py", "w", encoding="utf-8", newline="\n").write(s)
for f in ("fill", "post_admit", "chain", "watch"):
    t = open("scratch/s271_%s.sh" % f, encoding="utf-8").read(); assert "s274" not in t; t = t.replace("s271", "s274")
    p = "scratch/s274_%s.sh" % f; open(p, "w", encoding="utf-8", newline="\n").write(t); assert subprocess.call(["bash", "-n", p]) == 0, p
print("s274 wrappers + family written")
