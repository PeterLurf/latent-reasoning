"""§268 filler family s273 = the s271 family (template v6) with s271 -> s273 (the shared gen10 deploy gen_deploy_120_s267.sh kept). Each output bash -n checked.
    python -B scratch/s273_build/mk_family.py"""
import subprocess

for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s271_%s.sh" % f, encoding="utf-8").read(); assert "s273" not in s
    s = s.replace("s271", "s273"); assert "gen_deploy_120_s267.sh" in s or f != "fill"
    p = "scratch/s273_%s.sh" % f; open(p, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.call(["bash", "-n", p]) == 0, p
print("s273 family written")
