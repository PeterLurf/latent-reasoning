"""§262 filler family s269 = the s267 family (template v4, drain gen10) with s267 -> s269, except the shared gen10 deploy scripts (gen_deploy_120_s267.sh keeps its name:
it arms auto_drain_gen10_s267.sh with the s267 drain prefix). Each output bash -n checked.
    python -B scratch/s269_build/mk_family.py"""
import subprocess

for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s267_%s.sh" % f, encoding="utf-8").read().replace("s267", "s269")
    if f == "fill":
        assert s.count("gen_deploy_120_s269.sh") == 2; s = s.replace("gen_deploy_120_s269.sh", "gen_deploy_120_s267.sh")   # the dirty-tree check + the call (gen10)
    assert "gen_deploy_120_s269" not in s
    open("scratch/s269_%s.sh" % f, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", "scratch/s269_%s.sh" % f]).returncode == 0, f
print("s269 family written; bash -n OK")
