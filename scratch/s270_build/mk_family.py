"""§263 filler family s270 = the s267 family (template v4 + the PMAX <= admit / hours refusal) with s267 -> s270 (the shared gen10 deploy gen_deploy_120_s267.sh kept),
+ template v5 (Audit 11-03): fam() keys realised lines by attempt (scratch/fill_v5_build/mk_fam_v5.py) and GATES_H (env, default 0) - the host gates line's hours -
in the admit projection and in the PMAX refusal. Each output bash -n checked.
    python -B scratch/s270_build/mk_family.py"""
import subprocess

for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s267_%s.sh" % f, encoding="utf-8").read().replace("s267", "s270")
    if f == "fill":
        assert s.count("gen_deploy_120_s270.sh") == 2; s = s.replace("gen_deploy_120_s270.sh", "gen_deploy_120_s267.sh")
        for o, n in (("h = $STEPS * ${SPS_EST:-1.4} / 3600 + $SETUP_H + $EVAL_H;", "h = $STEPS * ${SPS_EST:-1.4} / 3600 + $SETUP_H + $EVAL_H + ${GATES_H:-0};   # v5: + GATES_H"),
                     ("proj=$(python -c \"print(round(($sps * $STEPS / 3600 + $SETUP_H + $EVAL_H * $NE) * $dph, 2))\")",
                      "proj=$(python -c \"print(round(($sps * $STEPS / 3600 + $SETUP_H + ${GATES_H:-0} + $EVAL_H * $NE) * $dph, 2))\")")):
            assert s.count(o) == 1, o[:60]; s = s.replace(o, n)
    assert "gen_deploy_120_s270" not in s
    open("scratch/s270_%s.sh" % f, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", "scratch/s270_%s.sh" % f]).returncode == 0, f
assert subprocess.call(["python", "-B", "scratch/fill_v5_build/mk_fam_v5.py", "scratch/s270_fill.sh"]) == 0
print("s270 family written (v5: fam by attempt + GATES_H); bash -n OK")
