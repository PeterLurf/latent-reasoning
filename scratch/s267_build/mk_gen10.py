"""Drain gen10 (Audit 11-02: b396 / b401 / b409 drain-time overruns; the b409 step-1200 false RUN): NEW copies only (the running gen9 / s224 files are untouched), each edit
asserted once, every output bash -n checked:
  scratch/s267_done_marker.sh       = s224_done_marker.sh; the step-1200 log rule (G-FH-EQ) only for *_eqA / *_eqB dirs (a --stop_at train dir with a ckpt is BADM, not RUN).
  scratch/drain_gen_partial_s267.sh = drain_gen_partial_s224.sh with
      (1) NOFETCH: the queue's '# NOFETCH <path>' lines (scratch/gen_q_b<n>.txt* on the host) are deleted ON THE HOST before any fetch, and dropped from the need list;
      (2) small files first: every RUN dir is fetched once WITHOUT *.pt (+ its gen log), then again in full (ckpt_last / ckpt_prev / NOFETCH excluded);
      (3) the s267 done marker.
  scratch/auto_drain_gen10_s267.sh  = auto_drain_gen9_s224.sh + prefix "s267" -> drain_gen_partial_s267.sh.
  scratch/gen_deploy_120_s267.sh    = gen_deploy_120_s253.sh with the syntax gate on the gen10 files and the armed watcher auto_drain_gen10_s267.sh "... s267".
  scratch/s267_{fill,post_admit,chain,watch}.sh = the s265 family (template v4) with s265 -> s267 and gen_deploy_120_s253.sh -> gen_deploy_120_s267.sh.
    python -B scratch/s267_build/mk_gen10.py"""
import subprocess


def edit(src, dst, E, sub=()):
    s = open(src, encoding="utf-8").read()
    for a, b in sub:
        s = s.replace(a, b)
    for o, n in E:
        assert s.count(o) == 1, (dst, s.count(o), o[:90]); s = s.replace(o, n)
    open(dst, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", dst]).returncode == 0, dst


edit("scratch/s224_done_marker.sh", "scratch/s267_done_marker.sh", [
    ("  elif [ -s \"$1/log.jsonl\" ] && grep -q '\"step\": 1200,' \"$1/log.jsonl\"; then echo RUN\n",
     "  elif case \"$1\" in *_eqA|*_eqB) true;; *) false;; esac && [ -s \"$1/log.jsonl\" ] && grep -q '\"step\": 1200,' \"$1/log.jsonl\"; then echo RUN   # gen10: G-FH-EQ dirs only\n")],
     sub=(("s224_mark", "s267_mark"), ("s224_need", "s267_need"), ("s224_check_fetched", "s267_check_fetched"), ("s224_resume_ckpt", "s267_resume_ckpt")))
edit("scratch/drain_gen_partial_s224.sh", "scratch/drain_gen_partial_s267.sh", [
    ("echo QD=\\$(tail -n 1 \\$L | grep -c QUEUE_DONE);",
     "grep -h '^# NOFETCH ' scratch/gen_q_b${TAG#gen_b}.txt* 2>/dev/null | awk '{print \\\"NOFETCH\\\", \\$3}' | sort -u; echo QD=\\$(tail -n 1 \\$L | grep -c QUEUE_DONE);"),
    ("runs=$(echo \"$st\" | sed -n 's/^RUN //p'); bad=$(echo \"$st\" | sed -n 's/^BADM\\{0,1\\} //p')\n",
     "runs=$(echo \"$st\" | sed -n 's/^RUN //p'); bad=$(echo \"$st\" | sed -n 's/^BADM\\{0,1\\} //p')\n"
     "nof=$(echo \"$st\" | sed -n 's/^NOFETCH //p'); [ -n \"$nof\" ] && $S \"cd /workspace/Latent && rm -f $(echo $nof)\" 2>/dev/null   # gen10 (1): NOFETCH deleted on the host\n"
     "nfx=\"\"; for f_ in $nof; do nfx=\"$nfx --exclude=$f_\"; done\n"),
    ("mkdir -p /d/drain_tmp\nfor j in $runs; do n=$(echo $j | tr '/' '_')\n",
     "mkdir -p /d/drain_tmp\nfor j in $runs; do n=$(echo $j | tr '/' '_')_small   # gen10 (2): small files first (no *.pt), so the scorer inputs land before the GB transfers\n"
     "  fetch $n --exclude='*.pt' $j logs/gen_$(basename $j).log || { echo \"FETCH_INCOMPLETE $ID ($j small)\"; exit 1; }\ndone\n"
     "for j in $runs; do n=$(echo $j | tr '/' '_')\n"),
    ("$xm $j logs/gen_$(basename $j).log || {", "$xm $nfx $j logs/gen_$(basename $j).log || {"),
    ("for j in $runs; do r_=$(s267_check_fetched $j); [ \"$r_\" = OK ] || { echo \"FETCH_INCOMPLETE $ID $j $r_\"; exit 1; }; done\n",
     "for j in $runs; do r_=$(s267_check_fetched $j)\n"
     "  if [ \"$r_\" != OK ]; then m_=\"\"; for f_ in ${r_#MISSING}; do echo \"$nof\" | grep -qx \"$j/$f_\" || m_=\"$m_ $f_\"; done; [ -z \"$m_\" ] && r_=OK || r_=\"MISSING$m_\"; fi   # gen10 (1)\n"
     "  [ \"$r_\" = OK ] || { echo \"FETCH_INCOMPLETE $ID $j $r_\"; exit 1; }; done\n")],
     sub=(("s224_done_marker.sh", "s267_done_marker.sh"), ("s224_mark", "s267_mark"), ("s224_check_fetched", "s267_check_fetched"), ("s224_resume_ckpt", "s267_resume_ckpt"),
          ("s224_need", "s267_need")))
edit("scratch/auto_drain_gen9_s224.sh", "scratch/auto_drain_gen10_s267.sh", [
    ("      elif [ \"$pre\" = s224 ]; then res=$(bash scratch/drain_gen_partial_s224.sh \"$h\" \"$p\" \"$t\" \"$id\" 2>&1)   # §224\n",
     "      elif [ \"$pre\" = s224 ]; then res=$(bash scratch/drain_gen_partial_s224.sh \"$h\" \"$p\" \"$t\" \"$id\" 2>&1)   # §224\n"
     "      elif [ \"$pre\" = s267 ]; then res=$(bash scratch/drain_gen_partial_s267.sh \"$h\" \"$p\" \"$t\" \"$id\" 2>&1)   # gen10 (NOFETCH, small files first)\n")])
edit("scratch/gen_deploy_120_s253.sh", "scratch/gen_deploy_120_s267.sh", [
    ("scratch/auto_drain_gen9_s224.sh scratch/drain_gen_partial_s224.sh scratch/s224_done_marker.sh; do",
     "scratch/auto_drain_gen10_s267.sh scratch/drain_gen_partial_s267.sh scratch/s267_done_marker.sh; do"),
    ('nohup bash scratch/auto_drain_gen9_s224.sh "$hh $pp gen_b$n $id s224"', 'nohup bash scratch/auto_drain_gen10_s267.sh "$hh $pp gen_b$n $id s267"')])
for f in ("fill", "post_admit", "chain", "watch"):
    s = open("scratch/s265_%s.sh" % f, encoding="utf-8").read().replace("s265", "s267")
    if f == "fill":
        assert s.count("gen_deploy_120_s253.sh") == 2; s = s.replace("gen_deploy_120_s253.sh", "gen_deploy_120_s267.sh")   # the dirty-tree check + the call
    open("scratch/s267_%s.sh" % f, "w", encoding="utf-8", newline="\n").write(s); assert subprocess.run(["bash", "-n", "scratch/s267_%s.sh" % f]).returncode == 0, f
print("gen10 written; bash -n OK")
