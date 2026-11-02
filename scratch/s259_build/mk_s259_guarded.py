"""Note 257-2: scratch/s259_guarded.py = scratch/s257_guarded.py + --long <K'list>:<ds>:<name> (each edit asserted once).
    python -B scratch/s259_build/mk_s259_guarded.py"""
s = open("scratch/s257_guarded.py", encoding="utf-8").read()
E = [('"""§257 DISTRACT-CHAIN host wrapper (DC2_v1 §5;',
      '"""Note 257-2 eval-only host wrapper = scratch/s257_guarded.py + --long <K\'list>:<ds>:<name> (after the parts: boxes.inject_aw_eval --kprime_list / --long_ds /\n'
      '--long_name = the §239 run_long, same model / dir). The §257 doc follows. §257 DISTRACT-CHAIN host wrapper (DC2_v1 §5;'),
     ('"--require", "--ship", "--gates", "--pilot_k", "--")):', '"--require", "--ship", "--gates", "--pilot_k", "--long", "--")):'),
     ('        json.dump({"eval_only": True, "model": mp, "model_sha256": s}, open(out + "/log.jsonl", "w"))',
      '        if multi(pre, "--long"):                                            # Note 257-2: run_long on the same model / dir\n'
      '            kp, lds, nm = multi(pre, "--long")[0].split(":")\n'
      '            rc = rc or subprocess.call([sys.executable, "-u", "-B", "-m", "boxes.inject_aw_eval", "--run", out, "--kprime_list", kp, "--long_ds", lds, "--long_name", nm,\n'
      '                                        "--eval_bs", "250", "--device", dev])\n'
      '        json.dump({"eval_only": True, "model": mp, "model_sha256": s}, open(out + "/log.jsonl", "w"))'),
     ]
for o, n in E:
    assert s.count(o) == 1, ("s259_guarded edit", s.count(o), o[:70]); s = s.replace(o, n)
open("scratch/s259_guarded.py", "w", encoding="utf-8", newline="\n").write(s); print("scratch/s259_guarded.py written")
