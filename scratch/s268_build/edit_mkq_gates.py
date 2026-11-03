"""Audit G-D b412 (1): the host gates line (scratch.s268_gates) before TRAIN + TRAIN --require <gates>/gates_pass.json, in scratch/s268_mkq.py (each edit asserted once).
    python -B scratch/s268_build/edit_mkq_gates.py"""
p = "scratch/s268_mkq.py"; s = open(p, encoding="utf-8").read()
E = [('''  pilot_tc_s1_mem / _gtime  K7 mem_check + G-TIME (the pilot flags; a crash refuses the host);
''', '''  pilot_tc_s1_mem / _gtime  K7 mem_check + G-TIME (the pilot flags; a crash refuses the host);
  pilot_tc_s1_gates         scratch.s268_gates (Audit G-D b412 (1)): gate_s268 offbit_aw / offbit_gen (flags-off bitwise vs the pre-§261 tree scratch/s268_ref.tar), smoke_plain,
                            smoke_tgt, gradflow on this host -> gates_pass.json (the TRAIN line --require's it);
'''),
     ('''                            --cue target (§261), --steps 8000 --stop_at 2000 (ckpt at 2000, exit 9), micro 2 pinned;''',
      '''                            --cue target (§261), --steps 8000 --stop_at 2000 (ckpt at 2000, exit 9), micro 2 pinned; requires pilot_tc_s1_gates/gates_pass.json;'''),
     ('''"scratch/s268_target.py", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]''',
      '''"scratch/s268_target.py", "scratch/s268_gates.py", "scratch/gate_s268.py", "scratch/gate_s257.py", "scratch/gate_s251.py", "scratch/gate_s228.py",
        "scratch/s222_det_wrap.py", "scratch/s268_ref.tar", "scratch/s256_ho_shas.json", "scratch/s257_ho_shas.json"]
GSHIP = ("scratch/s268_ref.tar scratch/gate_s268.py scratch/gate_s257.py scratch/gate_s251.py scratch/gate_s228.py scratch/s228_args_check.py scratch/s222_det_wrap.py "
         "scratch/s257_guarded.py boxes/runs_fsq/bun_s1_init/model.pt scratch/s256_ho_shas.json scratch/s257_ho_shas.json")'''),
     ('''"step-0 and 2k evals (acc + ws + insert_ho) under the target cue + the target-only gate.''',
      '''"host gates first (gates_pass.json required); step-0 and 2k evals (acc + ws + insert_ho) under the target cue + the target-only gate.'''),
     ('''     "%s_gtime boxes.inject_aw %s --time_steps 60\\n"
''', '''     "%s_gtime boxes.inject_aw %s --time_steps 60\\n"
     "%s_gates scratch.s268_gates --blob %s --gates offbit_aw,offbit_gen,smoke_plain,smoke_tgt,gradflow --ship %s\\n"
'''),
     ('''"%s scratch.s257_guarded --mode train --blob %s --init %s=%s --ship''', '''"%s scratch.s257_guarded --mode train --blob %s --init %s=%s --require %s_gates/gates_pass.json --ship'''),
     ('''% (W, W, fs, W, fs, W, blob, INIT[0], INIT[1], run, PARTS, INIT[0], run, SH, W, blob, INIT[0], INIT[1], INIT[0], SH,''',
      '''% (W, W, fs, W, fs, W, blob, GSHIP, W, blob, INIT[0], INIT[1], run, PARTS, INIT[0], run, SH, W, blob, INIT[0], INIT[1], W, INIT[0], SH,''')]
for o, n in E:
    assert s.count(o) == 1, o[:80]; s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s); print("s268_mkq: gates line added")
