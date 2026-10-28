"""§238 DAG-FREE (DF_v1 §1): appends to scratch/s234_build/mk_inject_aw.py's E list (applied after every earlier edit; each asserted once there):
--mul_s_p takes 2 (DF), and the training EXCL += the eval-own sets for any non-default law (REG 0 and DF 2)."""
p = "scratch/s234_build/mk_inject_aw.py"; s = open(p, encoding="utf-8").read()
o = "]\nfor o, n in E:"
add = '''# §238 DAG-FREE: --mul_s_p 2 (DF); the eval-own EXCL for any non-default law
(\'choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")\',
 \'choices=[0, 1, 2], help="§236: 0 = REG law (no forced mul_s); §238: 2 = DF (per-step mul_s, q_t); 1 = unchanged - on every stream / set (new eval seeds)")\'),
(\'(reg_excl(a, SETS["excl"]) if a.mul_s_p == 0 else set())   # §236: + the eval-own REG sets\',
 \'(reg_excl(a, SETS["excl"]) if a.mul_s_p != 1 else set())   # §236 / §238: + the eval-own sets of the run law\'),
]
for o, n in E:'''
assert s.count(o) == 1; s = s.replace(o, add); open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw: +2 edits (§238)")
