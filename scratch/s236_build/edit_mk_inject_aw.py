"""§236 REG-CHAIN (RC_v1 §1): adds the --mul_s_p edits to scratch/s234_build/mk_inject_aw.py (appended to its E list; each asserted once there) and the reg_excl helper to
the head. --mul_s_p 0: build_sets(mul_s_p=0) (REG sets, new seeds; their shas go to args.json), the training stream, calibration and memcheck streams under the REG law,
and the training EXCL extended by the eval module's own REG sets (reg_eval_sigs). Default 1 = unchanged (flags-off gate)."""
p = "scratch/s234_build/mk_inject_aw.py"; s = open(p, encoding="utf-8").read()
o = "]\nfor o, n in E:"
add = '''# §236 REG-CHAIN (RC_v1 §1): --mul_s_p 0 = the REG law on the training / calibration / memcheck streams and every eval set (new seeds); the training EXCL += the eval
# module's own REG sets (swap / probe); default 1 = unchanged
(\'    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")\',
 \'    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")\\n\'
 \'    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1], help="§236: 0 = REG law (no forced mul_s) on every stream / set (new eval seeds); 1 = unchanged")\'),
(\'    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"\',
 \'    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"\\n\'
 \'    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), "--mul_s_p 0: OW only, mul_s_p3 must be 1"\'),
(\'        SETS = build_sets(); EXCL = SETS["excl"]; stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, **OWKW)   # §235: training stream only\',
 \'        SETS = build_sets(mul_s_p=a.mul_s_p); EXCL = SETS["excl"] | (reg_excl(a, SETS["excl"]) if a.mul_s_p == 0 else set())   # §236: + the eval-own REG sets\\n\'
 \'        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream\'),
(\'    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, **OWKW) if a.src == "ow" else None\',
 \'    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW) if a.src == "ow" else None   # §236: the run\\\'s law\'),
(\'            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, **OWKW); its =\', \'            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its =\'),
]
for o, n in E:'''
assert s.count(o) == 1; s = s.replace(o, add); open(p, "w", encoding="utf-8", newline="\n").write(s); print("mk_inject_aw: +5 edits")
p = "scratch/s234_build/inject_aw_head.py"; s = open(p, encoding="utf-8").read()
o = '''@torch.no_grad()
def ws_readout('''
n = '''def reg_excl(a, excl):
    """§236: the union of the eval module's own REG sets' sigs (swap_k d 2, swap_kd d 3..d_max + d 3 t 2, probe) for the training EXCL (0 overlap by construction)."""
    from boxes import inject_aw_eval as AE
    return set().union(*AE.reg_eval_sigs(vars(a), excl).values())


@torch.no_grad()
def ws_readout('''
assert s.count(o) == 1; s = s.replace(o, n); open(p, "w", encoding="utf-8", newline="\n").write(s); print("head: + reg_excl")
