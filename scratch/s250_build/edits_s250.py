"""§250 FRAME-DIV (FD_v1 §2): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E244 (each asserted once).
--train_frames (default '' = OW.FRAME_TRAIN, the training stream bitwise unchanged): the training stream's frames, e.g. '0-5,8-11' (Stage 1 DIV). Only the training stream."""
E250 = [
(r'''    ap.add_argument("--keep_final_ckpt", action="store_true",''',
 r'''    ap.add_argument("--train_frames", default="", help="§250: the training stream's frames, e.g. 0-5,8-11 ('' = OW.FRAME_TRAIN, unchanged)")
    ap.add_argument("--keep_final_ckpt", action="store_true",'''),
(r'''        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream''',
 r'''        TRAIN_FR = tuple(x for p_ in a.train_frames.split(",") for x in (range(int(p_.split("-")[0]), int(p_.split("-")[1]) + 1) if "-" in p_ else (int(p_),))) if a.train_frames else OW.FRAME_TRAIN
        assert all(f_ in range(len(OW.FRAMES["add_c"])) for f_ in TRAIN_FR) and len(set(TRAIN_FR)) == len(TRAIN_FR), ("§250 --train_frames", TRAIN_FR)
        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=TRAIN_FR, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream; §250: --train_frames'''),
]
