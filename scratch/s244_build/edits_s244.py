"""§244 ANSWER-ONLY (AO_v1 §3, §4 STILL-RISING): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E243 (each asserted once).
--keep_final_ckpt (default off = bitwise the old path): the final step is also checkpointed, and at the end of the run (after the evals) ckpt_last.pt is RENAMED to ckpt_final.pt instead of deleted, so the
drain (which excludes ckpt_last.pt / ckpt_prev.pt) fetches the 12k state for a pre-registered +6k resume. Retention only; no training / eval change."""
E244 = [
(r'''    ap.add_argument("--hold_inc", type=float, default=0.0,''',
 r'''    ap.add_argument("--keep_final_ckpt", action="store_true", help="§244: keep the final ckpt_last.pt as ckpt_final.pt (fetched by the drain) instead of deleting it")
    ap.add_argument("--hold_inc", type=float, default=0.0,'''),
(r'''        if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.stop_at) and step < a.steps:''',
 r'''        if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.stop_at) and (step < a.steps or a.keep_final_ckpt):   # §244: + the final step under --keep_final_ckpt'''),
(r'''    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):''',
 r'''    if a.keep_final_ckpt and os.path.exists(os.path.join(a.out, "ckpt_last.pt")):   # §244: the final state kept for a resume (renamed: the drain skips ckpt_last.pt)
        os.replace(os.path.join(a.out, "ckpt_last.pt"), os.path.join(a.out, "ckpt_final.pt"))
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):'''),
]
