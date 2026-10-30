"""§251 ANSWER-EXTEND (AE_v1 §2): main() edits for boxes/inject_aw.py, applied by scratch/s234_build/mk_inject_aw.py AFTER E250 (each asserted once).
--aux_d_max (default 0 = off, the aux path bitwise unchanged): the per-pass aux CE only on items with d <= aux_d_max. A batch with no qualifying item (one d per step, so
every d > aux_d_max step) gets aux := an in-graph zero (la * 0.0) instead of torch.stack([]) (which raises)."""
E251 = [
(r'''    ap.add_argument("--train_frames", default="",''',
 r'''    ap.add_argument("--aux_d_max", type=int, default=0, help="§251: the aux CE only on items with d <= aux_d_max (0 = every item, unchanged)")
    ap.add_argument("--train_frames", default="",'''),
(r'''                    idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]''',
 r'''                    idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t and (not a.aux_d_max or it["d"] <= a.aux_d_max)]   # §251 --aux_d_max'''),
(r'''            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); loss = la + lam_now[0] * aux''',
 r'''            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean() if any(per) else la * 0.0   # §251: no qualifying item -> an in-graph zero
            loss = la + lam_now[0] * aux'''),
]
