

@torch.no_grad()
def part_stab(m, g, sets, a, dev, bs):
    """Note 260-3 (score_gbf, the descriptive r = 3 halt): per item of REG TEST2 d 1..7 the decoder's pred at K' 0..14 (hh_dump_inc, the run's own inputs - the generic cue
    for a generic-cue run), ans, sig; per d the TEST2 set sha. Layout as eval_stab_fresh.json. No halting decisions."""
    out = {"set_shas": {str(d): OW.set_sha(sets["test2"][d]) for d in range(1, 8)}}; T = tokd(sets, a, range(1, 8))
    for d in range(1, 8):
        _, P, _ = hh_dump_inc(m, T[d], dev, bs, SF_KMAX)
        out[str(d)] = {"n": len(T[d]), "items": {"pred": P.tolist(), "ans": [it["ans"] for it in T[d]], "sig": [it["raw"]["sig"] for it in T[d]]}}
    out["def"] = "Note 260-3 STAB: REG TEST2 d 1..7, per item the decoder's pred at K' 0..14 (the run's own pass inputs, S from zero), ans, sig; fp32"
    return out
