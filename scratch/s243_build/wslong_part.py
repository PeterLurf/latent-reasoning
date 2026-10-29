

WSL_DS, WSL_K = tuple(range(1, 11)), 10                                    # §243 DC_v1 §3 fix 5: TEST2 d 1..10, passes 1..10


@torch.no_grad()
def part_ws_long(m, g, sets, a, dev, bs):
    """§243 WS-LONG (DC_v1 §3 fix 5, descriptive + STEP-READ): WS-READOUT (AW.ws_readout: the trunk's argmax over the 11 number tokens at ':' of pass t, never fed
    back) on TEST2 of the run's law, d 1..10 (all 1,000 each), passes 1..10 (independent of think_k); per item {sig, pred_by_pass {t}, v [v_min(t, d) for t 1..10]}.
    ops / src_j are joined by the scorer from the TEST2 regen by sig."""
    T = tokd(sets, a, list(WSL_DS)); r_, pv = AW.ws_readout(m, T, dev, WSL_K, bs, per_item=True)
    return {"by_d": r_, "items": {str(d): {"sig": [it["sig"] for it in T[d]], "pred_by_pass": pv[str(d)],
                                           "v": [[int(it["raw"]["values"][min(t, d)]) for t in range(1, WSL_K + 1)] for it in T[d]]} for d in WSL_DS},
            "def": "WS-LONG (§243 DC_v1 fix 5): WS-READOUT per pass 1..10 on TEST2 d 1..10 (all items), fp32; v = v_min(t, d)"}
