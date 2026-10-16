"""VB-2G Stage 0 (user-approved 2026-10-16 12:32 via Audit): THEORY note 211-13 CONTENT-HALT on a §211 cell, eval-only (docs/research_bottleneck_2026-10-16/
H_halting.md §2.1, §5, §6). Local fp32, --eval_bs 32 (211-7), the §170 test set, T_max 24, card / bs / dtype recorded. NEW module; vbank /
vbank_eval read-only. -> <run>/eval_vbhalt.json (Theory's field contract 2026-10-16: by_n, stall, rows).
Per item: one rollout to 24 passes; Ê_t = argmax R(stored V_t) on the real slots; the coda answer at every t 0..24. The halt (halt_of): for t = 1..24,
  G-PROG: alive(Ê_{t-1}) - alive(Ê_t) >= 2 flags the trajectory (a later DONE is an abstention, status "gprog", gprog_t = the DONE pass);
  STUCK-STOP r 3: Ê_t == Ê_{t-1} == Ê_{t-2} with alive > 0 -> "stuck" (never-halt);
  DONE(Ê_t) = no operator, no bracket, exactly one value slot -> t_halt = t ("halt"); none by 24 -> "never".
acc_halt = the coda answer at t_halt; never / stuck / gprog read at T_max and counted wrong (event-only accuracy beside it).
frozen = never or stuck, with alive(Ê at the stop) recorded (frozen_nm8 = alive == n - 8).
STALL (the registered §211 intervention): stored V_t := V_{t-1} at t = ceil(n/2), rolled to 24, on n 4..8 (9..12 descriptive); among items with intact
t_halt = n: shift = P(stalled t_halt = n + 1); frozen1 = never / stuck with exactly 1 operator alive; other_never; at_n; ge_n2. Double STALL
(copies at ceil(n/2) and ceil(n/2) + 1) -> P(t_halt = n + 2), descriptive; STUCK-STOP is skipped at ceil(n/2) + 1 there (two copies give three equal
decodes by construction). t_conv_dec = the first t >= 1 with Ê_t == Ê_{t-1}; answer2 = the first t
>= 2 whose coda answer equals the answers at t - 1 and t - 2 (moved2 = the answer changes again later): descriptive, Experiments' definitions.
Not in this pass (descriptive, later): HEAD-STALL / CLOCK-PROBE / CLOCK-GAIN heads, ADVANCE, R-POS.
    python -m boxes.vbank_halt --run boxes/runs_fsq/s211_vb_pa_s1 --device cuda:0 [--fixture]"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

TMAX = 24
OPS_C = {VB.CID[" " + o] for o in N.OPS}; BR_C = {VB.CID[" ("], VB.CID[" )"]}; VAL_C = set(range(VB.VAL0, VB.NCLS))


def alive(e):
    return sum(1 for c in e if c in OPS_C)


def done(e):
    return alive(e) == 0 and not any(c in BR_C for c in e) and sum(1 for c in e if c in VAL_C) == 1


def halt_of(E, skip_stuck=()):
    """E: list of decoded tuples Ê_0..Ê_T (real slots). -> (status, t, gprog_t, alive at stop). status in halt / gprog / stuck / never."""
    flag = False
    for t in range(1, len(E)):
        if alive(E[t - 1]) - alive(E[t]) >= 2:
            flag = True
        if t >= 2 and t not in skip_stuck and E[t] == E[t - 1] == E[t - 2] and alive(E[t]) > 0:
            return "stuck", None, None, alive(E[t])
        if done(E[t]):
            return ("gprog", None, t, 0) if flag else ("halt", t, None, 0)
    return "never", None, None, alive(E[-1])


def fixture():
    """H Prop 5 trajectories on synthetic Ê (decoded = targets): content -> t_halt = n; a 2-op collapse at t 8 -> gprog at 8; stuck at 1 alive ->
    stuck (FROZEN-1); a clock that holds after t 8 -> never / stuck with n - 8 alive; an extra copied pass -> t_halt = n + 1."""
    rng = np.random.default_rng(21300); out = {"content": 0, "collapse": 0, "stuck1": 0, "clockhold": 0, "stall_shift": 0, "n": 0}
    for _ in range(300):
        n = int(rng.integers(9, 13)); r, _ = N.generate(rng, n); E = [tuple(e[1:-1]) for e in VB.recode_targets(r, n)]; out["n"] += 1
        Ec = E + [E[-1]] * (TMAX + 1 - len(E)); out["content"] += halt_of(Ec)[:2] == ("halt", n)
        Eh = E[:8] + [E[n]] * (TMAX + 1 - 8); out["collapse"] += halt_of(Eh)[0] == "gprog" and halt_of(Eh)[2] == 8
        Es = E[:n] + [E[n - 1]] * (TMAX + 1 - n); st = halt_of(Es); out["stuck1"] += st[0] == "stuck" and st[3] == 1
        Ek = E[:9] + [E[8]] * (TMAX + 1 - 9); st = halt_of(Ek); out["clockhold"] += st[0] == "stuck" and st[3] == n - 8
        m_ = math.ceil(n / 2); Ed = E[:m_] + [E[m_ - 1]] + E[m_:]; Ed = Ed + [Ed[-1]] * (TMAX + 1 - len(Ed)); out["stall_shift"] += halt_of(Ed)[:2] == ("halt", n + 1)
    out["pass"] = all(out[k] == out["n"] for k in ("content", "collapse", "stuck1", "clockhold", "stall_shift")); return out


def decode_rows(m, S, pad):
    """stored versions S[0..T] -> per item a list of tuples Ê_t over the real slots."""
    dec = [m.rec_logits(v).argmax(-1).cpu().numpy() for v in S]; L = pad.sum(1).cpu().numpy()
    return [[tuple(int(c) for c in d[i, :L[i]]) for d in dec] for i in range(pad.shape[0])]


@torch.no_grad()
def rollout_halt(m, its, dev, stall=()):
    """rollout to TMAX; stall: passes whose stored entry is a copy of the previous stored entry. -> per item (E list, coda answers t 0..TMAX)."""
    x, pad, lens = VB.batch_x(its, dev); st = {}

    def edit(t, V):
        if t in stall:                                                      # the copy is itself stored (a double STALL copies it again; 12:4x fix)
            st[t] = st[t - 1].clone(); return st[t]
        st[t] = V
        return None
    if stall:
        st[0] = m.rollout(x, pad, 0)["V"][0]
    r = m.rollout(x, pad, TMAX, coda_rows={t: torch.arange(len(its), device=dev) for t in range(TMAX + 1)}, edit=edit if stall else None)
    num = torch.tensor(N.NUM_IDS, device=dev)
    A = np.stack([m.answer_logits(r["coda"][t][1], lens)[:, num].argmax(-1).cpu().numpy() for t in range(TMAX + 1)], 1)
    E = decode_rows(m, r["S"], pad); del r, st
    return E, A


def run(m, TE, dev, bs, partial_path=None):
    Nn = np.array([it["n"] for it in TE]); rows = []; out = {"by_n": {}, "stall": {}, "stall_by_n": {}, "double_stall": {}}
    for n in range(1, 17):
        recs = []
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; E, A = rollout_halt(m, its, dev)
            for k_, j in enumerate(ch):
                y = its[k_]["t"]; s_, th, gt, al = halt_of(E[k_]); tc = next((t for t in range(1, TMAX + 1) if E[k_][t] == E[k_][t - 1]), None)
                a2 = next((t for t in range(2, TMAX + 1) if A[k_, t] == A[k_, t - 1] == A[k_, t - 2]), None)
                ra = None
                if s_ == "halt":
                    vs = [c for c in E[k_][th] if c in VAL_C]; ra = int(vs[0] - VB.VAL0) if len(vs) == 1 else None
                rec = {"item": int(j), "n": n, "y": int(y), "status": s_, "t_halt": th, "gprog_t": gt, "alive_stop": al,
                       "alive": [alive(e) for e in E[k_]], "coda": [int(v) for v in A[k_]], "r_answer": ra, "t_conv_dec": tc, "answer2": a2,
                       "moved2": bool(a2 is not None and any(A[k_, t] != A[k_, a2] for t in range(a2, TMAX + 1))),
                       "collapse_t": [t for t in range(1, TMAX + 1) if alive(E[k_][t - 1]) - alive(E[k_][t]) >= 2]}
                recs.append(rec); rows.append(rec)
        it_ = len(recs); f = lambda c: sum(1 for r_ in recs if c(r_)) / it_  # noqa: E731
        acc_halt = f(lambda r_: r_["status"] == "halt" and r_["coda"][r_["t_halt"]] == r_["y"])
        hist = {}
        for r_ in recs:
            k = str(r_["t_halt"]) if r_["status"] == "halt" else r_["status"]; hist[k] = hist.get(k, 0) + 1
        halted = [r_ for r_ in recs if r_["status"] == "halt"]
        out["by_n"][str(n)] = {"items": it_, "exact": f(lambda r_: r_["status"] == "halt" and r_["t_halt"] == n),
                               "premature": f(lambda r_: r_["status"] == "halt" and r_["t_halt"] < n), "late": f(lambda r_: r_["status"] == "halt" and r_["t_halt"] > n),
                               "never": f(lambda r_: r_["status"] in ("never", "stuck")), "gprog": f(lambda r_: r_["status"] == "gprog"), "acc_halt": acc_halt,
                               "acc_n2": f(lambda r_: r_["coda"][n + 2] == r_["y"]),
                               "halt_at_8": f(lambda r_: (r_["status"] == "halt" and r_["t_halt"] == 8) or (r_["status"] == "gprog" and r_["gprog_t"] == 8)),
                               "frozen_nm8": f(lambda r_: r_["status"] in ("never", "stuck") and r_["alive_stop"] == n - 8), "t_hist": hist,
                               "acc_halt_event_only": (sum(1 for r_ in halted if r_["coda"][r_["t_halt"]] == r_["y"]) / len(halted)) if halted else None,
                               "mean_t_halt_event_only": float(np.mean([r_["t_halt"] for r_ in halted])) if halted else None,
                               "r_answer_acc": (sum(1 for r_ in halted if r_["r_answer"] == r_["y"]) / len(halted)) if halted else None,
                               "r_coda_agree": (sum(1 for r_ in halted if r_["r_answer"] == r_["coda"][r_["t_halt"]]) / len(halted)) if halted else None,
                               "collapse_rate": float(np.mean([len(r_["collapse_t"]) > 0 for r_ in recs])), "stuck": f(lambda r_: r_["status"] == "stuck")}
        print("HALT n", n, {k: out["by_n"][str(n)][k] for k in ("exact", "premature", "never", "gprog", "acc_halt", "acc_n2")}, flush=True)
    if partial_path:                                                        # a late (STALL-part) crash keeps the by_n readings
        json.dump({"by_n": out["by_n"], "rows": rows, "partial": True}, open(partial_path, "w", newline="\n"))
    intact = {(r_["item"]): r_ for r_ in rows}
    for n in range(4, 13):                                                  # STALL (4..8 registered; 9..12 descriptive) and double STALL (4..8)
        ts = math.ceil(n / 2); sb = {"q": 0, "shift": 0, "frozen1": 0, "other_never": 0, "at_n": 0, "ge_n2": 0, "gprog": 0, "other": 0}; dbl = {"q": 0, "n2": 0}
        for ch in VE.chunks(np.nonzero(Nn == n)[0], bs):
            its = [TE[j] for j in ch]; E, _ = rollout_halt(m, its, dev, stall=(ts,))
            E2, _ = rollout_halt(m, its, dev, stall=(ts, ts + 1)) if n <= 8 else (None, None)
            for k_, j in enumerate(ch):
                if not (intact[int(j)]["status"] == "halt" and intact[int(j)]["t_halt"] == n):
                    continue
                s_, th, gt, al = halt_of(E[k_]); sb["q"] += 1
                if s_ == "halt" and th == n + 1:
                    sb["shift"] += 1
                elif s_ in ("never", "stuck") and al == 1:
                    sb["frozen1"] += 1
                elif s_ in ("never", "stuck"):
                    sb["other_never"] += 1
                elif s_ == "halt" and th == n:
                    sb["at_n"] += 1
                elif s_ == "halt" and th >= n + 2:
                    sb["ge_n2"] += 1
                elif s_ == "gprog":
                    sb["gprog"] += 1
                else:
                    sb["other"] += 1
                if E2 is not None:
                    s2 = halt_of(E2[k_], skip_stuck=(ts + 1,)); dbl["q"] += 1   # two copies make 3 equal decodes by construction; dbl["n2"] += s2[0] == "halt" and s2[1] == n + 2
        out["stall_by_n"][str(n)] = {"qualifying": sb["q"], **{k: (v / sb["q"] if sb["q"] else None) for k, v in sb.items() if k != "q"}, "stall_t": ts}
        if n <= 8:
            out["double_stall"][str(n)] = {"qualifying": dbl["q"], "p_n2": (dbl["n2"] / dbl["q"]) if dbl["q"] else None}
        print("STALL n", n, out["stall_by_n"][str(n)], flush=True)
    pool = {k: sum(out["stall_by_n"][str(n)][k] * out["stall_by_n"][str(n)]["qualifying"] for n in range(4, 9) if out["stall_by_n"][str(n)]["qualifying"])
            for k in ("shift", "frozen1", "other_never", "at_n", "ge_n2", "gprog", "other")}
    q = sum(out["stall_by_n"][str(n)]["qualifying"] for n in range(4, 9))
    out["stall"] = {"qualifying": q, **{k: (v / q if q else None) for k, v in pool.items()}, "pooled": "n 4..8"}
    out["rows"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--fixture", action="store_true"); a = ap.parse_args()
    fx = fixture(); print("FIXTURE", fx, flush=True)
    if a.fixture or not fx["pass"]:
        return
    dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, args = VE.load(a.run, dev); TE = N.test_set(a.per_n); t0 = time.time(); res = run(m, TE, dev, a.eval_bs, os.path.join(a.run, "eval_vbhalt.partial.json"))
    ev = os.path.join(a.run, "eval_vb.json")                                # gate (i): item counts at t = n + 2 == eval_vb's acc_n2
    if os.path.exists(ev) and a.per_n == 1000:
        a2 = json.load(open(ev))["acc_n2"]; res["gate_acc_n2"] = {n: {"halt": res["by_n"][n]["acc_n2"], "eval_vb": a2[n]} for n in res["by_n"]}
        res["gate_acc_n2_pass"] = all(round(v["halt"] * 1000) == round(v["eval_vb"] * 1000) for v in res["gate_acc_n2"].values())
    res.update({"fixture": fx, "run": a.run, "mode": m.mode, "mct": bool(args.get("mct")), "eval_bs": a.eval_bs, "dtype": "fp32", "per_n": a.per_n, "T_max": TMAX,
                "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "sec": round(time.time() - t0)})
    fo = os.path.join(a.run, "eval_vbhalt.json"); json.dump(res, open(fo + ".tmp", "w", newline="\n")); os.replace(fo + ".tmp", fo)
    print("DONE", fo, res["sec"], "gate", res.get("gate_acc_n2_pass"), flush=True)


if __name__ == "__main__":
    main()
