"""PROOFS_D4 check 3: brute-force the counting claims of Prop D0 / D0b / D1 on symbolic rollouts that mirror boxes/vbank.py rollout's bookkeeping.
Stored entries St[0..]: St[0] = V_0 (age 0). Pass t (t = 1..K) computes Vc[t] from a read over St[0..t-1] and stores
  - plain / reduction pass 'R', idle pass 'I', computed pause 'P': St[t] = Vc[t]   (edit None; input_edit adds pi for 'P') -> age(St[t]) = age(St[t-1]) + 1
  - copy-stall 'C': edit returns St[t-1] (stored AND carried)                         -> age(St[t]) = age(St[t-1])
At pass t's input: b = len(St) = t (bank index count), a = age(St[t-1]) (lineage age of the newest entry), rho = #R before t, r = n - rho.
Checks over every schedule (all orders) with n <= 8 and up to 6 extra C/P passes and idle tail:
  D0:   rho + r = n <= 8 on every pass; 'rho >= 8 => finished' never wrong.
  D0b:  for every pass with unfinished input: t <= 8 + (#marked passes before t)  [marked = 'P'; C passes are unmarked stalls]
        and #unmarked-computed passes before t (R + I, i.e. the MARKER-GATED count) == rho.
  D1:   unfinished input => (no C/P): b-1 == a == rho;  (C only): b-1 == rho + S_C, a == rho;  (P only): b-1 == a == rho + S_P;  (mixed): b-1 = rho+S_C+S_P, a = rho+S_P.
  Max t + r over 'R' passes == n + (#C + #P before the last R) + 1."""
import itertools, json, sys

def check(n, seq):
    St_age = [0]; rho = 0; ok = {"D0": True, "D0b": True, "D0b_gated_eq_rho": True, "D1": True}; mk = 0; sc = 0; sp = 0; unmarked = 0; trmax = 0
    for t, typ in enumerate(seq, 1):
        b = len(St_age); a = St_age[-1]; r = n - rho; F = rho >= n
        if rho + r != n or n > 8:
            ok["D0"] = False
        if rho >= 8 and not F:
            ok["D0"] = False
        if not F:
            if not t <= 8 + mk:
                ok["D0b"] = False
            if unmarked != rho:
                ok["D0b_gated_eq_rho"] = False
            if not (b - 1 == rho + sc + sp and a == rho + sp):
                ok["D1"] = False
        if typ == "R":
            trmax = max(trmax, t + r)
        # the pass
        if typ == "C":
            St_age.append(St_age[-1]); sc += 1 if not F else 0
        else:
            St_age.append(St_age[-1] + 1)
        if typ == "P":
            mk += 1; sp += 1 if not F else 0
        if typ in ("R", "I"):
            unmarked += 1
        if typ == "R":
            rho += 1
    return ok, trmax

def schedules(n, extra_max=4, idle_max=3):
    for e in range(0, extra_max + 1):
        for kinds in itertools.product("CP", repeat=e):
            L = n + e
            for pos in itertools.combinations(range(L), e):
                seq = []; it = iter(kinds)
                for j in range(L):
                    seq.append(next(it) if j in pos else "R")
                for idle in range(idle_max + 1):
                    yield seq + ["I"] * idle

tot = 0; bad = {"D0": 0, "D0b": 0, "D0b_gated_eq_rho": 0, "D1": 0, "trmax": 0}
for n in range(1, 9):
    for seq in schedules(n):
        ok, trm = check(n, seq); tot += 1
        for k, v in ok.items():
            bad[k] += (not v)
        last_R = max(i for i, x in enumerate(seq) if x == "R")
        pre = sum(1 for x in seq[:last_R] if x in "CP")
        bad["trmax"] += trm != n + pre + 1
out = {"schedules_checked": tot, "violations": bad}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1); print(out)
