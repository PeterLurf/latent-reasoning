"""§224 Stage-1 G-FH-EQ check (Theory decision 3, 10-21): on a RD / FH cell's host, eqB (the cell's FT recipe, no far branch) and eqA (+ --far
--far_w 0) ran 1,200 steps at micro 1. Their log records at steps 100..1200 must be bitwise equal on every key but sec and far_* / fh_* (eqA's own
fields). On s1 hosts eqB is also compared with s215_ftctrl_s1's log records 100..1200 (the comparator FT-CTRL s1 = that run): a mismatch there with
eqA == eqB is host-caused (FH_v1 §6: the control stands with a note); eqA != eqB is code-caused (HOLD).
    python scratch/s224_eq_check.py <eqA log.jsonl> <eqB log.jsonl> [<s215 log.jsonl>]  -> one line: EQ OK | EQ HOST-NOTE <why> | EQ CODE-MISMATCH <why> |
    EQ UNREADABLE <why>; exit 0 (OK / HOST-NOTE) / 1 (CODE-MISMATCH) / 2"""
import json
import sys


def recs(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if "loss" in r and 100 <= r["step"] <= 1200:
                out[r["step"]] = {k: v for k, v in r.items() if k != "sec" and not k.startswith(("far_", "fh_"))}
    return out


def main():
    try:
        A, B = recs(sys.argv[1]), recs(sys.argv[2]); C = recs(sys.argv[3]) if len(sys.argv) > 3 else None
    except Exception as e:
        print("EQ UNREADABLE", e); sys.exit(2)
    steps = list(range(100, 1201, 100))
    if sorted(A) != steps or sorted(B) != steps:
        print("EQ UNREADABLE records", sorted(A), sorted(B)); sys.exit(2)
    d = [s for s in steps if A[s] != B[s]]
    if d:
        print("EQ CODE-MISMATCH eqA != eqB first at step %d: %s" % (d[0], json.dumps({k: [A[d[0]].get(k), B[d[0]].get(k)] for k in set(A[d[0]]) | set(B[d[0]]) if A[d[0]].get(k) != B[d[0]].get(k)})[:400]))
        sys.exit(1)
    if C is not None:
        dc = [s for s in steps if s not in C or C[s] != B[s]]
        if dc:
            print("EQ HOST-NOTE eqA == eqB (12 records); eqB != s215_ftctrl_s1 first at step %d: %s" % (dc[0], json.dumps({k: [C.get(dc[0], {}).get(k), B[dc[0]].get(k)] for k in B[dc[0]] if C.get(dc[0], {}).get(k) != B[dc[0]].get(k)})[:400]))
            sys.exit(0)
    print("EQ OK eqA == eqB (12 records)" + (" == s215_ftctrl_s1" if C is not None else "")); sys.exit(0)


if __name__ == "__main__":
    main()
