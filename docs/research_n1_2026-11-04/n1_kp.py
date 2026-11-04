"""N1 v1 (THEORY S271) DATA-SIM side checks: perfect-loop K'-curve beyond m 8, and the hold-parity share.

Pure simulation of the task law; no repo imports; deterministic (random.Random(7)). Run: python -I n1_kp.py
Prints: for m 10 / 12 at n 5, the MOVED share and P(S_8 = S_m | MOVED) (a perfect loop stopped at K' 8);
then P(S_3 != S_4) at n 5 m 4 over all items (M-5: re-applying the swap-4 cue at hold passes costs this share).
"""
import random

R = random.Random(7)


def draw(n, m):
    s = R.randrange(n)
    S = [s]
    for _ in range(m):
        a, b = R.sample(range(n), 2)
        x = S[-1]
        S.append(b if x == a else a if x == b else x)
    return S


if __name__ == "__main__":
    for m in (10, 12):
        T = c = 0
        for _ in range(200000):
            S = draw(5, m)
            k = sum(S[i] != S[i - 1] for i in range(1, m + 1))
            if S[-1] != S[0] and k >= 2:
                T += 1
                c += S[8] == S[m]
        print(f"n5 m{m}: MOVED share {T / 200000:.4f}  P(S_8 = S_{m} | MOVED) {c / T:.4f}")
    T = c = 0
    for _ in range(200000):
        S = draw(5, 4)
        T += 1
        c += S[3] != S[4]
    print(f"n5 m4 P(S3 != S4) {c / T:.4f}")
