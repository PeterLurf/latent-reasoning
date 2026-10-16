"""HH-FIT: pass-free model acc_t = sum_h P(hh=h|t) q(h) fit to s175.aux_acc_by_role.handle (n ~ U{1..8} eval population). CPU."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1])); ns = d['n']; F = d['feats']; H = 5; M = np.zeros((8, H))
for t in range(1, 9):
    hh = np.array([min(f[t-1]['hh'], H) for n, f in zip(ns, F) if 1 <= n <= 8 and n >= t]); M[t-1] = [(hh == h).mean() for h in range(1, H+1)]
for m in sys.argv[2:]:
    e = json.load(open('D:/Latent/boxes/runs_fsq/%s/eval.json' % m))['s175']['aux_acc_by_role']; y = np.array([e[str(t)]['handle'] for t in range(1, 9)])
    q = np.ones(H) * 0.9
    for _ in range(20000):
        q = np.clip(q - 0.5 * (M.T @ (M @ q - y)), 0, 1)
    ss = ((y - y.mean())**2).sum(); r2 = 1 - ((y - M @ q)**2).sum() / ss if ss > 1e-9 else float('nan')
    print('%-20s q=%s R2=%.3f maxres=%.3f' % (m, ' '.join('%.3f' % v for v in q), r2, np.abs(y - M @ q).max()))
