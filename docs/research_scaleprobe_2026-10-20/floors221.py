import json, re, math
import numpy as np
T = json.load(open('prosqa/ckpt_ext/hf/data/gsm8k_test.json')); N = len(T['answer'])
def f(x):
    try: return float(str(x).replace(',', '').rstrip('.'))
    except ValueError: return None
gold = [f(a) for a in T['answer']]
inter = [set(x for x in (f(m.split('=')[-1].rstrip('>')) for m in re.findall(r'<<[^>]*>>', T['cot'][i]) if '=' in m) if x is not None and x != gold[i]) for i in range(N)]
rng = np.random.default_rng(22140)
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))
for name, fn in (('CODI20', 'boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json'), ('zen', 'scratch/s220/zen_codi_gpt2/eval_gcs.json')):
    d = json.load(open(fn)); L = d['logit_lens']; print('==', name)
    for key in ('h_num', 'z_num'):
        obs, flo = [], []
        for j in range(6):
            v = [L[key][i][j] for i in range(N)]
            o = sum(1 for i in range(N) if v[i] is not None and v[i] != gold[i] and v[i] in inter[i]) / N
            # coincidence floor: v_i in inter(k) for a random other item k (100 draws per item), same first-match rule (not equal to item i's gold)
            hits = 0; tot = 0
            for i in range(N):
                ks = rng.integers(0, N, 100)
                for k in ks:
                    if k == i: continue
                    tot += 1; hits += int(v[i] is not None and v[i] != gold[i] and v[i] in inter[k])
            obs.append(round(o, 3)); flo.append(round(hits / tot, 3))
        print('  %s inter-rate obs %s  floor %s' % (key, obs, flo))
    G = d['golds']
    for gk in ('graph_matched', 'graph_matched_dne0'):
        P = d[gk]['pairs']
        for arm in ('cache', 'state'):
            out = [p['out'].get(arm) for p in P]; ocf = [p.get('ocf') for p in P]
            k = sum(1 for x, c in zip(out, ocf) if x is not None and c is not None and abs(float(x) - float(c)) < 1e-6); n = len(P)
            fl = []
            for _ in range(1000):
                perm = rng.permutation(n); fl.append(np.mean([1.0 if (out[i] is not None and ocf[perm[i]] is not None and perm[i] != i and abs(float(out[i]) - float(ocf[perm[i]])) < 1e-6) else 0.0 for i in range(n)]))
            print('  %s %s int-follow %d/%d = %.3f CI %s ; permuted-oracle floor %.3f (95th pct %.3f)' % (gk, arm, k, n, k / n, wilson(k, n), float(np.mean(fl)), float(np.quantile(fl, 0.95))))
