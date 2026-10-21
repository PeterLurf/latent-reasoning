import json, re
T = json.load(open('prosqa/ckpt_ext/hf/data/gsm8k_test.json'))
def f(x):
    try: return float(str(x).replace(',', '').rstrip('.'))
    except ValueError: return None
def nums(s): return set(v for v in (f(x) for x in re.findall(r'-?\d[\d,]*\.?\d*', s)) if v is not None)
N = len(T['answer'])
for name, fn in (('CODI20', 'boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json'), ('zen', 'scratch/s220/zen_codi_gpt2/eval_gcs.json')):
    d = json.load(open(fn)); L = d['logit_lens']; print('==', name)
    for key in ('z_num', 'h_num'):
        print(' ', key)
        for j in range(6):
            c = {'gold': 0, 'inter': 0, 'qnum': 0, 'other': 0}
            for i in range(N):
                v = L[key][i][j]; gold = f(T['answer'][i])
                inter = set(x for x in (f(m.split('=')[-1].rstrip('>')) for m in re.findall(r'<<[^>]*>>', T['cot'][i]) if '=' in m) if x is not None and x != gold)
                k = 'other' if v is None else 'gold' if v == gold else 'inter' if v in inter else 'qnum' if v in nums(T['question'][i]) else 'other'
                c[k] += 1
            print('    j%d ' % (j + 1) + ' '.join('%s %.3f' % (k, c[k] / N) for k in c))
