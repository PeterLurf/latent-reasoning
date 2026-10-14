import re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
# usage: python -I parse_arx.py <xml> <out.jsonl>
t = open(sys.argv[1], encoding='utf-8').read()
out = open(sys.argv[2], 'a', encoding='utf-8')
for e in t.split('<entry>')[1:]:
    i = re.search(r'<id>(.*?)</id>', e).group(1).split('/abs/')[-1]
    ti = ' '.join(re.search(r'<title>(.*?)</title>', e, re.S).group(1).split())
    ab = ' '.join(re.search(r'<summary>(.*?)</summary>', e, re.S).group(1).split())
    pub = re.search(r'<published>(.*?)</published>', e).group(1)[:10]
    au = re.findall(r'<name>(.*?)</name>', e)
    print(i, '|', ti, '|', pub, '|', ', '.join(au[:5]) + (' et al.' if len(au) > 5 else ''))
    out.write(json.dumps(dict(id=i, title=ti, pub=pub, authors=au, abstract=ab)) + '\n')
