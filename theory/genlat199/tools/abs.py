import sys, re, html, glob, os
sys.stdout.reconfigure(encoding="utf-8")
d = sys.argv[1]
ids = sys.argv[2:]
for i in ids:
    p = os.path.join(d, i + ".html")
    s = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r'<meta name="citation_abstract" content="(.*?)"\s*/?>', s, re.S)
    a = html.unescape(m.group(1)) if m else "NO ABSTRACT"
    au = re.findall(r'<meta name="citation_author" content="(.*?)"', s)
    dt = re.search(r'<meta name="citation_date" content="(.*?)"', s)
    print("=== %s | %s | %s" % (i, "; ".join(au[:4]) + (" et al." if len(au) > 4 else ""), dt.group(1) if dt else ""))
    print(re.sub(r"\s+", " ", a).strip())
