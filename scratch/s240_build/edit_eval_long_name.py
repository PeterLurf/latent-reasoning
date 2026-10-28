"""§240: run_long writes <out>/<name> (--long_name; default eval_long.json = §239 unchanged) so eval_long_k128.json does not overwrite §239's file."""
p = "scratch/s234_build/inject_aw_eval_parts.py"; s = open(p, encoding="utf-8").read()
E = [('def run_long(out, dev, bs, kplist, ds):', 'def run_long(out, dev, bs, kplist, ds, name="eval_long.json"):'),
     ('''    """§239: load <out>/model.pt as load() does, build the run law's sets (set shas asserted vs args.json), part_long, write ONLY <out>/eval_long.json."""''',
      '''    """§239: load <out>/model.pt as load() does, build the run law's sets (set shas asserted vs args.json), part_long, write ONLY <out>/<name> (default
    eval_long.json; §240: eval_long_k128.json)."""'''),
     (r'''    json.dump(r_, open(os.path.join(out, "eval_long.json"), "w", newline="\n")); print("LONG done", out, r_["sec"], "s", flush=True)''',
      r'''    json.dump(r_, open(os.path.join(out, name), "w", newline="\n")); print("LONG done", out, name, r_["sec"], "s", flush=True)''')]
for o, n in E:
    assert s.count(o) == 1, ("parts", o[:60]); s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = "scratch/s234_build/mk_inject_aw_eval.py"; s = open(p, encoding="utf-8").read()
o = r''' '    ap.add_argument("--long_ds", default="2,3,4,5", help="§239: the d values of run_long")\n'
'''
n = r''' '    ap.add_argument("--long_ds", default="2,3,4,5", help="§239: the d values of run_long")\n'
 '    ap.add_argument("--long_name", default="eval_long.json", help="§240: the run_long output file (e.g. eval_long_k128.json)")\n'
'''
assert s.count(o) == 1, "mk long_ds"; s = s.replace(o, n)
o2 = '''[int(x) for x in a.long_ds.split(",")]); return'''
n2 = '''[int(x) for x in a.long_ds.split(",")], a.long_name); return'''
assert s.count(o2) == 1, "mk call"; s = s.replace(o2, n2); open(p, "w", encoding="utf-8", newline="\n").write(s); print("ok")
