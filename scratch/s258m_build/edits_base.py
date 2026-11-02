"""§258 S0 INSTALL (LB2_v1 §2): --base gpt2 | gpt2-medium. Each edit asserted once; GPT-2 small stays bitwise unchanged (same loader call, same DeltaMem ints 6 / 6 + 6,
no rng drawn); gated by gate_s258m offbit (inject_aw + inject_ow direct, defaults, det) on the S3 card class.
  boxes/s5.py         load_gpt2(dev, base="gpt2") (+ MEDIUM / medium_sd: HF gpt2-medium model.safetensors, sha256 asserted, 'transformer.' stripped, lm_head dropped; H 16)
  boxes/nested170.py  make_trunk passes a.get("base")
  boxes/inject_ow.py  --base; build_model: base + DeltaMem geometry (L // 2, D / 128 + D / 128)
  scratch/s234_build/inject_aw_head.py  build_aw: base + the same geometry; the G12 assert on block L / 2
  scratch/s258m_build/edits_s258m.py (E258M, applied by mk_inject_aw.py)  inject_aw main(): --base
  scratch/s234_build/inject_aw_eval_parts.py  hh_* : the hidden width from the model (was 768)
then regenerates boxes/inject_aw.py / boxes/inject_aw_eval.py.
    python -B scratch/s258m_build/edits_base.py"""
import subprocess
import sys


def edit(p, E):
    s = open(p, encoding="utf-8").read()
    for o, n in E:
        assert s.count(o) == 1, ("§258 S0 edit", p, s.count(o), o[:80]); s = s.replace(o, n)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


s = open("boxes/s5.py", encoding="utf-8").read(); assert "def medium_sd" not in s, "already applied"
i = s.index("def load_gpt2(dev):"); end = "    return bc.GPT2KV(sd).to(dev)\n"; j = s.index(end, i) + len(end)
open("boxes/s5.py", "w", encoding="utf-8", newline="\n").write(s[:i] + open("scratch/s258m_build/s5_medium.py", encoding="utf-8").read() + s[j:])
edit("boxes/nested170.py", [("        return s5.load_gpt2(dev)\n", "        return s5.load_gpt2(dev, a.get(\"base\") or \"gpt2\")                   # §258 S0: --base\n")])
ARM = '    ap.add_argument("--arm", required=True, choices=["direct", "inject"]); ap.add_argument("--think_k", type=int, default=0); ap.add_argument("--d_max", type=int, default=8)\n'
BASE = ('    ap.add_argument("--base", default="gpt2", choices=["gpt2", "gpt2-medium"], help="§258 S0: the pretrained trunk; DeltaMem at layer L // 2 with D / 128 + D / 128 '
        'heads of 64 (small: 6, 6 + 6)")\n')
edit("boxes/inject_ow.py", [
    ('    g = load_trunk(dev, {"scratch": ""})\n    if a.arm == "direct":\n        return None, g, g\n    m = InjectOW(g, 6, a.mem_write, 6, 6, None).to(dev)',
     '    g = load_trunk(dev, {"scratch": "", "base": getattr(a, "base", "gpt2")})   # §258 S0: --base\n    if a.arm == "direct":\n        return None, g, g\n'
     '    lm, hm = g.L // 2, g.D // 128                                          # §258 S0: DeltaMem geometry (small: 6, 6 + 6)\n'
     '    m = InjectOW(g, lm, a.mem_write, hm, hm, None).to(dev)'),
    (ARM, ARM + BASE)])
edit("scratch/s234_build/inject_aw_head.py", [
    ('    g = load_trunk(dev, {"scratch": ""})\n    if getattr(a, "init_trunk", ""):', '    g = load_trunk(dev, {"scratch": "", "base": getattr(a, "base", "gpt2")})   # §258 S0: --base\n    if getattr(a, "init_trunk", ""):'),
    ('    m = cls(g, 6, a.mem_write, 6, 6, None).to(dev); m.nopass = False\n'
     '    assert torch.equal(m.mem.qkv.weight.detach(), g.w("h.6.attn.c_attn.weight").detach().float().T), "G12: DeltaMem qkv = the loaded block-6 c_attn"',
     '    lm, hm = g.L // 2, g.D // 128                                          # §258 S0: DeltaMem at L // 2, D / 128 + D / 128 heads of 64 (small: 6, 6 + 6)\n'
     '    m = cls(g, lm, a.mem_write, hm, hm, None).to(dev); m.nopass = False\n'
     '    assert torch.equal(m.mem.qkv.weight.detach(), g.w("h.%d.attn.c_attn.weight" % lm).detach().float().T), "G12: DeltaMem qkv = the loaded block-L/2 c_attn"')])
edit("scratch/s234_build/inject_aw_eval_parts.py", [
    ("    N = len(T); H = np.zeros((N, K + 1, 768), dtype=np.float32)", "    N = len(T); H = np.zeros((N, K + 1, m.g.D), dtype=np.float32)   # §258 S0: the model's width (was 768)"),
    ("    Hf = np.zeros((HH_N, HH_KMAX + 1, 768), dtype=np.float32)", "    Hf = np.zeros((HH_N, HH_KMAX + 1, m.g.D), dtype=np.float32)"),
    ("    Xtr, ytr = Hf[tr].reshape(-1, 768), lab[tr].reshape(-1); Xva, yva = Hf[va].reshape(-1, 768), lab[va].reshape(-1)",
     "    Xtr, ytr = Hf[tr].reshape(-1, m.g.D), lab[tr].reshape(-1); Xva, yva = Hf[va].reshape(-1, m.g.D), lab[va].reshape(-1)"),
    ("p = hh_prob(Hd.reshape(-1, 768), mu, sd, w, b)", "p = hh_prob(Hd.reshape(-1, m.g.D), mu, sd, w, b)")])
m = "scratch/s234_build/mk_inject_aw.py"; t = open(m, encoding="utf-8").read(); a = 'assert "tok_st(" not in main and "inject_st_eval" not in main'
assert t.count(a) == 1 and "E258M" not in t
t = t.replace(a, 'for o, n in runpy.run_path("scratch/s258m_build/edits_s258m.py")["E258M"]:   # §258 S0: --base (default gpt2 = unchanged)\n'
              '    assert main.count(o) == 1, ("§258 S0 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)\n' + a, 1)
open(m, "w", encoding="utf-8", newline="\n").write(t)
open("scratch/s258m_build/edits_s258m.py", "w", encoding="utf-8", newline="\n").write('"""§258 S0: inject_aw main() --base (applied by mk_inject_aw.py after E257)."""\nE258M = [(%r, %r)]\n' % (ARM, ARM + BASE))
for g_ in ("scratch/s234_build/mk_inject_aw.py", "scratch/s234_build/mk_inject_aw_eval.py"):
    assert subprocess.call([sys.executable, "-B", g_]) == 0, g_
print("§258 S0 base edits applied")
