import pathlib
p = pathlib.Path(r"D:\Latent\latent\model.py")
s = p.read_text(encoding="utf-8")
rep = [
 ('''                 init="gpt2", feedback="post_ln"):
        super().__init__()''',
  '''                 init="gpt2", feedback="post_ln", step_emb=True):
        super().__init__()'''),
 ('''        if init == "gpt2":
            self.apply(self._init)
        # init == "torch": PyTorch defaults (kaiming-uniform Linear, N(0,1) embeddings)''',
  '''        if init == "gpt2":
            self.apply(self._init)
        # init == "torch": PyTorch defaults (kaiming-uniform Linear, N(0,1) embeddings)
        self.use_step = step_emb
        if not step_emb:            # ablation: no step embedding (table zeroed; it gets no gradient, so it stays 0)
            nn.init.zeros_(self.step_emb.weight)'''),
 ('''        x = (base + te[T_LAT] + self.step_emb.weight[k])[:, None]''',
  '''        x = (base + te[T_LAT] + (self.step_emb.weight[k] if getattr(self, "use_step", True) else 0))[:, None]'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:50]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

p = pathlib.Path(r"D:\Latent\latent\train.py")
s = p.read_text(encoding="utf-8")
a = '''    p.add_argument("--resume", action="store_true",'''
b = '''    p.add_argument("--no_step_emb", action="store_true", help="ablation: latent steps get no step embedding")
    p.add_argument("--resume", action="store_true",'''
assert s.count(a) == 1; s = s.replace(a, b)
a = '''emb=a.emb, d_id=a.d_id, init=a.init, feedback=a.feedback).to(dev)'''
b = '''emb=a.emb, d_id=a.d_id, init=a.init, feedback=a.feedback, step_emb=not a.no_step_emb).to(dev)'''
assert s.count(a) == 1; s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

for f, old in ((r"D:\Latent\latent\competence.py", '''init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln")).to(device)'''),
               (r"D:\Latent\latent\development.py", '''init=c.get("init", "torch"),
                       feedback=c.get("feedback", "post_ln")).to(device)''')):
    p = pathlib.Path(f)
    s = p.read_text(encoding="utf-8")
    assert s.count(old) == 1, f
    s = s.replace(old, old.replace('feedback=c.get("feedback", "post_ln"))', 'feedback=c.get("feedback", "post_ln"), step_emb=not c.get("no_step_emb", False))'))
    p.write_text(s, encoding="utf-8")
print("ok")
