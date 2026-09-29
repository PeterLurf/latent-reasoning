"""Whole-training-step CUDA-graph capture (forward + backward + AdamW).

Shapes are static (fixed batch, edge count, query count), so the step can be
captured once and replayed; this removes the kernel-launch overhead that
dominates such a small model under Windows/WDDM. Numerically this is the same
computation as the eager step (fp32 on sm<80, bf16 autocast otherwise).
Loss weights and the learning rate live in device tensors so schedules can
change them between replays.

K may be a list of latent budgets: one graph is captured per K, all sharing one
memory pool (safe here: they never run concurrently and every graph's outputs
stay alive), and step(batch, K) replays the chosen one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GraphedStep:
    def __init__(self, model, opt, K, B, E, Q, upfront_weight, device, amp_dtype=None, clip=1.0, Qp=8):
        self.model, self.opt = model, opt
        self.Ks = [K] if isinstance(K, int) else sorted(K)
        self.clip, self.amp = clip, amp_dtype
        L = lambda *s: torch.zeros(*s, dtype=torch.long, device=device)
        Fz = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
        from .graphs import V
        self.inp = dict(src=L(B, E), dst=L(B, E), start=L(B), q=L(B, Q), qy=Fz(B, Q), tgt=L(B), ty=Fz(B),
                        pq=L(B, Qp), py=L(B, Qp), cq=L(B, Qp), cy=Fz(B, Qp, V))
        self.w_children = torch.tensor(0.0, device=device)
        self.w_up = torch.tensor(float(upfront_weight), device=device)
        self.w_aux = torch.tensor(0.0, device=device)
        self.device = device
        torch.cuda.set_device(device)   # capture must run with the model's device current
        self._capture()

    def _fwd_bwd(self, K):
        i = self.inp
        with torch.autocast("cuda", dtype=self.amp or torch.float32, enabled=self.amp is not None,
                            cache_enabled=False):
            ld, lu, lp, lc = self.model.run_both(i["src"], i["dst"], i["start"], K, i["q"], i["tgt"],
                                                 i["pq"], i["cq"])
        self.loss_d = F.binary_cross_entropy_with_logits(ld.float(), i["qy"])
        self.loss_u = F.binary_cross_entropy_with_logits(lu.float(), i["ty"])
        self.loss_p = F.cross_entropy(lp.float().flatten(0, 1), i["py"].flatten())
        self.loss_c = F.binary_cross_entropy_with_logits(lc.float(), i["cy"]) * lc.shape[-1] / 4
        self.loss = (self.loss_d + self.w_up * self.loss_u + self.w_aux * self.loss_p
                     + self.w_aux * self.w_children * self.loss_c)
        self.loss.backward()
        self.gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.opt.step()

    def _capture(self):
        # warm up on a dummy batch, then restore params and optimizer state in place
        saved = [p.detach().clone() for p in self.model.parameters()]
        self.graphs, self.outs, pool = {}, {}, None
        for K in reversed(self.Ks):             # largest first, so the shared pool is sized once
            s = torch.cuda.Stream(self.device)
            s.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.opt.zero_grad(set_to_none=False)
                    self._fwd_bwd(K)
            torch.cuda.current_stream(self.device).wait_stream(s)
            self.loss = self.loss_d = self.loss_u = self.loss_p = self.loss_c = self.gn = None   # release warmup graph
            g = torch.cuda.CUDAGraph()
            self.opt.zero_grad(set_to_none=False)
            with torch.cuda.graph(g, pool=pool):
                self.opt.zero_grad(set_to_none=False)
                self._fwd_bwd(K)
            pool = g.pool()
            self.graphs[K] = g
            # detached views share storage (replays still update them) but free this capture's autograd graph
            self.outs[K] = tuple(t.detach() for t in (self.loss, self.loss_d, self.loss_u, self.loss_p,
                                                      self.loss_c, self.gn))
            self.loss = self.loss_d = self.loss_u = self.loss_p = self.loss_c = self.gn = None
        self._select(self.Ks[-1])
        with torch.no_grad():
            for p, s0 in zip(self.model.parameters(), saved):
                p.copy_(s0)
            for st in self.opt.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        v.zero_()

    def _select(self, K):
        self.loss, self.loss_d, self.loss_u, self.loss_p, self.loss_c, self.gn = self.outs[K]

    def step(self, batch, K=None):
        K = self.Ks[-1] if K is None else K
        for k, v in self.inp.items():
            v.copy_(batch[k], non_blocking=True)
        self.graphs[K].replay()
        self._select(K)
