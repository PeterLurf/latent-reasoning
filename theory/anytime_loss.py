"""Anytime (minimum-progress) supervision as a standalone loss (THEORY §9/§16/§18 Prop. 2; PROPOSALS T5).

After every latent step k (1..K) the answer is read out and supervised only where the model can already be expected to
know it: negatives (unreachable) always; positives whose reachability distance is <= k + C. Deeper positives are masked,
so reaching them early is never penalised and never required. The final step k = K is the ordinary loss (all targets).

    loss = anytime_bce(logits, y, dist, C=1)

  logits : (K, *S) per-step answer logits (step k's readout at index k-1), any trailing shape S
  y      : (*S) targets in {0, 1} (1 = reachable)
  dist   : (*S) reachability distance of each positive (ignored where y == 0); any value where y == 0
  C      : look-ahead slack — hops the readout itself adds after the latents (our delayed query: C = 1)
  window : optional 0/1 scalar or tensor; 0 disables the intermediate terms (plain final-step loss), as in the
           stage-windowed arms of THEORY §23 (--anytime_stages)
Returns the masked-mean BCE over all (step, target) pairs that are supervised. Identical to
theory/train_arch.py AnytimeMixin._anytime_loss (checked by the self-test below).

For a multi-class readout (e.g. an LLM choosing an answer token), use anytime_ce: per step, cross-entropy on examples
whose required depth <= k + C, the final step on all examples.
    python theory/anytime_loss.py        # self-test
"""
import torch
import torch.nn.functional as F


def _mask(y, dist, K, C, window=None):
    ks = torch.arange(1, K + 1, device=dist.device, dtype=dist.dtype).view(K, *([1] * y.dim()))
    m = ((y[None] == 0) | (dist[None] <= ks + C)).float()
    if window is not None:
        m = m * window
    m[-1] = 1.0
    return m


def anytime_bce(logits, y, dist, C=1, window=None):
    K = logits.shape[0]
    m = _mask(y, dist, K, C, window)
    b = F.binary_cross_entropy_with_logits(logits.float(), y.float()[None].expand_as(logits), reduction="none")
    return (b * m).sum() / m.sum()


def anytime_ce(logits, target, depth, C=1, window=None):
    """logits (K, B, V); target (B,) class ids; depth (B,) hops the example needs. Step k supervises examples with
    depth <= k + C; the last step supervises all."""
    K = logits.shape[0]
    ks = torch.arange(1, K + 1, device=depth.device, dtype=depth.dtype)[:, None]
    m = (depth[None] <= ks + C).float()
    if window is not None:
        m = m * window
    m[-1] = 1.0
    ce = F.cross_entropy(logits.float().flatten(0, 1), target.repeat(K), reduction="none").view(K, -1)
    return (ce * m).sum() / m.sum()


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_arch import AnytimeMixin
    g = torch.Generator().manual_seed(0)
    K, B, Q = 6, 32, 16
    ld, lu = torch.randn(K, B, Q, generator=g), torch.randn(K, B, generator=g)
    qy, ty = torch.randint(0, 2, (B, Q), generator=g).float(), torch.randint(0, 2, (B,), generator=g).float()
    qd, tgtd = torch.randint(1, 10, (B, Q), generator=g), torch.randint(1, 10, (B,), generator=g)
    ref_d, ref_u = AnytimeMixin()._anytime_loss(ld, lu, qd, qy, tgtd, ty, 1)
    print("delayed diff", (anytime_bce(ld, qy, qd, 1) - ref_d).abs().item(),
          "upfront diff", (anytime_bce(lu, ty, tgtd, 1) - ref_u).abs().item())
    print("window=0 equals final-step BCE:",
          (anytime_bce(ld, qy, qd, 1, window=0.0) - F.binary_cross_entropy_with_logits(ld[-1], qy)).abs().item())
    lc = torch.randn(K, B, 5, generator=g); tg = torch.randint(0, 5, (B,), generator=g); dp = torch.randint(1, 9, (B,), generator=g)
    print("anytime_ce window=0 equals final CE:",
          (anytime_ce(lc, tg, dp, 1, window=0.0) - F.cross_entropy(lc[-1], tg)).abs().item())
