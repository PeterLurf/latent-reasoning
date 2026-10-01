"""THEORY §60 Phase A model: learned leaf encoder + shared merger over boundary-response tables, Q8 commits.

Tables are 16 x 4 over port slots (row r = seed bits PATTERNS[r]); slots >= nports and rows seeding them are padding.
Merger: a 2-layer d = 256 transformer over <= 56 tokens: 32 child row tokens (seed bits, 4 response codes, the local
vertex ids of the child's 4 slots, child role), 16 parent row queries (seed bits, the parent's slot ids) and <= 8 wiring
tokens (one per local vertex: in A / in B / survives into the parent). Local vertex ids are a random permutation per
call; no global id, age, size, depth or order feature; fresh activations per call. Output: sigmoid, padding -> 0, then
Q8 = round(255 p) / 255 with a straight-through gradient.
"""
import torch
import torch.nn as nn

from bm.data import MAXP, PATTERNS

NID = 9                                                       # 8 local ids + "none"
SEED = torch.tensor(PATTERNS, dtype=torch.float32)            # (16, 4)


def q8(p):
    q = torch.round(p * 255) / 255
    return p + (q - p).detach()


class BM(nn.Module):
    def __init__(self, d=256, layers=2, heads=8, binarise=False):
        super().__init__()
        self.binarise = binarise                              # evaluation: threshold every commit to {0, 1}
        self.leaf = nn.Sequential(nn.Linear(2 * (MAXP + 1), d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 16 * MAXP))
        self.idemb = nn.Embedding(NID, 32)
        self.row_in = nn.Linear(MAXP + MAXP + MAXP * 32 + 1, d)  # seed bits, codes, slot ids, valid flag
        self.q_in = nn.Linear(MAXP + MAXP * 32 + 1, d)
        self.w_in = nn.Linear(32 + 3, d)
        self.role = nn.Embedding(4, d)                        # 0 child A, 1 child B, 2 parent query, 3 wiring
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, MAXP)

    def commit(self, logits, rmask, cmask):
        p = torch.sigmoid(logits) * rmask[..., :, None] * cmask[..., None, :]
        if self.binarise:
            return (p > 0.5).float()
        return q8(p)

    def leaves(self, su, sv, rmask, cmask):
        """su, sv: (N,) slot of the edge's tail / head (MAXP = not a port: an interior endpoint). Returns (N, 16, 4)."""
        x = torch.cat([nn.functional.one_hot(su, MAXP + 1), nn.functional.one_hot(sv, MAXP + 1)], -1).float()
        return self.commit(self.leaf(x).view(-1, 16, MAXP), rmask, cmask)

    def merge(self, TA, TB, idA, idB, idP, rmA, rmB, rmP, cmP, wire):
        """TA, TB: (N, 16, 4) child tables; idA/idB/idP: (N, 4) local ids of slots (NID-1 = none);
        rm*: (N, 16) valid rows; cmP: (N, 4) valid parent columns; wire: (N, 8, 3) flags per local id (in A, in B, in P).
        Returns the committed parent table (N, 16, 4)."""
        N = TA.shape[0]; dev = TA.device
        seed = SEED.to(dev)[None].expand(N, -1, -1)
        def rows(T, ids, rm, role):
            sid = self.idemb(ids).reshape(N, 1, -1).expand(-1, 16, -1)
            x = self.row_in(torch.cat([seed, T, sid, rm[..., None].float()], -1))
            return x + self.role.weight[role]
        xa, xb = rows(TA, idA, rmA, 0), rows(TB, idB, rmB, 1)
        sidP = self.idemb(idP).reshape(N, 1, -1).expand(-1, 16, -1)
        xq = self.q_in(torch.cat([seed, sidP, rmP[..., None].float()], -1)) + self.role.weight[2]
        wid = self.idemb(torch.arange(8, device=dev))[None].expand(N, -1, -1)
        xw = self.w_in(torch.cat([wid, wire], -1)) + self.role.weight[3]
        x = torch.cat([xa, xb, xq, xw], 1)
        pad = torch.cat([~rmA, ~rmB, torch.zeros(N, 16, dtype=torch.bool, device=dev), wire.sum(-1) == 0], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        return self.commit(self.out(h[:, 32:48]), rmP, cmP)
