"""THEORY §228 LAT-OW (LAT_v1.md at e78d7211; draft §2 / §8 for code): §184's INJECT recipe (boxes/nested170.py --arm inject --aux_step 1 --aux_fade
48000,48000 --aux_reweight --fast_add) ported to the TOPO word programs (boxes/ow_words.py, order_mode topo, dag on, mod 11, train frames 0-5).
nested170 / deltamem / deltas5 / fsq120 are imported read-only; class InjectOW(InjectN) overrides only think_inputs / dec_logits / aux_logits, for
token-list items (the adapters below). Every other kernel (write, lower, think, aux_prompt, aux_greedy, AUX_W) is nested170's.
    item (tok_ow)  think = [EOT] + enc(" " + text)                          (the question ends "... have?"; a fresh pass re-reads it every pass)
                   dec   = think + enc(" The answer is:")                   -> the number token " 0" .. " 10" (one GPT-2 token each)
                   aux_t = enc(" " + R_step[t]) + [EOT] after [EOT, " Step", " t", " :"]   (t <= d; read-only on S_t; no target for t > d; dropped at test)
    arms           direct = the plain trunk on dec (one pass, no memory; nested170's direct); inject = K think passes, then dec reads S_K
    d per step     one d per optimiser step from rng = default_rng(seed + 22331), d = rng.integers(1, d_max + 1) (codi_ps's law: TEXT draws the same
                   sequence); items from OW.Stream(22930, excl, frames 0-5, topo, dag on) - their own stream, so DIRECT and LAT see the identical items.
    excl           TEST2 (22960) + TRANSFER (frames 6-7, 22961) + PAIRS2 candidates (22970 / 22971, recipients and donors) + floor donors (22975) +
                   the §226 / §227 sets (OW.test_set, OW.pair_candidates, §227's 22950 decode set), all TOPO / dag on.
Gate only: --src nested trains on nested170's own items (train_items, its EXCL, its 17001 calibration) rendered to token lists (tok_nested), for the
port-kernel identity gate vs nested170 --arm inject (scratch/gate_s228.py). Logs (log.jsonl): nested170's 100-step record (+ gn_preclip_mean100) and a
1k-bin record per 1,000 steps: by_d {steps, acc, ce, clip} (train batches; acc = argmax over the 11 number tokens), aux value-token acc by pass (inject),
dseq (the bin's d sequence as digits: G-EXP) and ihash (sha1 of the bin's (d, item sig) list); exposure_by_d at every bin; entry_d1 (the first 100
consecutive d = 1 steps with mean answer CE < 0.5). Full-state checkpoints (--resume <ckpt>), --stop_at, --args_only / --args_ref, --mem_check /
--auto_micro (the longest of 256 d_max items at K, then the d 12 eval shape at K' 14), --time_steps, --futility (LAT_v1 §4). After training:
model.pt, then boxes/inject_ow_eval.py's --eval_parts (default acc) -> eval_<part>.json + eval.json.
    python -m boxes.inject_ow --arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 8000 --out ..."""
import os
os.environ.setdefault("BOXES_NB", "5")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import boxes.nested170 as N7  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import InjectN, AUX_W, EOT, NUM_IDS, pad  # noqa: E402

ENC = N7.ENC
SFX = ENC.encode(" The answer is:"); HAS_TOK = ENC.encode(" has")[0]
assert len(SFX) == 4 and all(len(ENC.encode(" %d" % v)) == 1 for v in range(11))
SEED_D = 22331                                                              # + seed: the d-per-step stream (codi_ps's)
SEED_TEST2, SEED_TRANSFER, SEED_PAIRS2, SEED_FLOOR2, SEED_PERM2 = 22960, 22961, 22970, 22975, 22980   # LAT_v1 draft §3 (unused elsewhere, grep 10-23)
SEED_CAL, SEED_MEMCHECK = 22996, 22997                                      # g_add calibration / memory-check items (unused elsewhere, grep 10-23)
OWKW = {"order_mode": "topo", "dag": True}
PAIR_DS = range(3, 9); PAIR_TRIES = 3000; PAIR_CAP = 1000; FLOOR_K = 3; TEST_DS = range(1, 13); TEST_N = 1000


def tok_ow(it):
    th = [EOT] + ENC.encode(" " + it["text"]); aux = [ENC.encode(" " + r) + [EOT] for r in it["R_step"]]
    return {"think": th, "dec": th + SFX, "ans": int(it["answer"]), "aux": aux, "vidx": [q.index(HAS_TOK) + 1 for q in aux], "d": it["d"], "sig": it["sig"]}


def tok_nested(it):
    """gate only: a nested170 item as token lists (think [EOT, expr, ' .'], dec [EOT, expr, ' ='], aux e_{t+1} + EOT) - nested170's own sequences."""
    return {"think": [EOT] + N7.ids(it["syms"]) + [N7.TID[" ."]], "dec": [EOT] + N7.ids(it["syms"]) + [N7.TID[" ="]], "ans": int(it["t"]),
            "aux": [N7.ids(e) + [EOT] for e, _ in N7.aux_of(it)], "vidx": None, "d": it["n"], "sig": " ".join(it["syms"])}


class InjectOW(InjectN):
    """nested170.InjectN on token-list items: only the three input builders change; the code below is nested170's with it["think"] / it["dec"] /
    it["aux"] in place of the rendered expression (bitwise the same ops on the same token lists: the port-kernel identity gate)."""
    def think_inputs(self, items, dev):
        ids_, m_ = pad([it["think"] for it in items]); return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev)

    def dec_logits(self, S, items, dev):
        from boxes.deltas5 import fwd_read
        seqs = [it["dec"] for it in items]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
        h = fwd_read(self.g, self.g.embed(x), self.mem, S); pos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
        return self.g.logits(h[torch.arange(len(seqs), device=dev), pos]).float()

    def aux_logits(self, S, items, t, dev, hold=False):
        from boxes.deltas5 import fwd_read
        assert not hold, "§228: no hold targets (NOFADE, t <= d only)"
        pre = self.aux_prompt(t); seqs, tg = [], []
        for it in items:
            assert t <= len(it["aux"]), "§228: no aux target past the last step"
            tgt = it["aux"][t - 1]; seqs.append(pre + tgt[:-1]); tg.append(tgt)
        x_, _ = pad(seqs); y_, _ = pad(tg, -100); x = torch.tensor(x_, device=dev); y = torch.tensor(y_, device=dev)
        assert x.shape[1] == len(pre) + y.shape[1] - 1
        h = fwd_read(self.g, self.g.embed(x), self.mem, S)
        return self.g.logits(h[:, len(pre) - 1:]).float(), y


def direct_logits(g, items, dev, ac):
    """nested170's direct arm on dec: the plain trunk under autocast, the logits at the last dec position outside it (nested170's order)."""
    seqs = [it["dec"] for it in items]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
    with ac():
        h, _ = g(g.embed(x), None)
    pos = torch.tensor([len(q) - 1 for q in seqs], device=dev); return g.logits(h[torch.arange(len(seqs), device=dev), pos]).float()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cf(rec, don, t, mode):
    """lat_checks.cf: the counterfactual answer when the recipient's chain continues from the donor's state at pass t: mode 'a' = every state
    0..t from the donor (back-references read from memory); 'b' = only S_t from the donor, earlier states the recipient's own (re-read from the text)."""
    S = list(don["values"][:t + 1]) if mode == "a" else list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(OW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def build_sets():
    """the §228 evaluation sets (deterministic; lat_checks part C's pair / floor law with the exact-length rule made binding) and the training
    exclusion. -> dict: test2 {d: [items]}, transfer {d: [items]}, pairs {d: [{rec, don, t, ca, cb}]} (kept, first PAIR_CAP), floors {d: [[donor or
    None] x FLOOR_K]} (per kept pair), pair_stats, excl (sigs), shas."""
    s = OW.Stream(SEED_TEST2, frames=OW.FRAME_TRAIN, **OWKW); test2 = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    s = OW.Stream(SEED_TRANSFER, frames=(6, 7), **OWKW); transfer = {d: [s.draw(d) for _ in range(TEST_N)] for d in TEST_DS}
    r = np.random.default_rng(SEED_PAIRS2); s = OW.Stream(SEED_PAIRS2 + 1, frames=OW.FRAME_TRAIN, **OWKW); rf = np.random.default_rng(SEED_FLOOR2)
    pairs, floors, pst, cand = {}, {}, {}, []
    for d in PAIR_DS:
        t = math.ceil(d / 2); kept = []; n_try = 0; len_mis = 0
        while n_try < PAIR_TRIES:
            rec = s.draw(d); don = OW.redraw(r, rec)
            if don is None or don["sig"] == rec["sig"]:
                continue
            n_try += 1; cand += [rec, don]
            own, dfin, ca, cb = rec["answer"], don["answer"], cf(rec, don, t, "a"), cf(rec, don, t, "b")
            if len({own, dfin, ca}) == 3 and cb not in (own, dfin):
                if len(ENC.encode(rec["text"])) != len(ENC.encode(don["text"])):
                    len_mis += 1; continue                                  # exact token length (lat_checks: 0 mismatches)
                kept.append({"rec": rec, "don": don, "t": t, "ca": ca, "cb": cb})
        fl = []
        for p_ in kept[:PAIR_CAP]:                                          # lat_checks: 3 redraw donors per kept pair (None = no donor found)
            fl.append([OW.redraw(rf, p_["rec"]) for _ in range(FLOOR_K)])
        pairs[d] = kept[:PAIR_CAP]; floors[d] = fl; pst[str(d)] = {"t": t, "candidates": n_try, "kept": len(kept), "len_mismatch": len_mis, "used": len(pairs[d])}
        cand += [x for row in fl for x in row if x is not None]
    TE6 = OW.test_set(**OWKW); PC6 = OW.pair_candidates(**OWKW); ex0 = {it["sig"] for d, its in TE6.items() for it in its}   # §226 / §227 (codi_ps's order)
    es = OW.Stream(22950, ex0, frames=OW.FRAME_TRAIN, **OWKW); ED7 = [es.draw(d) for d in range(1, 9) for _ in range(200)]
    excl = ex0 | {it["sig"] for it in ED7} | {x["sig"] for d, ps in PC6.items() for pr in ps for x in pr}
    excl |= {it["sig"] for d in test2 for it in test2[d]} | {it["sig"] for d in transfer for it in transfer[d]} | {x["sig"] for x in cand}
    shas = {"test2_sha": OW.set_sha([it for d in TEST_DS for it in test2[d]]), "transfer_sha": OW.set_sha([it for d in TEST_DS for it in transfer[d]]),
            "pairs2_sha": OW.set_sha([x for d in PAIR_DS for p_ in pairs[d] for x in (p_["rec"], p_["don"])]),
            "floors2_sha": OW.set_sha([x for d in PAIR_DS for row in floors[d] for x in row if x is not None]),
            "s226_test_sha": OW.set_sha([it for d in sorted(TE6) for it in TE6[d]]), "s227_decode_sha": OW.set_sha(ED7)}
    return {"test2": test2, "transfer": transfer, "pairs": pairs, "floors": floors, "pair_stats": pst, "excl": excl, "shas": shas}


def load_trunk(dev, n7args):
    return N7.make_trunk(n7args, dev)


def build_model(a, dev):
    """-> (m or None, g, params). torch.manual_seed(a.seed) must precede (nested170's init order: seed, trunk, DeltaMem)."""
    g = load_trunk(dev, {"scratch": ""})
    if a.arm == "direct":
        return None, g, g
    m = InjectOW(g, 6, a.mem_write, 6, 6, None).to(dev); m.nopass = False; return m, g, m


META = {"device", "train_gpu", "train_driver", "torch_version", "cuda_version", "cudnn_version", "python_version", "image_tag", "hostprobe_sps", "micro", "accumulation",
        "out", "resume", "args_only", "args_ref", "stop_at", "auto_micro", "mem_check", "time_steps", "mem_budget", "eval_bs", "eval_parts", "module_sha256",
        "nested170_sha256", "deltamem_sha256", "eval_module_sha256", "set_build_sec", "log_every", "ckpt_every"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["direct", "inject"]); ap.add_argument("--think_k", type=int, default=0); ap.add_argument("--d_max", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true"); ap.add_argument("--mem_write", default="top", choices=["top"])
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000); ap.add_argument("--cache_lower", action="store_true")
    ap.add_argument("--aux_step", type=float, default=0.0); ap.add_argument("--aux_fade", default="48000,48000"); ap.add_argument("--aux_reweight", action="store_true")
    ap.add_argument("--fast_add", action="store_true"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes / gates only")
    ap.add_argument("--src", default="ow", choices=["ow", "nested"], help="nested = GATE ONLY (port-kernel identity vs nested170)"); ap.add_argument("--n_max_train", type=int, default=8, help="--src nested only")
    ap.add_argument("--futility", default="", help="LAT_v1 §4: 'step1,step2' (24000,32000): stop if the d = 2 train acc of the last 1k bin < 0.50 and the OLS slope of the "
                    "last 4 d = 2 bins <= +0.02 per bin (a still-rising first check defers to the second); '' = off")
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--stop_at", type=int, default=0, help="train to this step under the --steps schedule (ckpt at the stop step), then exit 9 (no eval)")
    ap.add_argument("--time_steps", type=int, default=0); ap.add_argument("--args_only", action="store_true"); ap.add_argument("--args_ref", default="")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--eval_parts", default="acc"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); a.task = "ow_topo" if a.src == "ow" else "nested"
    import boxes.deltamem as DM_
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    from boxes.deltas5 import write_grad_norm
    DM_.FAST_ADD = bool(a.fast_add); os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale
    if a.arm == "direct":
        assert not (a.think_k or a.aux_step or a.aux_reweight or a.cache_lower or a.fast_add or a.futility), "DIRECT has no latent flags"
        a.mem_write = a.write_mix_start = a.write_mix_end = None
    else:
        assert a.think_k >= 1 and a.aux_step > 0, "§228 inject: K >= 1 with the step aux"
    assert a.src == "ow" or a.d_max == 8, "--src nested: n ~ U{1..n_max_train} per item (nested170's law)"
    assert 1 <= a.d_max <= 8 and a.bs % a.micro == 0
    K = a.think_k if a.arm == "inject" else 0
    torch.manual_seed(a.seed); rng_d = np.random.default_rng(a.seed + SEED_D); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    t_sets = time.time()
    if a.src == "ow":
        SETS = build_sets(); EXCL = SETS["excl"]; stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=OW.FRAME_TRAIN, **OWKW)
    else:                                                                   # gate only: nested170's TE / SPOOL exclusion and item law
        TE = N7.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
        for _ in range(2200):
            r__, _t = N7.generate(r_sp, 8); EXCL.add(tuple(N7.render(r__)))
        SETS = None; stream = None
    set_sec = round(time.time() - t_sets)
    m, g, params = build_model(a, dev)
    f0_, f1_ = [float(x_) for x_ in a.aux_fade.split(",")]

    def lam_at(step):
        if not a.aux_step:
            return 0.0
        return a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc))
    lam_now = [lam_at(1)]

    def draw_items(d):
        if a.src == "nested":
            return [tok_nested(it) for it in N7.train_items(rng, a.bs, a.n_max_train, EXCL)]
        return [tok_ow(stream.draw(d)) for _ in range(a.bs)]

    def loss_items(items, stat=None):
        """nested170.loss_items (direct; inject with the aux on) on token-list items; stat (dict) collects the log fields (no effect on the graph)."""
        tgt = torch.tensor(NUM_IDS[[it["ans"] for it in items]], device=dev)
        if a.arm == "direct":
            lg = direct_logits(g, items, dev, ac); la = Fn.cross_entropy(lg, tgt); loss = la
        else:
            from torch.utils.checkpoint import checkpoint
            x, msk = m.think_inputs(items, dev); per = [[] for _ in items]; vok = {}
            with ac():
                S = m.mem.zero_state(len(items), dev); hl = m.lower(x) if a.cache_lower else None
                for t in range(1, K + 1):
                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)
                    idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]
                    if idx:
                        lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev)
                        ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                        for r_, i in enumerate(idx):
                            per[i].append(ce[r_] * AUX_W[t - 1] if a.aux_reweight else ce[r_])
                        if stat is not None and items[idx[0]]["vidx"] is not None:   # aux value-token acc (teacher-forced argmax at the value position)
                            with torch.no_grad():
                                pr_ = lg_a.detach().argmax(-1); vi = torch.tensor([items[i]["vidx"][t - 1] for i in idx], device=dev); ar = torch.arange(len(idx), device=dev)
                                vok[t] = (pr_[ar, vi] == y_a[ar, vi]).float().mean().item()
                lg = m.dec_logits(S, items, dev)
            la = Fn.cross_entropy(lg, tgt)
            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean(); loss = la + lam_now[0] * aux
            if stat is not None:
                stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))
        if stat is not None:
            with torch.no_grad():
                stat["acc"] = stat.get("acc", 0.0) + float((lg.detach()[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array([it["ans"] for it in items])).sum())
            stat["la"] = stat.get("la", 0.0) + float(la.detach()) * len(items); stat["n"] = stat.get("n", 0) + len(items)
        return loss

    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, **OWKW) if a.src == "ow" else None
    MC_TR = sorted([tok_ow(s_mc.draw(a.d_max)) for _ in range(256)], key=lambda it: -len(it["dec"])) if s_mc else None
    MC_EV = sorted([tok_ow(s_mc.draw(12)) for _ in range(256)], key=lambda it: -len(it["dec"]))[:1] if s_mc else None

    def mem_train(micro):
        """the training step's peak at bs / micro of the longest d_max items + AdamW's two fp32 states (nested170's rule)."""
        torch.cuda.reset_peak_memory_stats(dev); lam_now[0] = a.aux_step; l_ = loss_items(MC_TR[:a.bs // micro]); l_.backward(); params.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 2 * sum(p_.numel() * 4 for p_ in params.parameters() if p_.requires_grad) / 2 ** 30

    def mem_eval(ebs):
        """the eval's peak (the optimiser state is cleared before the in-process eval): the longest d 12 item x ebs at K' 0..14, fp32."""
        from boxes import inject_ow_eval as EV
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, 14 if K else 0, dev, ebs, None)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.src == "ow" and not a.args_only and (a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume)):
        mc = []
        for kind, fn in (("train", mem_train), ("eval", mem_eval)):         # micro from the training peak, then eval_bs from the eval peak (separately)
            while True:
                try:
                    pk = fn(a.micro if kind == "train" else a.eval_bs)
                except torch.OutOfMemoryError:
                    pk = float("inf")
                params.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                mc.append({"kind": kind, "micro": a.micro, "eval_bs": a.eval_bs, "peak_gib": pk}); print("MEM", kind, a.micro, a.eval_bs, pk, flush=True)
                if pk <= a.mem_budget or a.mem_check:
                    break
                if kind == "train" and a.micro < a.bs:
                    a.micro *= 2
                elif kind == "eval" and a.eval_bs > 10:
                    a.eval_bs //= 2
                else:
                    break
        json.dump({"checks": mc, "micro": a.micro, "eval_bs": a.eval_bs, "peak_gib_train": [c for c in mc if c["kind"] == "train"][-1]["peak_gib"],
                   "peak_gib_eval": [c for c in mc if c["kind"] == "eval"][-1]["peak_gib"],
                   "train": "the longest bs / micro of 256 d = %d items (SEED_MEMCHECK) at K = %d, + AdamW states" % (a.d_max, K),
                   "eval": "the longest of 256 d = 12 items x eval_bs at K' 0..14 (fp32, no optimiser state)"}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return 0
        assert all(c["peak_gib"] <= a.mem_budget for c in mc[-1:]) and [c for c in mc if c["kind"] == "train"][-1]["peak_gib"] <= a.mem_budget, mc
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "module_sha256": sha_file(os.path.abspath(__file__)),
            "eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow_eval.py")), "nested170_sha256": sha_file(os.path.join(ROOT, "boxes", "nested170.py")),
            "deltamem_sha256": sha_file(os.path.join(ROOT, "boxes", "deltamem.py")), "ow_words_sha256": sha_file(os.path.join(ROOT, "boxes", "ow_words.py")),
            "nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")), "set_build_sec": set_sec, "excl_n": len(EXCL), "d_law": "d = default_rng(seed + 22331).integers(1, d_max + 1) per step",
            "format": {"direct": "dec = [EOT] + enc(' ' + text) + enc(' The answer is:') -> the number token (plain trunk, one pass)",
                       "inject": "think [EOT] + enc(' ' + text) (padded writes masked), K passes; dec as direct, read-only on S_K; aux [EOT, ' Step', ' t', ' :'] + "
                                 "enc(' ' + R_step[t]) + EOT, read-only on S_t, t <= d"}[a.arm]}
    if SETS is not None:
        info.update(SETS["shas"]); info["pair_stats"] = SETS["pair_stats"]
    if a.args_ref:
        ref = json.load(open(a.args_ref)); dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k not in META and ref.get(k) != info.get(k)}
        json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    if a.args_only:
        print("ARGS_ONLY", a.out, flush=True); return 0
    opt = make_opt(params, a.lr); step0 = 1; n_nonfinite = 0; hist = []; wg_hist = []; wwin = []; wlast = []; gn_log = []; aux_hist = []
    expo = {}; binacc = {}; vbin = {}; dseq = []; iseq = []; d2_bins = []; ent = {"buf": [], "entry_d1": None}; fut = {"checked": [], "stop": None}
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng_d.bit_generator.state = ck["rng_d"]; rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        if stream is not None:
            stream.rng.bit_generator.state = ck["stream_rng"]; stream.stats = dict(ck["stream_stats"])
        step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; gn_log = list(ck["gn_log"]); t0 = time.time() - ck["elapsed"]
        expo, binacc, vbin, dseq, iseq, d2_bins, ent, fut, wlast = ck["expo"], ck["binacc"], ck["vbin"], ck["dseq"], ck["iseq"], ck["d2_bins"], ck["ent"], ck["fut"], ck["wlast"]
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif m is not None and K > 0:                                           # note 147-2: g_add calibration at the think hand-offs (nested170's)
        if a.src == "nested":
            its = [tok_nested(it) for it in N7.train_items(np.random.default_rng(17001), 64, a.n_max_train, set())]
        else:
            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, **OWKW); its = [tok_ow(s_.draw(1 + i % a.d_max)) for i in range(64)]
        x, msk = m.think_inputs(its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            m.think(m.mem.zero_state(64, dev), x, msk, min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0, "items": "64 items, d = 1 + i % d_max (SEED_CAL)" if a.src == "ow" else "nested170 17001"},
                                                                    open(os.path.join(a.out, "s228_init.json"), "w", newline="\n"))
    FUT = [int(x_) for x_ in a.futility.split(",")] if a.futility else []
    last = (a.stop_at or a.steps) if not a.time_steps else min(a.stop_at or a.steps, step0 + a.time_steps - 1); tt0 = time.time(); step = step0 - 1
    for step in range(step0, last + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        params.train()
        if m is not None:
            m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
            m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        lam_now[0] = lam_at(step)
        d = int(rng_d.integers(1, a.d_max + 1)) if a.src == "ow" else 0
        its = draw_items(d); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; st_ = {}
        for k_ in range(a.micro):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st_); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if m is not None:
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_log.append(float(gn))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        la_ = st_["la"] / st_["n"]; acc_ = st_["acc"] / st_["n"]; aux_hist.append((la_, st_["aux"] / st_["n"] if "aux" in st_ else None))
        dk = str(its[0]["d"]) if a.src == "ow" else "mix"; expo[dk] = expo.get(dk, 0) + 1
        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0
        for vok, _n in st_.get("vacc", []):
            for t_, v_ in vok.items():
                vb_ = vbin.setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1
        dseq.append(dk if a.src == "ow" else "0"); iseq.append(dk + ":" + ",".join(it["sig"] for it in its))
        if dk == "1" and ent["entry_d1"] is None:                           # entry_d1: the first 100 consecutive d = 1 steps with mean answer CE < 0.5
            ent["buf"].append(la_)
            if len(ent["buf"]) == 100:
                if float(np.mean(ent["buf"])) < 0.5:
                    ent["entry_d1"] = step; log.write(json.dumps({"step": step, "entry_d1": step}) + "\n"); log.flush()
                ent["buf"] = []
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0), "aux_lambda": lam_now[0],
                   "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None}
            if m is not None:
                rec["aux_ce"] = float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None
            hist = []; aux_hist = []
            if m is not None and wwin:
                md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
                rec.update({"gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]})
            rec.update({"gn_preclip": gn_log[-1], "gn_preclip_mean100": float(np.mean(gn_log[-100:])), "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_log[-100:]]))})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if step % 1000 == 0 or step == last:                                # the 1k-bin record (a partial bin at a stop / the end)
            bd = {k_: {"steps": v_[0], "acc": v_[1] / v_[0], "ce": v_[2] / v_[0], "clip": v_[3] / v_[0]} for k_, v_ in sorted(binacc.items())}
            br = {"step": step, "bin": (step - 1) // 1000, "bin_steps": len(dseq), "by_d": bd, "exposure_by_d": dict(sorted(expo.items())), "dseq": "".join(dseq),
                  "ihash": hashlib.sha1("\n".join(iseq).encode()).hexdigest()[:16]}
            if vbin:
                br["aux_vacc_by_pass"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(vbin.items(), key=lambda kv: int(kv[0]))}
            log.write(json.dumps(br) + "\n"); log.flush()
            if "2" in bd and step % 1000 == 0:
                d2_bins.append([step, bd["2"]["acc"]])
            binacc = {}; vbin = {}; dseq = []; iseq = []
        if FUT and step in FUT and fut["stop"] is None:                     # LAT_v1 §4 futility (the d = 2 train acc of complete 1k bins)
            l4 = [v_ for _, v_ in d2_bins[-4:]]; sl = float(np.polyfit(np.arange(len(l4)), l4, 1)[0]) if len(l4) >= 2 else None
            accl = l4[-1] if l4 else None; rising = sl is not None and sl > 0.02
            frec = {"step": step, "d2_last_bin_acc": accl, "slope_last4": sl, "bins": d2_bins[-4:], "rising": rising,
                    "stop": bool(accl is not None and accl < 0.50 and not rising)}
            fut["checked"].append(frec); json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1)
            log.write(json.dumps({"futility": frec}) + "\n"); log.flush()
            if frec["stop"]:
                fut["stop"] = step; json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1); break
        if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.stop_at) and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "rng_d": rng_d.bit_generator.state, "np_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "stream_rng": stream.rng.bit_generator.state if stream is not None else None, "stream_stats": dict(stream.stats) if stream is not None else None,
                        "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist, "gn_log": gn_log, "elapsed": time.time() - t0, "expo": expo, "binacc": binacc,
                        "vbin": vbin, "dseq": dseq, "iseq": iseq, "d2_bins": d2_bins, "ent": ent, "fut": fut, "wlast": wlast}, cp)
    if a.time_steps:
        sps = (time.time() - tt0) / max(1, last - step0 + 1)
        tj = {"arm": a.arm, "think_k": K, "d_max": a.d_max, "steps_timed": last - step0 + 1, "sec_per_step": sps, "est_hours_steps": sps * a.steps / 3600,
              "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "micro": a.micro, "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return 0
    if a.stop_at and step == a.stop_at and a.stop_at < a.steps:
        print("STOP_AT", step, flush=True); return 9
    train_sec = round(time.time() - t0); torch.save(params.state_dict(), os.path.join(a.out, "model.pt"))
    opt.zero_grad(set_to_none=True); opt.state.clear()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    gl = np.array(gn_log) if gn_log else np.array([np.nan])
    train = {"train_sec": train_sec, "last_step": step, "n_nonfinite": n_nonfinite, "entry_d1": ent["entry_d1"], "futility": fut if FUT else None,
             "clip_rate": float((gl > 1.0).mean()), "grad_norm_preclip": {"median": float(np.median(gl)), "p90": float(np.percentile(gl, 90))},
             "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "micro": a.micro, "train_gpu": info["train_gpu"]}
    json.dump(train, open(os.path.join(a.out, "train_summary.json"), "w", newline="\n"), indent=1)
    if a.src == "nested":
        return 0
    from boxes import inject_ow_eval as EV
    EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SETS, train=train)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("DONE", a.out, flush=True); return 0


if __name__ == "__main__":
    sys.exit(main())
