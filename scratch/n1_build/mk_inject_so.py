"""builds boxes/inject_so.py = HEAD (below) + inject_aw.main() copied with explicit edits (each asserted to apply exactly once); THEORY §271 N1 (N1_v1 §10 item 2,
Note 271-1 DIRECT-SI). Run from the repo root: python -B scratch/n1_build/mk_inject_so.py"""
src = open("boxes/inject_aw.py", encoding="utf-8").read()
i0 = src.index("def main(argv=None):"); i1 = src.index('if __name__ == "__main__":')
main = src[i0:i1]

HEAD = '''"""THEORY §271 N1 SHUFFLED-OBJECTS trainer (N1_v1 §2, §3.3-3.5, §10 item 2; Note 271-1 DIRECT-SI). A NEW module: inject_aw / inject_aw_eval / inject_st /
inject_ow / nested170 / deltamem / ow_words / nested_words stay byte-identical (their sha256 recorded in args.json). main() below is inject_aw.main() copied with
listed edits (scratch/n1_build/mk_inject_so.py, each asserted to apply once), so `--src ow` (every new flag at its default) is inject_aw exactly (gate G1); the
model classes / helpers (InjectAW, build_aw, hold_step, cue_inputs, ...) are imported from inject_aw / inject_st / inject_ow unchanged.
  --src so            training stream = so_words.Stream(SEED_TRAIN, EXCL, frames FRAME_TRAIN, ns --so_n, modes --so_modes, offset bool(--so_offset), names NAMES_SO
                      minus --so_names_excl); EXCL = so_words.excl_sigs(so_words.test_sets()) (the sets built once, passed to the eval); m = d per step from the
                      unchanged d law (default_rng(seed + SEED_D).integers(1, d_max + 1)). Items: so_tok.tok_so(it, cue) (no tok_st, F14).
  --cue index/entity/generic  the SO pass cues (so_words.cue_si / cue_en / cue_generic); inject needs --write_src answer (InjectAW, the ':' write).
  VALUE VOCABULARY    VOC = so_tok.VAL_IDS (38 NAMES_SO tokens) under --src so, nested170.NUM_IDS otherwise (the same object: ow bitwise): training CE target
                      VOC[ans] (full-vocabulary CE, as loss_reg), the logged train acc argmax over VOC. The aux CE is unchanged (full vocabulary on the aux tokens;
                      vidx = S_t's name token from tok_so).
  OW-only features    --perm_p / --ask_p / --nk / --distract_p / --distract_mix / --cue_ans / --train_frames / --perm_scope all / --mul_s_p3 / aux_target
                      cumulative: asserted off under --src so; WS-READOUT (OW TEST2) off; --mul_s_p is accepted and inert under --src so (no OW set is built;
                      kept so the args diff vs of_s1 is the registered one).
  --arm direct --src so   DIRECT-OF: one pass, dec = [EOT] + enc(' ' + text) + SFX, CE on VAL_IDS[ans]; --init_trunk (strict, sha asserted) as inject_aw.
  --so_direct_si 1    DIRECT-SI (Note 271-1 MED-1; only with --arm direct --src so --cue index, --aux_step > 0): every training item also contributes its
                      per-swap questions as their own one-pass items: for t = 1..m, input = tok_so's pass-t SI input [EOT] + enc(' ' + text + cue_si(it, t)) + SFX,
                      label S_t (CE full-vocabulary on VAL_IDS[S_t] at ':'). "Same item share and labels" is read as the loop's aux law, term for term:
                      loss = CE(answer) + lambda(step) * mean_items(mean_{t <= m} w_t CE_t), lambda = the loop's lam_at (--aux_step / --aux_fade), w_t = AUX_W[t - 1]
                      under --aux_reweight else 1, items filtered by --aux_d_max as the loop's. So each item carries exactly the m swap questions the loop
                      supervises at passes 1..m (the loop: the aux readout of S_t), with the loop's per-pass weights; the difference is the contrast under test
                      (separate one-pass items; the target is the S_t name token, not the loop's teacher-forced aux sentence). Logged as aux_ce and the 1k-bin
                      aux_vacc_by_pass (full-vocabulary argmax at ':' == S_t). Default 0: every other path unchanged.
  EVAL                --src so: boxes.inject_so_eval.run(out, parts, dev, eval_bs, model, sets = the SO sets, train) (imported at eval time; a missing module
                      raises there, after model.pt / train_summary.json are written); otherwise inject_aw_eval.run as inject_aw.
    python -m boxes.inject_so --src so --arm inject --cue index --write_src answer --think_k 8 --aux_step 1 --aux_reweight --fast_add --cache_lower ..."""
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
import boxes.deltamem as DM  # noqa: E402, F401
import boxes.nested170 as N7  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
import boxes.perm_render as PR  # noqa: E402
import boxes.insert_render as IR  # noqa: E402
import boxes.so_words as SO  # noqa: E402
from boxes.nested170 import AUX_W, EOT, NUM_IDS, pad  # noqa: E402, F401
from boxes.inject_ow import (ENC, SFX, SEED_D, SEED_CAL, SEED_MEMCHECK, OWKW, META, tok_ow, tok_nested, direct_logits, sha_file, build_sets,  # noqa: E402, F401
                             InjectOW, load_trunk)
from boxes.inject_st import CUE_FMT, cue_str, tok_st, cue_inputs, st_len  # noqa: E402, F401
from boxes.inject_aw import (COLON, AW_FMT, InjectAW, tok_aw, tok_ask, build_aw, reg_excl, ws_readout, state_n, hold_step, hold_acc, hold_mean,  # noqa: E402, F401
                             SEED_ASK_STREAM, SEED_ASK_COIN, SEED_ASK_PERM)
from boxes.so_tok import tok_so, VAL_IDS, val_index  # noqa: E402

SO_CUE = {"index": "index", "entity": "entity", "generic": "generic"}       # --cue -> tok_so's cue


def si_weight(it, t):
    """Note 271-5: DIRECT-SI's per-(item, t) CE weight = 1 / n_tok, n_tok = the loop's aux target length at t (len(aux[t - 1]) = (y_a != -100).sum for that row in
    aux_logits, which pads with -100 only past the row): the name token gets the loop's per-token share of CE_t."""
    return 1.0 / len(it["aux"][t - 1])


def so_format(a):
    """args.json 'format' under --src so."""
    cue = {"index": "SI ' After swap <L(min(t, m))>, who <verb> <o*>?'", "entity": "EN ' After <a_t> and <b_t> trade, who <verb> <o*>?'",
           "generic": "GENERIC ' Who <verb> <o*> now?'"}[a.cue]
    if a.arm == "direct":
        return ("§271 DIRECT: dec = [EOT] + enc(' ' + text) + enc(' The answer is:') -> CE on VAL_IDS[S_m] (plain trunk, one pass)"
                + (" | Note 271-1 DIRECT-SI: + per item, for t <= m, the one-pass item [EOT] + enc(' ' + text + SI cue(t)) + SFX -> CE on VAL_IDS[S_t]; "
                   "lambda(step) * mean_items mean_t (AUX_W[t - 1] if aux_reweight) CE_t" if a.so_direct_si else ""))
    return ("§271 INJECT (src so): pass t input = [EOT] + enc(' ' + text + CUE(min(t, m))) + enc(' The answer is:'), CUE = " + cue + "; only the ':' state writes; "
            "K passes; dec = [EOT] + enc(' ' + text) + SFX read-only on S_K -> VAL_IDS[S_m]; aux [EOT, ' Step', ' t', ' :'] + enc(' ' + R_step[t]) + EOT, "
            "R_step[t] = 'After swap <L(t)>, <S_t> <verb> <o*>.', read-only on S_t, t <= m")
'''

E = [
# flags
('ap.add_argument("--src", default="ow", choices=["ow", "nested"], help="nested = GATE ONLY (port-kernel identity vs nested170)")',
 'ap.add_argument("--src", default="ow", choices=["ow", "nested", "so"], help="nested = GATE ONLY (port-kernel identity vs nested170); so = §271 N1 SHUFFLED-OBJECTS")'),
('choices=["none", "step", "all", "generic", "target"], help="§233: step',
 'choices=["none", "step", "all", "generic", "target", "index", "entity"], help="§271: index / entity / generic = the SO cues (src so); §233: step'),
('                    "(unchanged); all = every statement over all slots (seeds 27000 / 27001; same coins), question last")\n',
 '                    "(unchanged); all = every statement over all slots (seeds 27000 / 27001; same coins), question last")\n'
 '    ap.add_argument("--so_offset", type=int, default=1, choices=[0, 1], help="§271 F3: training NUM labels o+1..o+m, o ~ U{0..12-m} (1) or o = 0 (src so only)")\n'
 '    ap.add_argument("--so_n", default="3,4,5", help="§271: the training n set (src so only)"); ap.add_argument("--so_modes", default="num,ord", help="§271: the training modes (src so only)")\n'
 '    ap.add_argument("--so_names_excl", default="Eve,Bob", help="§271 F12a: names removed from NAMES_SO for the training stream (src so only)")\n'
 '    ap.add_argument("--so_direct_si", type=int, default=0, choices=[0, 1], help="Note 271-1 DIRECT-SI: + the per-swap SI questions as one-pass items (direct, src so; 0 = off)")\n'),
('a.task = "ow_topo" if a.src == "ow" else "nested"', 'a.task = {"ow": "ow_topo", "so": "so"}.get(a.src, "nested")'),
# asserts
('assert not (a.think_k or a.aux_step or a.aux_reweight or a.cache_lower',
 'assert not (a.think_k or (a.aux_step and not a.so_direct_si) or (a.aux_reweight and not a.so_direct_si) or a.cache_lower'),
('assert a.src == "ow" or a.d_max == 8, ', 'assert a.src != "nested" or a.d_max == 8, '),
('(a.arm == "inject" and not a.ckpt_passes and a.src == "ow"), "--cue step / generic: LAT-ST',
 '(a.arm == "inject" and not a.ckpt_passes and (a.src == "ow" or (a.src == "so" and a.cue == "generic"))), "--cue step / generic: LAT-ST'),
('assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic", "target")), ',
 'assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic", "target", "index", "entity")), '),
('    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"\n',
 '    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"\n'
 '    SOCUE = SO_CUE.get(a.cue); SI = bool(a.so_direct_si)                    # §271: tok_so\'s cue; Note 271-1 DIRECT-SI\n'
 '    assert a.src != "so" or (SOCUE is not None and ((a.arm == "inject" and a.write_src == "answer" and not a.ckpt_passes) or (a.arm == "direct" and a.cue == "index"))), \\\n'
 '        "§271 --src so: --cue index / entity / generic (inject: --write_src answer, no ckpt_passes; direct: --cue index)"\n'
 '    assert a.cue not in ("index", "entity") or a.src == "so", "--cue index / entity: --src so only"\n'
 '    assert a.src != "so" or not (a.perm_p or a.ask_p or a.nk or a.distract_p or a.distract_mix or a.cue_ans or a.train_frames or a.perm_scope != "op" or a.mul_s_p3 != 1.0\n'
 '                                 or a.aux_target != "value"), "§271 --src so: the OW-only features are off"\n'
 '    assert not SI or (a.src == "so" and a.arm == "direct" and a.cue == "index" and a.aux_step > 0), "Note 271-1 --so_direct_si: --arm direct --src so --cue index, --aux_step > 0"\n'),
('assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src == "ow"), ', 'assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src in ("ow", "so")), '),
('a.cue in ("step", "generic", "target") and not a.ckpt_passes and a.src == "ow" and a.think_k >= 3), "--hold_*',
 'a.cue in ("step", "generic", "target", "index", "entity") and not a.ckpt_passes and a.src in ("ow", "so") and a.think_k >= 3), "--hold_*'),
('    K = a.think_k if a.arm == "inject" else 0\n',
 '    K = a.think_k if a.arm == "inject" else 0\n'
 '    VOC = VAL_IDS if a.src == "so" else NUM_IDS                            # §271: the value vocabulary (ow: the number ids, the same object)\n'),
# sets / stream
('    t_sets = time.time()\n', '    t_sets = time.time(); SO_SETS = None\n'),
('    else:                                                                   # gate only: nested170\'s TE / SPOOL exclusion and item law\n',
 '    elif a.src == "so":                                                     # §271 N1 §3.5: TEST / TRANSFER / OFFSET built once; EXCL = their sigs\n'
 '        SO_SETS = SO.test_sets(); EXCL = SO.excl_sigs(SO_SETS); SETS = None; SO_EXCL_NAMES = set(x for x in a.so_names_excl.split(",") if x)\n'
 '        SO_NAMES = [x for x in SO.NAMES_SO if x not in SO_EXCL_NAMES]; SO_NS = tuple(int(x) for x in a.so_n.split(",")); SO_MODES = tuple(a.so_modes.split(","))\n'
 '        assert SO_NAMES and set(SO_NS) <= set(SO.NS) and set(SO_MODES) <= set(SO.MODES) and not ({"Eve", "Bob"} & set(SO_NAMES)), ("§271 --so_*", SO_NS, SO_MODES)\n'
 '        stream = SO.Stream(SO.SEED_TRAIN, EXCL, frames=SO.FRAME_TRAIN, ns=SO_NS, modes=SO_MODES, offset=bool(a.so_offset), names=SO_NAMES)\n'
 '    else:                                                                   # gate only: nested170\'s TE / SPOOL exclusion and item law\n'),
# VOC at every value-vocabulary site
('NUM_IDS[[it["ans"] for it in items]]', 'VOC[[it["ans"] for it in items]]'),
('NUM_IDS[[items[i]["ans"] for i in cidx]]', 'VOC[[items[i]["ans"] for i in cidx]]'),
('[:, NUM_IDS].argmax(-1).tolist()', '[:, VOC].argmax(-1).tolist()'),
('lg.detach()[:, NUM_IDS]', 'lg.detach()[:, VOC]'),
('NUM_IDS[[it["ask_v"] for it in items]]', 'VOC[[it["ask_v"] for it in items]]'),
# items
('        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs - NASK)]\n',
 '        if a.src == "so":                                                   # §271: so_tok items (NASK = 0)\n'
 '            return [tok_so(stream.draw(d), SOCUE) for _ in range(a.bs)]\n'
 '        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs - NASK)]\n'),
# DIRECT-SI
('    def loss_reg(items, stat=None):\n',
 '    def si_aux(items, stat, zero):\n'
 '        """Note 271-1 DIRECT-SI: per item i and t <= m (aux_d_max filter as the loop\'s), the one-pass item think_cues[t - 1] = [EOT] + enc(\' \' + text + cue_si(it, t))\n'
 '        + SFX -> CE(the \':\' logits, VAL_IDS[S_t]); -> mean_items(mean_t w_t CE_t / n_tok(i, t)), w_t = AUX_W[t - 1] under --aux_reweight (the loop\'s aux law), n_tok = si_weight (Note 271-5); stat: aux / vacc."""\n'
 '        pairs = [(i, t) for i, it in enumerate(items) for t in range(1, len(it["think_cues"]) + 1) if not a.aux_d_max or it["d"] <= a.aux_d_max]\n'
 '        assert all(it["cue"] == "index" and len(it["think_cues"]) == it["d"] for it in items), "DIRECT-SI: SI cues, one per swap"\n'
 '        if not pairs:\n'
 '            return zero\n'
 '        y = torch.tensor(VOC[[val_index(items[i]["raw"]["values"][t]) for i, t in pairs]], device=dev)\n'
 '        lg_s = direct_logits(g, [{"dec": items[i]["think_cues"][t - 1]} for i, t in pairs], dev, ac); ce = Fn.cross_entropy(lg_s, y, reduction="none"); per = [[] for _ in items]\n'
 '        for r_, (i, t) in enumerate(pairs):\n'
 '            w_ = si_weight(items[i], t); per[i].append(ce[r_] * w_ * AUX_W[t - 1] if a.aux_reweight else ce[r_] * w_)   # Note 271-5: x 1 / n_tok\n'
 '        aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()\n'
 '        if stat is not None:\n'
 '            ok_ = (lg_s.detach().argmax(-1) == y).float(); vok = {t: float(ok_[[r_ for r_, (_, t_) in enumerate(pairs) if t_ == t]].mean()) for t in sorted({t for _, t in pairs})}\n'
 '            stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))\n'
 '            stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))\n'
 '        return aux\n\n'
 '    def loss_reg(items, stat=None):\n'),
('            lg = direct_logits(g, items, dev, ac); la = Fn.cross_entropy(lg, tgt); loss = la\n',
 '            lg = direct_logits(g, items, dev, ac); la = Fn.cross_entropy(lg, tgt); loss = la\n'
 '            if SI:                                                          # Note 271-1 DIRECT-SI: + lambda * the per-swap one-pass CE\n'
 '                loss = la + lam_now[0] * si_aux(items, stat, la * 0.0)\n'),
('            if m is not None:\n                rec["aux_ce"] =', '            if m is not None or SI:\n                rec["aux_ce"] ='),
# memory check / WS-READOUT
('for d_ in range(1, a.d_max + 1)} if a.write_src == "answer" else None   # §234 WS-READOUT',
 'for d_ in range(1, a.d_max + 1)} if (a.write_src == "answer" and a.src == "ow") else None   # §234 WS-READOUT'),
('    def mem_train(micro):\n',
 '    if a.src == "so":                                                       # §271: memory-check items (train: n 5, m d_max; eval: TEST n 5 m 12\'s longest)\n'
 '        s_mcs = SO.Stream(SEED_MEMCHECK, frames=SO.FRAME_TRAIN, ns=(5,), offset=bool(a.so_offset))\n'
 '        MC_TR = sorted([tok_so(s_mcs.draw(a.d_max), SOCUE) for _ in range(256)], key=lambda it: -st_len(it))\n'
 '        MC_EV = sorted([tok_so(it, SOCUE) for md_ in SO.MODES for it in SO_SETS["test"][(5, 12, md_)]], key=lambda it: -st_len(it))[:1]\n\n'
 '    def mem_train(micro):\n'),
('MC_EV * ebs, 14 if K else 0, dev, ebs, None)', 'MC_EV * ebs, (64 if a.src == "so" else 14) if K else 0, dev, ebs, None)'),
('    if a.src == "ow" and not a.args_only and (a.mem_check', '    if a.src in ("ow", "so") and not a.args_only and (a.mem_check'),
('(fp32, no optimiser state)"}',
 '(fp32, no optimiser state)", **({"so": "src so: train = the longest bs / micro of 256 n 5 m d_max items; eval = TEST n 5 m 12\'s longest x eval_bs at K\' 0..64"}\n'
 '                                                                         if a.src == "so" else {})}'),
# info
('"nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")),',
 '"nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")), "inject_aw_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw.py")),\n'
 '            "inject_aw_eval_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw_eval.py")), "so_words_sha256": sha_file(os.path.join(ROOT, "boxes", "so_words.py")),\n'
 '            "so_tok_sha256": sha_file(os.path.join(ROOT, "boxes", "so_tok.py")),\n'
 '            "so_eval_module_sha256": (sha_file(os.path.join(ROOT, "boxes", "inject_so_eval.py")) if os.path.exists(os.path.join(ROOT, "boxes", "inject_so_eval.py")) else None) if a.src == "so" else None,'),
('"format": {"direct": "dec = [EOT]', '"format": so_format(a) if a.src == "so" else {"direct": "dec = [EOT]'),
('        info.update(SETS["shas"]); info["pair_stats"] = SETS["pair_stats"]\n',
 '        info.update(SETS["shas"]); info["pair_stats"] = SETS["pair_stats"]\n'
 '    if SO_SETS is not None:                                                 # §271: the SO set shas (TEST per cell, TRANSFER per m, OFFSET) + the training names\n'
 '        info["so_set_shas"] = {**{"test|%d|%d|%s" % k_: SO.set_sha(v_) for k_, v_ in sorted(SO_SETS["test"].items())},\n'
 '                               **{"transfer|%d" % k_: SO.set_sha(v_) for k_, v_ in sorted(SO_SETS["transfer"].items())}, "offset": SO.set_sha(SO_SETS["offset"])}\n'
 '        info["so_train_names"] = SO_NAMES\n'),
# calibration
('        else:\n            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its = [tok_aw(',
 '        elif a.src == "so":                                                 # §271: the SO training law\'s items\n'
 '            s_ = SO.Stream(SEED_CAL, frames=SO.FRAME_TRAIN, ns=SO_NS, modes=SO_MODES, offset=bool(a.so_offset), names=SO_NAMES); its = [tok_so(s_.draw(1 + i % a.d_max), SOCUE) for i in range(64)]\n'
 '        else:\n            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its = [tok_aw('),
('(SEED_CAL)" if a.src == "ow" else "nested170 17001"', '(SEED_CAL)" if a.src in ("ow", "so") else "nested170 17001"'),
# loop: d law / bins
('d = int(rng_d.integers(1, a.d_max + 1)) if a.src == "ow" else 0', 'd = int(rng_d.integers(1, a.d_max + 1)) if a.src in ("ow", "so") else 0'),
('dk = str(its[0]["d"]) if a.src == "ow" else "mix"', 'dk = str(its[0]["d"]) if a.src in ("ow", "so") else "mix"'),
('dseq.append(dk if a.src == "ow" else "0")', 'dseq.append(dk if a.src in ("ow", "so") else "0")'),
# eval
('    from boxes import inject_aw_eval as EV\n    EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SETS, train=train)\n',
 '    if a.src == "so":                                                       # §271: the SO eval, imported here (a missing module fails loudly at eval time)\n'
 '        try:\n'
 '            from boxes import inject_so_eval as EV\n'
 '        except ImportError as e_:\n'
 '            raise RuntimeError("§271: boxes/inject_so_eval.py missing / broken; model.pt and train_summary.json are written, run the SO eval separately") from e_\n'
 '        EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SO_SETS, train=train)\n'
 '    else:\n'
 '        from boxes import inject_aw_eval as EV\n'
 '        EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SETS, train=train)\n'),
]
for o, n in E:
    assert main.count(o) == 1, ("edit not applied exactly once", main.count(o), o[:100]); main = main.replace(o, n)
assert main.count("NUM_IDS") == 1, "NUM_IDS left in main (only VOC's definition may name it)"
open("boxes/inject_so.py", "w", encoding="utf-8", newline="\n").write(HEAD + "\n\n" + main + 'if __name__ == "__main__":\n    sys.exit(main())\n')
print("written boxes/inject_so.py,", len(E), "edits applied once each")
