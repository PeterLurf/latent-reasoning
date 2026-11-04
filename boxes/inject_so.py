"""THEORY §271 N1 SHUFFLED-OBJECTS trainer (N1_v1 §2, §3.3-3.5, §10 item 2; Note 271-1 DIRECT-SI). A NEW module: inject_aw / inject_aw_eval / inject_st /
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["direct", "inject"]); ap.add_argument("--think_k", type=int, default=0); ap.add_argument("--d_max", type=int, default=8)
    ap.add_argument("--base", default="gpt2", choices=["gpt2", "gpt2-medium"], help="§258 S0: the pretrained trunk; DeltaMem at layer L // 2 with D / 128 + D / 128 heads of 64 (small: 6, 6 + 6)")
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true"); ap.add_argument("--mem_write", default="top", choices=["top"])
    ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000); ap.add_argument("--cache_lower", action="store_true")
    ap.add_argument("--aux_step", type=float, default=0.0); ap.add_argument("--aux_fade", default="48000,48000"); ap.add_argument("--aux_reweight", action="store_true")
    ap.add_argument("--aux_target", default="value", choices=["value", "cumulative"], help="§229: cumulative = the re-rooted program R_t per pass (default value = §228)")
    ap.add_argument("--fast_add", action="store_true"); ap.add_argument("--fp32", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes / gates only")
    ap.add_argument("--src", default="ow", choices=["ow", "nested", "so"], help="nested = GATE ONLY (port-kernel identity vs nested170); so = §271 N1 SHUFFLED-OBJECTS"); ap.add_argument("--n_max_train", type=int, default=8, help="--src nested only")
    ap.add_argument("--futility", default="", help="LAT_v1 §4: 'step1,step2' (24000,32000): stop if the d = 2 train acc of the last 1k bin < 0.50 and the OLS slope of the "
                    "last 4 d = 2 bins <= +0.02 per bin (a still-rising first check defers to the second); '' = off")
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--stop_at", type=int, default=0, help="train to this step under the --steps schedule (ckpt at the stop step), then exit 9 (no eval)")
    ap.add_argument("--time_steps", type=int, default=0); ap.add_argument("--args_only", action="store_true"); ap.add_argument("--args_ref", default="")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--eval_parts", default="acc"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--distract_mix", default="", help="§257 DC2_v1: insert types drawn uniformly per insert (s256,chain,givop) under --distract_p ('' = §256 rule)")
    ap.add_argument("--distract_p", type=float, default=0.0, help="Note 255-5: P(an item gets U{0,1,2} irrelevant new-name op sentences) (0 = off, unchanged)")
    ap.add_argument("--distract_nmax", type=int, default=2, help="§265: inserts n ~ U{0..nmax} (default 2 = the pre-§265 draw, bitwise); capped at the 40-name budget")
    ap.add_argument("--perm_p", type=float, default=0.0, help="§252: P(a d >= 2 training item is re-rendered dependency-violating) (0 = off, unchanged)")
    ap.add_argument("--cue", default="none", choices=["none", "step", "all", "generic", "target", "index", "entity"], help="§271: index / entity / generic = the SO cues (src so); §233: step = LAT-ST (the S_min(t, d) cue at pass t), all = CUED-DIRECT (cues S_1..S_d on the decoder); none = inject_ow")
    ap.add_argument("--write_src", default="all", choices=["all", "answer"], help="§234: answer = + ' The answer is:' on every pass input, only the ':' state writes")
    ap.add_argument("--init_trunk", default="", help="§234: GPT-2 loaded strict from this file before DeltaMem is built"); ap.add_argument("--init_sha", default="", help="its sha256 prefix (asserted)")
    ap.add_argument("--mul_s_p3", type=float, default=1.0, help="§235: P(forced mul_s | d 3) on the TRAINING stream only (1.0 = unchanged)")
    ap.add_argument("--mul_s_p", type=int, default=1, choices=[0, 1, 2], help="§236: 0 = REG law (no forced mul_s); §238: 2 = DF (per-step mul_s, q_t); 1 = unchanged - on every stream / set (new eval seeds)")
    ap.add_argument("--cue_ans", type=float, default=0.0, help="Note 251-10: + λ CE(full-vocab LM logits at pass d's ':', the answer token) (0 = off, unchanged)")
    ap.add_argument("--aux_d_max", type=int, default=0, help="§251: the aux CE only on items with d <= aux_d_max (0 = every item, unchanged)")
    ap.add_argument("--train_frames", default="", help="§250: the training stream's frames, e.g. 0-5,8-11 ('' = OW.FRAME_TRAIN, unchanged)")
    ap.add_argument("--keep_final_ckpt", action="store_true", help="§244: keep the final ckpt_last.pt as ckpt_final.pt (fetched by the drain) instead of deleting it")
    ap.add_argument("--hold_inc", type=float, default=0.0, help="§240: lambda of L_inc (HOLD-INC; 0 = off)"); ap.add_argument("--hold_start", type=int, default=500)
    ap.add_argument("--hold_far_n", type=int, default=16); ap.add_argument("--hold_far_lo", type=int, default=16); ap.add_argument("--hold_far_hi", type=int, default=40)
    ap.add_argument("--hold_log", action="store_true", help="§240: the hold stats + gn_plain without the penalty (CTRL arms); implied by --hold_inc > 0")
    ap.add_argument("--init_model", default="", help="§237: the full params state (trunk + DeltaMem + W_o + gains) loaded strict after the build; calibration skipped")
    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")
    ap.add_argument("--nk", type=int, default=0, choices=[0, 1], help="§262 NK_v1: the name-keyed head packed into S (boxes/nk.py; 0 = off, unchanged)")
    ap.add_argument("--ask_p", type=float, default=0.0, help="§262 Note 262-2: the share of each batch that is one-hop ask items (':' CE on v_X at every pass; 0 = off)")
    ap.add_argument("--ask_perm_p", type=float, default=0.5, help="§262: P(an ask item with d >= 2 is permuted before its class)")
    ap.add_argument("--perm_scope", choices=("op", "all"), default="op", help="§270 FP_v1: what --perm_p / --ask_p permute: op = the op sentences over their slots "
                    "(unchanged); all = every statement over all slots (seeds 27000 / 27001; same coins), question last")
    ap.add_argument("--so_offset", type=int, default=1, choices=[0, 1], help="§271 F3: training NUM labels o+1..o+m, o ~ U{0..12-m} (1) or o = 0 (src so only)")
    ap.add_argument("--so_n", default="3,4,5", help="§271: the training n set (src so only)"); ap.add_argument("--so_modes", default="num,ord", help="§271: the training modes (src so only)")
    ap.add_argument("--so_names_excl", default="Eve,Bob", help="§271 F12a: names removed from NAMES_SO for the training stream (src so only)")
    ap.add_argument("--so_direct_si", type=int, default=0, choices=[0, 1], help="Note 271-1 DIRECT-SI: + the per-swap SI questions as one-pass items (direct, src so; 0 = off)")
    a = ap.parse_args(argv); a.task = {"ow": "ow_topo", "so": "so"}.get(a.src, "nested")
    import boxes.deltamem as DM_
    from boxes.fsq120 import make_opt, lr_at, driver_version, env_info
    from boxes.deltas5 import write_grad_norm
    DM_.FAST_ADD = bool(a.fast_add); os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale
    if a.arm == "direct":
        assert not (a.think_k or (a.aux_step and not a.so_direct_si) or (a.aux_reweight and not a.so_direct_si) or a.cache_lower or a.fast_add or a.futility or a.aux_target != "value"), "DIRECT has no latent flags"
        a.mem_write = a.write_mix_start = a.write_mix_end = None
    else:
        assert a.think_k >= 1 and a.aux_step > 0, "§228 inject: K >= 1 with the step aux"
    assert a.src != "nested" or a.d_max == 8, "--src nested: n ~ U{1..n_max_train} per item (nested170's law)"
    assert a.cue not in ("step", "generic", "target") or (a.arm == "inject" and not a.ckpt_passes and (a.src == "ow" or (a.src == "so" and a.cue == "generic"))), "--cue step / generic: LAT-ST (inject, OW, no ckpt_passes)"
    assert not a.perm_p or (a.src == "ow" and a.arm == "inject" and a.write_src == "answer" and not a.ckpt_passes), "--perm_p: LAT-AW (inject, OW, write_src answer)"
    assert a.cue != "all" or (a.arm == "direct" and a.src == "ow"), "--cue all: CUED-DIRECT (direct, OW)"
    assert a.write_src != "answer" or (a.arm == "inject" and a.cue in ("step", "generic", "target", "index", "entity")), "--write_src answer: LAT-AW (inject, --cue step / generic)"
    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"
    SOCUE = SO_CUE.get(a.cue); SI = bool(a.so_direct_si)                    # §271: tok_so's cue; Note 271-1 DIRECT-SI
    assert a.src != "so" or (SOCUE is not None and ((a.arm == "inject" and a.write_src == "answer" and not a.ckpt_passes) or (a.arm == "direct" and a.cue == "index"))), \
        "§271 --src so: --cue index / entity / generic (inject: --write_src answer, no ckpt_passes; direct: --cue index)"
    assert a.cue not in ("index", "entity") or a.src == "so", "--cue index / entity: --src so only"
    assert a.src != "so" or not (a.perm_p or a.ask_p or a.nk or a.distract_p or a.distract_mix or a.cue_ans or a.train_frames or a.perm_scope != "op" or a.mul_s_p3 != 1.0
                                 or a.aux_target != "value"), "§271 --src so: the OW-only features are off"
    assert not SI or (a.src == "so" and a.arm == "direct" and a.cue == "index" and a.aux_step > 0), "Note 271-1 --so_direct_si: --arm direct --src so --cue index, --aux_step > 0"
    NASK = int(round(a.ask_p * a.bs))                                      # §262: the ask items per step (last in the batch)
    assert not (a.nk or a.ask_p) or (a.arm == "inject" and a.cue == "step" and a.write_src == "answer" and a.src == "ow" and not a.ckpt_passes and not a.cue_ans
                                     and NASK < a.bs and a.bs - NASK >= a.hold_far_n), "§262 --nk / --ask_p: LAT-AW (inject, cue step, write_src answer); ask items < bs"
    assert 1 <= a.d_max <= 8 and a.bs % a.micro == 0
    assert a.perm_scope == "op" or a.perm_p or a.ask_p, "§270 --perm_scope all needs --perm_p or --ask_p"
    PALL = a.perm_scope == "all"
    K = a.think_k if a.arm == "inject" else 0
    VOC = VAL_IDS if a.src == "so" else NUM_IDS                            # §271: the value vocabulary (ow: the number ids, the same object)
    torch.manual_seed(a.seed); rng_d = np.random.default_rng(a.seed + SEED_D); rng = np.random.default_rng(a.seed)
    rng_far = np.random.default_rng(a.seed + 24800)                         # §240: T_far ~ U{hold_far_lo..hold_far_hi}, one draw per step (HOLD only)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    t_sets = time.time(); SO_SETS = None
    if a.src == "ow":
        SETS = build_sets(mul_s_p=a.mul_s_p); EXCL = SETS["excl"] | (reg_excl(a, SETS["excl"]) if a.mul_s_p != 1 else set())   # §236 / §238: + the eval-own sets of the run law
        TRAIN_FR = tuple(x for p_ in a.train_frames.split(",") for x in (range(int(p_.split("-")[0]), int(p_.split("-")[1]) + 1) if "-" in p_ else (int(p_),))) if a.train_frames else OW.FRAME_TRAIN
        assert all(f_ in range(len(OW.FRAMES["add_c"])) for f_ in TRAIN_FR) and len(set(TRAIN_FR)) == len(TRAIN_FR), ("§250 --train_frames", TRAIN_FR)
        stream = OW.Stream(OW.SEED_TRAIN, EXCL, frames=TRAIN_FR, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW)   # §235 / §236: training stream; §250: --train_frames
    elif a.src == "so":                                                     # §271 N1 §3.5: TEST / TRANSFER / OFFSET built once; EXCL = their sigs
        SO_SETS = SO.test_sets(); EXCL = SO.excl_sigs(SO_SETS); SETS = None; SO_EXCL_NAMES = set(x for x in a.so_names_excl.split(",") if x)
        SO_NAMES = [x for x in SO.NAMES_SO if x not in SO_EXCL_NAMES]; SO_NS = tuple(int(x) for x in a.so_n.split(",")); SO_MODES = tuple(a.so_modes.split(","))
        assert SO_NAMES and set(SO_NS) <= set(SO.NS) and set(SO_MODES) <= set(SO.MODES) and not ({"Eve", "Bob"} & set(SO_NAMES)), ("§271 --so_*", SO_NS, SO_MODES)
        stream = SO.Stream(SO.SEED_TRAIN, EXCL, frames=SO.FRAME_TRAIN, ns=SO_NS, modes=SO_MODES, offset=bool(a.so_offset), names=SO_NAMES)
    else:                                                                   # gate only: nested170's TE / SPOOL exclusion and item law
        TE = N7.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
        for _ in range(2200):
            r__, _t = N7.generate(r_sp, 8); EXCL.add(tuple(N7.render(r__)))
        SETS = None; stream = None
    set_sec = round(time.time() - t_sets)
    assert a.mul_s_p3 == 1.0 or a.src == "ow", "--mul_s_p3: OW only"
    assert a.mul_s_p == 1 or (a.mul_s_p3 == 1.0 and a.src in ("ow", "so")), "--mul_s_p 0: OW only, mul_s_p3 must be 1"
    HOLD = a.hold_inc > 0 or a.hold_log                                     # §240: keep the detached states (penalty and / or the hold stats)
    assert not HOLD or (a.arm == "inject" and a.cue in ("step", "generic", "target", "index", "entity") and not a.ckpt_passes and a.src in ("ow", "so") and a.think_k >= 3), "--hold_*: LAT (inject, cue step)"
    m, g, params = build_aw(a, dev)
    assert bool(a.init_model) == bool(a.init_model_sha), "--init_model needs --init_model_sha"
    if a.init_model:                                                        # §237: the full model state, strict; the sha logged (init_model_sha256)
        sha_m = hashlib.sha256(open(a.init_model, "rb").read()).hexdigest(); assert sha_m.startswith(a.init_model_sha), ("init_model sha", sha_m, a.init_model_sha)
        mk_ = params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=not a.nk); a.init_model_sha256 = sha_m
        assert not a.nk or (not mk_.unexpected_keys and sorted(mk_.missing_keys) == ["nk.W_n.weight", "nk.W_on.weight", "nk.W_vn.weight"]), ("§262 NK init: only nk.* new", mk_)
    f0_, f1_ = [float(x_) for x_ in a.aux_fade.split(",")]

    def lam_at(step):
        if not a.aux_step:
            return 0.0
        return a.aux_step if step <= f0_ * sc else a.aux_step * max(0.0, (f1_ * sc - step) / max(1.0, (f1_ - f0_) * sc))
    lam_now = [lam_at(1)]

    # §262 NK_v1 §2: --distract_p with --perm_p = perm-then-insert (§252's coin / permutation first, then the §257 inserts placed by the S labels)
    INS_TEXTS = ({it["text"] for its_ in IR.insert_eval_sets(SETS["test2"][4])[0].values() for it in its_}
                 | {it["text"] for d_ in IR.held_out_sets(SETS["test2"][4])[0].values() for its_ in d_.values() for it in its_}
                 | {it["text"] for d_ in IR.held_out2_sets(SETS["test2"][4])[0].values() for its_ in d_.values() for it in its_}) if (a.distract_p and a.d_max >= 4) else set()   # Note 255-5 (b) + §256 held-out sets
    DSTAT = {}                                                              # Note 255-5: {n: count} in the open 1k bin
    DCAP = {}                                                               # §265: {drawn n: count} of the name-budget-capped draws in the open 1k bin
    MIXT = tuple(a.distract_mix.split(",")) if a.distract_mix else (); DTYPE = {}   # §257: the insert types; {type: count} in the open 1k bin
    assert not MIXT or (a.distract_p > 0 and set(MIXT) <= set(IR.MIX_TYPES)), ("--distract_mix needs --distract_p and known types", MIXT)
    TP_TEXTS = {it["text"] for d_, its_ in PR.test_perm_sets({d_: SETS["test2"][d_] for d_ in range(1, a.d_max + 1)}, range(1, a.d_max + 1))[0].items()
                for it in its_} if (a.perm_p or a.ask_p) else set()                     # §252 Note 252-1: the TEST-PERM texts (622fb717's family) for the disjointness assert
    if PALL:                                                                # §270 FP_v1 §2: + the full_perm / shuffle_graded texts; no base item shares a TEST2 d 1-8 statement multiset
        import boxes.fp_parts as FPP
        FP_TEXTS = FPP.fp_texts(SETS["test2"]) | FPP.sg_texts(SETS["test2"]); T2_MSIG = {FPP.msig(it) for d_ in range(1, 9) for it in SETS["test2"][d_]}

    MSREJ = [0]                                                             # §270 Note 270-1 (1): cumulative multiset rejects (since process start; not in the resume ckpt)

    def sdraw(st_, d):
        """stream draw; under scope all a base item whose statement multiset equals a TEST2 d 1-8 item's is rejected and the next item drawn (op: one plain draw)."""
        for _ in range(1000):                                               # Audit LOW (c): a retry cap (op: the first draw returns)
            r_ = st_.draw(d)
            if not PALL or FPP.msig(r_) not in T2_MSIG:
                return r_
            MSREJ[0] += 1
        raise AssertionError("§270 sdraw: 1000 consecutive multiset rejects")

    def pall_check(base, final_texts):
        """§270 scope all: the base items' multisets are not TEST2's (sdraw resampled them); the final texts are not full_perm / shuffle_graded texts."""
        assert not any(FPP.msig(r_) in T2_MSIG for r_ in base), "§270: a base item shares its statement multiset with a TEST2 d 1-8 item"
        assert not any(t_ in FP_TEXTS for t_ in final_texts), "§270: a training text equals a full_perm / shuffle_graded eval text"

    def draw_items(d):
        if a.src == "nested":
            return [tok_nested(it) for it in N7.train_items(rng, a.bs, a.n_max_train, EXCL)]
        if a.distract_p:                                                    # Note 255-5 --distract_p: per-item seeded coin / n / draw; targets, solve, disjointness asserted
            raws = [sdraw(stream, d) for _ in range(a.bs - NASK)]; out_ = []
            for k_, r_ in enumerate(raws):
                n_ = 0
                if a.perm_p and d >= 2 and np.random.default_rng((a.seed, PR.SEED_PERM_TRAIN + 1, step, k_)).random() < a.perm_p:   # §262 perm-then-insert: §252's coin / perm
                    r_ = PR.train_perm_all(r_, a.seed, step, k_) if PALL else PR.train_perm(r_, a.seed, step, k_)   # §270: scope all, the same coin
                if np.random.default_rng((a.seed, IR.SEED_DISTRACT_COIN, step, k_)).random() < a.distract_p:
                    r2_, n_ = IR.distract_mix(r_, a.seed, step, k_, MIXT, nmax=a.distract_nmax) if MIXT else IR.distract(r_, a.seed, step, k_, nmax=a.distract_nmax)   # §257 --distract_mix; §265 nmax
                    if n_:
                        t_new, t_old = tok_aw(r2_, a.aux_target, a.cue, a.write_src), tok_aw(r_, a.aux_target, a.cue, a.write_src)
                        assert t_new["aux"] == t_old["aux"] and t_new["ans"] == t_old["ans"] and len(t_new["think_cues"]) == len(t_old["think_cues"]) and all(
                            q1[len(t_new["think"]):] == q0[len(t_old["think"]):] for q1, q0 in zip(t_new["think_cues"], t_old["think_cues"])), "DISTRACT: targets / cues changed"
                        r_ = r2_
                        for t_ in r2_.get("distract_types", ()):                # §257: per-type insert counts
                            DTYPE[t_] = DTYPE.get(t_, 0) + 1
                DSTAT[n_] = DSTAT.get(n_, 0) + 1; out_.append(r_)
                if r_.get("distract_capped") is not None:                       # §265: a draw the name budget capped
                    DCAP[r_["distract_capped"]] = DCAP.get(r_["distract_capped"], 0) + 1
            assert not any(r_["text"] in INS_TEXTS for r_ in out_), "DISTRACT: a training text equals an INSERT eval text"
            assert not any(r_["text"] in TP_TEXTS for r_ in out_), "§262: a training text equals a TEST-PERM text"
            if PALL:
                pall_check(raws, [r_["text"] for r_ in out_])
            return [tok_aw(r_, a.aux_target, a.cue, a.write_src) for r_ in out_]
        if a.perm_p:                                                        # §252 --perm_p: per-item seeded coin / permutation (d >= 2), solve + disjointness asserted
            raws = [sdraw(stream, d) for _ in range(a.bs - NASK)]; base_ = raws
            tp_ = PR.train_perm_all if PALL else PR.train_perm               # §270: scope all, the same coin
            raws = [tp_(r_, a.seed, step, k_) if (d >= 2 and np.random.default_rng((a.seed, PR.SEED_PERM_TRAIN + 1, step, k_)).random() < a.perm_p) else r_
                    for k_, r_ in enumerate(raws)]
            assert not any(r_["text"] in TP_TEXTS for r_ in raws), "§252: a training text equals a TEST-PERM text"
            if PALL:
                pall_check(base_, [r_["text"] for r_ in raws])
            return [tok_aw(r_, a.aux_target, a.cue, a.write_src) for r_ in raws]
        if a.src == "so":                                                   # §271: so_tok items (NASK = 0)
            return [tok_so(stream.draw(d), SOCUE) for _ in range(a.bs)]
        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs - NASK)]

    ask_stream = OW.Stream(SEED_ASK_STREAM, EXCL, frames=TRAIN_FR, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW) if NASK else None   # §262: own stream, same EXCL

    def ask_items(d):
        """§262 Note 262-2: NASK one-hop ask items at the step's d (own stream; per item k a perm coin / permutation, then ST_v1's class; disjointness asserted)."""
        out_ = []
        for k_ in range(NASK):
            r_ = sdraw(ask_stream, d); b_ = r_
            if d >= 2 and np.random.default_rng((a.seed, SEED_ASK_COIN, step, k_)).random() < a.ask_perm_p:
                if PALL:                                                    # §270: scope all (default_rng((seed, 27001, step, k)) replaces SEED_ASK_PERM's draw; same coin)
                    r_ = PR.train_perm_all(r_, a.seed, step, k_, PR.SEED_ASK_PERM_ALL)
                else:
                    pit, _, inv = PR.permuted(r_, np.random.default_rng((a.seed, SEED_ASK_PERM, step, k_))); PR.check(r_, pit); assert inv >= 1; pit["perm"] = 1; r_ = pit
            o_ = tok_ask(r_, a.aux_target, a.write_src); assert o_["ask_text"] not in TP_TEXTS and o_["ask_text"] not in INS_TEXTS, "§262: an ask text equals an eval text"
            if PALL:
                pall_check([b_], [o_["ask_text"]])
            out_.append(o_)
        return out_

    def si_aux(items, stat, zero):
        """Note 271-1 DIRECT-SI: per item i and t <= m (aux_d_max filter as the loop's), the one-pass item think_cues[t - 1] = [EOT] + enc(' ' + text + cue_si(it, t))
        + SFX -> CE(the ':' logits, VAL_IDS[S_t]); -> mean_items(mean_t w_t CE_t / n_tok(i, t)), w_t = AUX_W[t - 1] under --aux_reweight (the loop's aux law), n_tok = si_weight (Note 271-5); stat: aux / vacc."""
        pairs = [(i, t) for i, it in enumerate(items) for t in range(1, len(it["think_cues"]) + 1) if not a.aux_d_max or it["d"] <= a.aux_d_max]
        assert all(it["cue"] == "index" and len(it["think_cues"]) == it["d"] for it in items), "DIRECT-SI: SI cues, one per swap"
        if not pairs:
            return zero
        y = torch.tensor(VOC[[val_index(items[i]["raw"]["values"][t]) for i, t in pairs]], device=dev)
        lg_s = direct_logits(g, [{"dec": items[i]["think_cues"][t - 1]} for i, t in pairs], dev, ac); ce = Fn.cross_entropy(lg_s, y, reduction="none"); per = [[] for _ in items]
        for r_, (i, t) in enumerate(pairs):
            w_ = si_weight(items[i], t); per[i].append(ce[r_] * w_ * AUX_W[t - 1] if a.aux_reweight else ce[r_] * w_)   # Note 271-5: x 1 / n_tok
        aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()
        if stat is not None:
            ok_ = (lg_s.detach().argmax(-1) == y).float(); vok = {t: float(ok_[[r_ for r_, (_, t_) in enumerate(pairs) if t_ == t]].mean()) for t in sorted({t for _, t in pairs})}
            stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))
            stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))
        return aux

    def loss_reg(items, stat=None):
        """nested170.loss_items (direct; inject with the aux on) on token-list items; stat (dict) collects the log fields (no effect on the graph)."""
        tgt = torch.tensor(VOC[[it["ans"] for it in items]], device=dev)
        if a.arm == "direct":
            lg = direct_logits(g, items, dev, ac); la = Fn.cross_entropy(lg, tgt); loss = la
            if SI:                                                          # Note 271-1 DIRECT-SI: + lambda * the per-swap one-pass CE
                loss = la + lam_now[0] * si_aux(items, stat, la * 0.0)
        else:
            from torch.utils.checkpoint import checkpoint
            xs, msk = cue_inputs(m, items, dev); per = [[] for _ in items]; vok = {}; per_c = {}   # Note 251-10: per_c = {item: the ACUE CE}   # §233: pass t reads xs[min(t, len(xs)) - 1] (cue none: one input)
            with ac():
                S = m.mem.zero_state(len(items), dev); hls = [m.lower(x_) for x_ in xs] if a.cache_lower else None   # the lower cache per distinct input
                Ssv = [S.detach()] if (HOLD and stat is not None) else None   # §240: S_0..S_K detached (the penalty's inputs / the hold stats)
                for t in range(1, K + 1):
                    i_ = min(t, len(xs)) - 1; x = xs[i_]; hl = hls[i_] if hls is not None else None
                    cidx = [i for i, it in enumerate(items) if it["d"] == t] if a.cue_ans else []   # Note 251-10 --cue_ans: pass t == d -> the ':' LM logits vs the answer
                    if cidx:
                        assert not a.ckpt_passes and a.write_src == "answer", "--cue_ans: needs --write_src answer (pass_aw), no --ckpt_passes"
                        S, h_c, pos_c = m.pass_aw(S, x, msk, hl); lg_c = m.colon_logits(h_c[cidx], pos_c[cidx])
                        y_c = torch.tensor(VOC[[items[i]["ans"] for i in cidx]], device=dev); ce_c = Fn.cross_entropy(lg_c, y_c, reduction="none")
                        for r_, i in enumerate(cidx):
                            per_c[i] = ce_c[r_]
                        if stat is not None:
                            stat.setdefault("acue", []).append((str(t), int((lg_c.detach().argmax(-1) == y_c).sum()), len(cidx), float(ce_c.detach().sum())))
                    elif a.perm_p:                                          # §252: pass_aw (S = m.write's) keeps h for the split WS readout
                        S, h_c, pos_c = m.pass_aw(S, x, msk, hl)
                    else:
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if (a.ckpt_passes and torch.is_grad_enabled()) else m.write(S, x, msk, hl)
                    if a.perm_p and stat is not None:                       # §252 ws_readout_perm / _chain on the step's own training items (no_grad, never fed back)
                        with torch.no_grad():
                            pw_ = m.colon_logits(h_c.detach(), pos_c)[:, VOC].argmax(-1).tolist()
                        for i_, it_ in enumerate(items):
                            stat.setdefault("wsp", []).append((int(it_["raw"].get("perm", 0)), str(it_["d"]), str(t), int(pw_[i_] == int(it_["raw"]["values"][min(t, it_["d"])]))))
                    if Ssv is not None:
                        Ssv.append(S.detach())
                    idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t and (not a.aux_d_max or it["d"] <= a.aux_d_max)]   # §251 --aux_d_max
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
            aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean() if any(per) else la * 0.0   # §251: no qualifying item -> an in-graph zero
            loss = la + lam_now[0] * aux
            if a.cue_ans:                                                   # Note 251-10: + λ · mean CE over the items whose pass d ran (an in-graph zero if none)
                acue = torch.stack([per_c[i] for i in sorted(per_c)]).mean() if per_c else la * 0.0; loss = loss + a.cue_ans * acue
            if stat is not None:
                stat["aux"] = stat.get("aux", 0.0) + float(aux.detach()) * len(items); stat.setdefault("vacc", []).append((vok, len(items)))
                stat.setdefault("vacc_d", []).append((str(items[0]["d"]), vok, len(items)))   # §233 aux_vacc_by_d_pass (one d per step)
                if Ssv is not None:
                    stat.setdefault("_keep", []).append((Ssv, xs, msk, [it["d"] for it in items]))
        if stat is not None:
            with torch.no_grad():
                stat["acc"] = stat.get("acc", 0.0) + float((lg.detach()[:, VOC].argmax(-1).cpu().numpy() == np.array([it["ans"] for it in items])).sum())
            stat["la"] = stat.get("la", 0.0) + float(la.detach()) * len(items); stat["n"] = stat.get("n", 0) + len(items)
        return loss

    def loss_ask(items, stat=None):
        """§262 Note 262-2: CE(the ':' full-vocabulary LM logits, v_X) at every pass 1..K, the mean over passes (batch mean CE per pass); no decoder / aux CE."""
        xs, msk = cue_inputs(m, items, dev); assert len(xs) == 1 and all(it.get("ask") for it in items), "§262: one input per ask item"
        y = torch.tensor(VOC[[it["ask_v"] for it in items]], device=dev); ces = []
        with ac():
            S = m.mem.zero_state(len(items), dev); hl = m.lower(xs[0]) if a.cache_lower else None
            for t in range(1, K + 1):
                S, h_c, pos_c = m.pass_aw(S, xs[0], msk, hl); lg_c = m.colon_logits(h_c, pos_c); ces.append(Fn.cross_entropy(lg_c, y))
                if stat is not None:
                    stat.setdefault("ask", []).append((str(t), int((lg_c.detach().argmax(-1) == y).sum()), len(items)))
            if stat is not None and PALL:                                   # §270 (log-only): the pass-K ':' acc by ask class
                stat.setdefault("askc", []).extend(zip([it["ask_class"] for it in items], (lg_c.detach().argmax(-1) == y).tolist()))
        return torch.stack(ces).mean()

    def loss_items(items, stat=None):
        """§262: the ask items (last in the batch) take loss_ask, the rest loss_reg; the item-weighted mean (no ask items: loss_reg exactly)."""
        rg = [it for it in items if not it.get("ask")]; ak = [it for it in items if it.get("ask")]
        if not ak:
            return loss_reg(items, stat)
        assert all(it is r for it, r in zip(items, rg)), "§262: the ask items come last"
        l_a = loss_ask(ak, stat)
        return l_a if not rg else (loss_reg(rg, stat) * len(rg) + l_a * len(ak)) / len(items)

    s_mc = OW.Stream(SEED_MEMCHECK, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW) if a.src == "ow" else None   # §236: the run's law
    if a.distract_p and s_mc:                                               # Note 255-5 (d): the memory-check items carry 2 insert pairs (ops after the last op sentence)
        def _mc2(i_):
            r_ = s_mc.draw(a.d_max); g_, o_, m_ = IR.draw_pairs(r_, max(2, a.distract_nmax), np.random.default_rng((a.seed, IR.SEED_DISTRACT + 2, i_)))   # §265: the nmax-insert shape
            return IR.assemble(r_, g_, [(("after", "S%d" % r_["d"]), x_) for x_ in o_], m_)
        MC_TR = sorted([tok_aw(_mc2(i_), a.aux_target, a.cue, a.write_src) for i_ in range(256)], key=lambda it: -st_len(it))
    else:
        MC_TR = sorted([tok_aw(s_mc.draw(a.d_max), a.aux_target, a.cue, a.write_src) for _ in range(256)], key=lambda it: -st_len(it)) if s_mc else None
    MC_EV = sorted([tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][12]], key=lambda it: -st_len(it))[:1] if s_mc else None   # §233: TEST2's own longest d 12 (cued)
    WS_ITEMS = {d_: [tok_aw(it, a.aux_target, a.cue, a.write_src) for it in SETS["test2"][d_][:200]] for d_ in range(1, a.d_max + 1)} if (a.write_src == "answer" and a.src == "ow") else None   # §234 WS-READOUT (Note 234-5 c: d 1..d_max)

    if a.src == "so":                                                       # §271: memory-check items (train: n 5, m d_max; eval: TEST n 5 m 12's longest)
        s_mcs = SO.Stream(SEED_MEMCHECK, frames=SO.FRAME_TRAIN, ns=(5,), offset=bool(a.so_offset))
        MC_TR = sorted([tok_so(s_mcs.draw(a.d_max), SOCUE) for _ in range(256)], key=lambda it: -st_len(it))
        MC_EV = sorted([tok_so(it, SOCUE) for md_ in SO.MODES for it in SO_SETS["test"][(5, 12, md_)]], key=lambda it: -st_len(it))[:1]

    def mem_train(micro):
        """the training step's peak at bs / micro of the longest d_max items + AdamW's two fp32 states (nested170's rule)."""
        torch.cuda.reset_peak_memory_stats(dev); lam_now[0] = a.aux_step; st_m = {} if HOLD else None; l_ = loss_items(MC_TR[:a.bs // micro], st_m); l_.backward()
        if HOLD:                                                            # §240: the penalty (on) and the far pass at hold_far_hi inside the peak
            hold_step(m, st_m.pop("_keep"), MC_TR[:min(a.hold_far_n, a.bs // micro)], a.hold_far_hi, K, a.bs, a.hold_inc > 0, a.hold_inc, ac, dev)
        params.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30 + 2 * sum(p_.numel() * 4 for p_ in params.parameters() if p_.requires_grad) / 2 ** 30

    def mem_eval(ebs):
        """the eval's peak (the optimiser state is cleared before the in-process eval): the longest d 12 item x ebs at K' 0..14, fp32."""
        from boxes import inject_aw_eval as EV
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); EV.answers_by_kprime(m, g, MC_EV * ebs, (64 if a.src == "so" else 14) if K else 0, dev, ebs, None)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.src in ("ow", "so") and not a.args_only and (a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume)):
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
                   "eval": "the longest of 256 d = 12 items x eval_bs at K' 0..14 (fp32, no optimiser state)", **({"so": "src so: train = the longest bs / micro of 256 n 5 m d_max items; eval = TEST n 5 m 12's longest x eval_bs at K' 0..64"}
                                                                         if a.src == "so" else {})}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return 0
        assert all(c["peak_gib"] <= a.mem_budget for c in mc[-1:]) and [c for c in mc if c["kind"] == "train"][-1]["peak_gib"] <= a.mem_budget, mc
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "hostprobe_sps": os.environ.get("HOSTPROBE_SPS"), "module_sha256": sha_file(os.path.abspath(__file__)), "inject_ow_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_ow.py")), "inject_st_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_st.py")),
            "eval_module_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw_eval.py")), "nested170_sha256": sha_file(os.path.join(ROOT, "boxes", "nested170.py")),
            "deltamem_sha256": sha_file(os.path.join(ROOT, "boxes", "deltamem.py")), "ow_words_sha256": sha_file(os.path.join(ROOT, "boxes", "ow_words.py")),
            "nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")), "inject_aw_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw.py")),
            "inject_aw_eval_sha256": sha_file(os.path.join(ROOT, "boxes", "inject_aw_eval.py")), "so_words_sha256": sha_file(os.path.join(ROOT, "boxes", "so_words.py")),
            "so_tok_sha256": sha_file(os.path.join(ROOT, "boxes", "so_tok.py")),
            "so_eval_module_sha256": (sha_file(os.path.join(ROOT, "boxes", "inject_so_eval.py")) if os.path.exists(os.path.join(ROOT, "boxes", "inject_so_eval.py")) else None) if a.src == "so" else None, "nk_sha256": sha_file(os.path.join(ROOT, "boxes", "nk.py")) if a.nk else None,
            "ask_render_sha256": sha_file(os.path.join(ROOT, "boxes", "ask_render.py")) if a.ask_p else None, "set_build_sec": set_sec, "excl_n": len(EXCL), "d_law": "d = default_rng(seed + 22331).integers(1, d_max + 1) per step",
            "format": so_format(a) if a.src == "so" else {"direct": "dec = [EOT] + enc(' ' + text) + enc(' The answer is:') -> the number token (plain trunk, one pass)",
                       "inject": "think [EOT] + enc(' ' + text) (padded writes masked), K passes; dec as direct, read-only on S_K; aux [EOT, ' Step', ' t', ' :'] + "
                                 "enc(' ' + R_step[t]) + EOT, read-only on S_t, t <= d"}[a.arm] + (CUE_FMT[a.cue] if a.cue not in ("generic", "target") else {"generic": " | §252 GP: think_t = [EOT] + enc(' ' + text + ' What is the next value?') (every pass)", "target": " | §261 TC: think_t = [EOT] + enc(' ' + text + ' To find how many <u> <S_d> has, what is the next value?') (every pass)"}[a.cue]) + AW_FMT[a.write_src] + (" | §262 NK: one name-keyed head packed into S (k = q = normalize(W_n e(name)); write key = the cue's name, v = W_vn ln_w(h_11 at ':'), delta rule; read at block 6 at name tokens + W_on S_n q, W_on = 0 at init, no fade)" if a.nk else "") + (" | §262 ASK: round(ask_p bs) one-hop ask items per step (ST_v1 classes after perm p ask_perm_p; QUESTION(u, X) + SFX at every pass; CE(':' logits, v_X) at every pass, no decoder / aux CE)" if a.ask_p else "")}
    if SETS is not None:
        info.update(SETS["shas"]); info["pair_stats"] = SETS["pair_stats"]
    if SO_SETS is not None:                                                 # §271: the SO set shas (TEST per cell, TRANSFER per m, OFFSET) + the training names
        info["so_set_shas"] = {**{"test|%d|%d|%s" % k_: SO.set_sha(v_) for k_, v_ in sorted(SO_SETS["test"].items())},
                               **{"transfer|%d" % k_: SO.set_sha(v_) for k_, v_ in sorted(SO_SETS["transfer"].items())}, "offset": SO.set_sha(SO_SETS["offset"])}
        info["so_train_names"] = SO_NAMES
    if a.args_ref:
        ref = json.load(open(a.args_ref)); dif = {k: [ref.get(k), info.get(k)] for k in sorted(set(ref) | set(info)) if k not in META and ref.get(k) != info.get(k)}
        json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    if a.args_only:
        print("ARGS_ONLY", a.out, flush=True); return 0
    opt = make_opt(params, a.lr); step0 = 1; n_nonfinite = 0; hist = []; wg_hist = []; wwin = []; wlast = []; gn_log = []; aux_hist = []
    wsbin = {}                                                              # §252: {"perm" / "chain": {d: {t: [correct, n]}}} (the open 1k bin; not in the resume ckpt)
    acbin = {}                                                              # Note 251-10: {d: [correct, n, ce sum]} (the open 1k bin; not in the resume ckpt)
    askbin = {}                                                             # §262: {pass: [correct, n]} (the open 1k bin; not in the resume ckpt)
    askcbin = {}                                                            # §270 (scope all, log-only): {ask class: [correct at pass K, n]} (the open 1k bin)
    expo = {}; binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; gn_plain_log = []; dseq = []; iseq = []; d2_bins = []; ent = {"buf": [], "entry_d1": None}; fut = {"checked": [], "stop": None}
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); params.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng_d.bit_generator.state = ck["rng_d"]; rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        if stream is not None:
            stream.rng.bit_generator.state = ck["stream_rng"]; stream.stats = dict(ck["stream_stats"])
        if ask_stream is not None:                                          # §262
            ask_stream.rng.bit_generator.state = ck["ask_stream_rng"]
        step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; gn_log = list(ck["gn_log"]); t0 = time.time() - ck["elapsed"]
        expo, binacc, vbin, dseq, iseq, d2_bins, ent, fut, wlast = ck["expo"], ck["binacc"], ck["vbin"], ck["dseq"], ck["iseq"], ck["d2_bins"], ck["ent"], ck["fut"], ck["wlast"]
        vbin_d = ck.get("vbin_d", {}); gnbin = ck.get("gnbin", {}); hbin = ck.get("hbin", {}); hbin_d = ck.get("hbin_d", {}); gn_plain_log = list(ck.get("gn_plain_log", []))
        if ck.get("rng_far") is not None:
            rng_far.bit_generator.state = ck["rng_far"]
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif m is not None and K > 0 and not a.init_model:                      # note 147-2: g_add calibration (§237: skipped under --init_model, the loaded s_add kept)
        if a.src == "nested":
            its = [tok_nested(it) for it in N7.train_items(np.random.default_rng(17001), 64, a.n_max_train, set())]
        elif a.src == "so":                                                 # §271: the SO training law's items
            s_ = SO.Stream(SEED_CAL, frames=SO.FRAME_TRAIN, ns=SO_NS, modes=SO_MODES, offset=bool(a.so_offset), names=SO_NAMES); its = [tok_so(s_.draw(1 + i % a.d_max), SOCUE) for i in range(64)]
        else:
            s_ = OW.Stream(SEED_CAL, frames=OW.FRAME_TRAIN, mul_s_p=a.mul_s_p, **OWKW); its = [tok_aw(s_.draw(1 + i % a.d_max), a.aux_target, a.cue, a.write_src) for i in range(64)]
        xs, msk = cue_inputs(m, its, dev); m.mem.norms = []
        with torch.no_grad(), ac():
            S_c = m.mem.zero_state(64, dev)
            for t_ in range(1, min(K, 4) + 1):                              # InjectN.think's write sequence, the pass-t input xs[min(t, len) - 1]
                S_c = m.write(S_c, xs[min(t_, len(xs)) - 1], msk)
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0, "items": "64 items, d = 1 + i % d_max (SEED_CAL)" if a.src in ("ow", "so") else "nested170 17001"},
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
        d = int(rng_d.integers(1, a.d_max + 1)) if a.src in ("ow", "so") else 0
        its = draw_items(d) + (ask_items(d) if NASK else []); assert len(its) == a.bs; opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; st_ = {}   # §262: + the ask items
        for k_ in range(a.micro):
            lk = loss_items(its[k_ * ch:(k_ + 1) * ch], st_); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        if HOLD:                                                            # §240 (LH_v1 §2, Note 240-1): the plain grads are complete -> gn_plain (exact)
            gn_plain_log.append(float(torch.sqrt(sum((p_.grad.float() ** 2).sum() for p_ in params.parameters() if p_.grad is not None))))
            T_far = int(rng_far.integers(a.hold_far_lo, a.hold_far_hi + 1)); inc_on = a.hold_inc > 0 and step >= a.hold_start * sc
            hs_ = hold_step(m, st_.pop("_keep"), its[:a.hold_far_n], T_far, K, a.bs, inc_on, a.hold_inc, ac, dev); hold_acc(hbin, hs_)
            hold_acc(hbin_d.setdefault(str(its[0]["d"]), {}), {k_: hs_[k_] for k_ in ("s8", "sd2")})   # §243 DC_v1 fix 2: the per-d norm guard (one d per step)
            if step == step0 and not a.resume:                              # the norm guard at step 0 (this step's forward, before any update)
                log.write(json.dumps({"step": 0, "norm_guard": {"s8": hs_["s8"][0] / hs_["s8"][1], "sd2": hs_["sd2"][0] / hs_["sd2"][1]},
                                      "hold_ratios": {k_: (v_[0] / v_[1] if v_[1] else None) for k_, v_ in hs_.items() if k_ in ("comp", "hold")}}) + "\n"); log.flush()
        if m is not None:
            wwin.append(write_grad_norm(m))
        gn = torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0); gn_log.append(float(gn))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        la_ = st_["la"] / st_["n"]; acc_ = st_["acc"] / st_["n"]; aux_hist.append((la_, st_["aux"] / st_["n"] if "aux" in st_ else None))
        dk = str(its[0]["d"]) if a.src in ("ow", "so") else "mix"; expo[dk] = expo.get(dk, 0) + 1
        b_ = binacc.setdefault(dk, [0, 0.0, 0.0, 0]); b_[0] += 1; b_[1] += acc_; b_[2] += la_; b_[3] += float(gn) > 1.0
        gb_ = gnbin.setdefault(dk, [0.0, 0]); gb_[0] += float(gn); gb_[1] += 1   # §237: pre-clip grad norm per d (emitted under --init_model)
        for pp_, dd_, tt_, ok_ in st_.get("wsp", []):                      # §252: the split WS readout (per 1k bin)
            wb_ = wsbin.setdefault("perm" if pp_ else "chain", {}).setdefault(dd_, {}).setdefault(tt_, [0, 0]); wb_[0] += ok_; wb_[1] += 1
        for dd_, c_, n_, s_ in st_.get("acue", []):                        # Note 251-10: ACUE acc / CE by d (per 1k bin)
            ab_ = acbin.setdefault(dd_, [0, 0, 0.0]); ab_[0] += c_; ab_[1] += n_; ab_[2] += s_
        for tt_, c_, n_ in st_.get("ask", []):                              # §262: ask acc per pass (per 1k bin)
            kb_ = askbin.setdefault(tt_, [0, 0]); kb_[0] += c_; kb_[1] += n_
        for cl_, ok_ in st_.get("askc", []):                               # §270 (scope all): pass-K ask acc by class
            kc_ = askcbin.setdefault(str(cl_), [0, 0]); kc_[0] += int(ok_); kc_[1] += 1
        for vok, _n in st_.get("vacc", []):
            for t_, v_ in vok.items():
                vb_ = vbin.setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1
        for dd_, vok, _n in (st_.get("vacc_d", []) if a.cue != "none" else []):
            for t_, v_ in vok.items():
                vb_ = vbin_d.setdefault(dd_, {}).setdefault(str(t_), [0.0, 0]); vb_[0] += v_; vb_[1] += 1
        dseq.append(dk if a.src in ("ow", "so") else "0"); iseq.append(dk + ":" + ",".join(it["sig"] for it in its))
        if dk == "1" and ent["entry_d1"] is None:                           # entry_d1: the first 100 consecutive d = 1 steps with mean answer CE < 0.5
            ent["buf"].append(la_)
            if len(ent["buf"]) == 100:
                if float(np.mean(ent["buf"])) < 0.5:
                    ent["entry_d1"] = step; log.write(json.dumps({"step": step, "entry_d1": step}) + "\n"); log.flush()
                ent["buf"] = []
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0), "aux_lambda": lam_now[0],
                   "answer_ce": float(np.mean([x_[0] for x_ in aux_hist])) if aux_hist else None}
            if m is not None or SI:
                rec["aux_ce"] = float(np.mean([x_[1] for x_ in aux_hist])) if aux_hist else None
            hist = []; aux_hist = []
            if m is not None and wwin:
                md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
                w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
                wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
                rec.update({"gamma": float(m.mem.gamma), "fade": float(m.mem.fade), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]})
            if HOLD and gn_plain_log:                                       # §240 LP0-W: the plain-loss grad norm (exact, every step)
                rec["gn_plain_mean100"] = float(np.mean(gn_plain_log[-100:]))
            rec.update({"gn_preclip": gn_log[-1], "gn_preclip_mean100": float(np.mean(gn_log[-100:])), "clip_frac_100": float(np.mean([v_ > 1.0 for v_ in gn_log[-100:]]))})
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if step % 1000 == 0 or step == last:                                # the 1k-bin record (a partial bin at a stop / the end)
            bd = {k_: {"steps": v_[0], "acc": v_[1] / v_[0], "ce": v_[2] / v_[0], "clip": v_[3] / v_[0]} for k_, v_ in sorted(binacc.items())}
            br = {"step": step, "bin": (step - 1) // 1000, "bin_steps": len(dseq), "by_d": bd, "exposure_by_d": dict(sorted(expo.items())), "dseq": "".join(dseq),
                  "ihash": hashlib.sha1("\n".join(iseq).encode()).hexdigest()[:16]}
            if vbin:
                br["aux_vacc_by_pass"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(vbin.items(), key=lambda kv: int(kv[0]))}
            if vbin_d:                                                      # §233 LP2 key (+ the per-cell counts: G10 pools them back to aux_vacc_by_pass)
                br["aux_vacc_by_d_pass"] = {d_: {k_: v_[0] / v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}
                br["aux_vacc_by_d_pass_n"] = {d_: {k_: v_[1] for k_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(vbin_d.items())}
            if a.distract_p:                                                # Note 255-5: the bin's insert-count distribution
                br["distract_n"] = {str(k_): v_ for k_, v_ in sorted(DSTAT.items())}; DSTAT.clear()
                if DCAP:                                                    # §265: the capped draws (drawn n: count)
                    br["distract_capped"] = {str(k_): v_ for k_, v_ in sorted(DCAP.items())}; DCAP.clear()
                if MIXT:                                                    # §257: the bin's insert-type distribution
                    br["distract_types"] = dict(sorted(DTYPE.items())); DTYPE.clear()
            for k_ in ("perm", "chain") if a.perm_p else ():                # §252 Note 252-1: ws_readout_perm / ws_readout_chain = {d: {pass: acc}} (+ _n)
                wv_ = wsbin.get(k_, {})
                br["ws_readout_" + k_] = {d_: {t_: v_[0] / v_[1] for t_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(wv_.items())}
                br["ws_readout_" + k_ + "_n"] = {d_: {t_: v_[1] for t_, v_ in sorted(tv.items(), key=lambda kv: int(kv[0]))} for d_, tv in sorted(wv_.items())}
            if acbin:                                                       # Note 251-10 --cue_ans
                br["acue_by_d"] = {k_: {"acc": v_[0] / v_[1], "ce": v_[2] / v_[1], "n": v_[1]} for k_, v_ in sorted(acbin.items())}
            if askbin:                                                      # §262 Note 262-2: the ask items' ':' acc per pass (full-vocabulary argmax == v_X)
                br["ask_acc_by_pass"] = {t_: v_[0] / v_[1] for t_, v_ in sorted(askbin.items(), key=lambda kv: int(kv[0]))}; br["ask_n"] = askbin["1"][1]
            if askcbin:                                                     # §270 (scope all, log-only): the pass-K ask acc by class
                br["ask_by_class"] = {k_: {"acc": v_[0] / v_[1], "n": v_[1]} for k_, v_ in sorted(askcbin.items())}
            if PALL:                                                        # §270 Note 270-1 (1): cumulative resampled base items (chain + ask streams)
                br["multiset_rejects"] = MSREJ[0]
            if WS_ITEMS is not None:                                        # §234 WS-READOUT: TEST2 d 1..d_max (first 200 each), fp32, no_grad, never fed back
                br["ws_readout"] = ws_readout(m, WS_ITEMS, dev, K)
            if a.mul_s_p3 < 1.0 and stream is not None:                    # §235: cumulative accepted d 3 training items with a mul_s step / all d 3 items
                br["stream_d3_mul_s"] = [stream.stats.get("d3_mul_s", 0), stream.stats.get("d3_items", 0)]
            if a.init_model and gnbin:                                       # §237 (Note 237-1): the mean pre-clip grad norm per d over the bin
                br["gn_preclip_by_d"] = {k_: v_[0] / v_[1] for k_, v_ in sorted(gnbin.items())}
            if HOLD and hbin:                                               # §240: inc (the penalised squared ratio), far, comp / hold ratios, the norm guard
                hm_ = hold_mean(hbin); br["hold"] = {"inc_ratio": hm_.get("inc"), "far_ratio": hm_.get("far"), "comp_ratio": hm_.get("comp"), "hold_ratio": hm_.get("hold"),
                                                     "norm_guard": {"s8": hm_.get("s8"), "sd2": hm_.get("sd2")}}
                br["hold"]["norm_guard_by_d"] = {k_: hold_mean(v_) for k_, v_ in sorted(hbin_d.items())}   # §243: {d: {s8, sd2}} (s8 = ||S_K||, K = think_k)
            log.write(json.dumps(br) + "\n"); log.flush()
            if "2" in bd and step % 1000 == 0:
                d2_bins.append([step, bd["2"]["acc"]])
            binacc = {}; vbin = {}; vbin_d = {}; gnbin = {}; hbin = {}; hbin_d = {}; dseq = []; iseq = []; acbin = {}; wsbin = {}; askbin = {}; askcbin = {}
        if FUT and step in FUT and fut["stop"] is None:                     # LAT_v1 §4 futility (the d = 2 train acc of complete 1k bins)
            l4 = [v_ for _, v_ in d2_bins[-4:]]; sl = float(np.polyfit(np.arange(len(l4)), l4, 1)[0]) if len(l4) >= 2 else None
            accl = l4[-1] if l4 else None; rising = sl is not None and sl > 0.02
            frec = {"step": step, "d2_last_bin_acc": accl, "slope_last4": sl, "bins": d2_bins[-4:], "rising": rising,
                    "stop": bool(accl is not None and accl < 0.50 and not rising)}
            fut["checked"].append(frec); json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1)
            log.write(json.dumps({"futility": frec}) + "\n"); log.flush()
            if frec["stop"]:
                fut["stop"] = step; json.dump(fut, open(os.path.join(a.out, "futility.json"), "w", newline="\n"), indent=1); break
        if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.stop_at) and (step < a.steps or a.keep_final_ckpt):   # §244: + the final step under --keep_final_ckpt
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": params.state_dict(), "opt": opt.state_dict(), "rng_d": rng_d.bit_generator.state, "np_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None,
                        "stream_rng": stream.rng.bit_generator.state if stream is not None else None, "ask_stream_rng": ask_stream.rng.bit_generator.state if ask_stream is not None else None, "stream_stats": dict(stream.stats) if stream is not None else None,
                        "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist, "gn_log": gn_log, "elapsed": time.time() - t0, "expo": expo, "binacc": binacc,
                        "vbin": vbin, "vbin_d": vbin_d, "gnbin": gnbin, "hbin": hbin, "hbin_d": hbin_d, "gn_plain_log": gn_plain_log, "rng_far": rng_far.bit_generator.state, "dseq": dseq, "iseq": iseq, "d2_bins": d2_bins, "ent": ent, "fut": fut, "wlast": wlast}, cp)
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
    if PALL:                                                                # §270 Note 270-1 (1)
        train["multiset_rejects"] = MSREJ[0]
    json.dump(train, open(os.path.join(a.out, "train_summary.json"), "w", newline="\n"), indent=1)
    if a.src == "nested":
        return 0
    if a.src == "so":                                                       # §271: the SO eval, imported here (a missing module fails loudly at eval time)
        try:
            from boxes import inject_so_eval as EV
        except ImportError as e_:
            raise RuntimeError("§271: boxes/inject_so_eval.py missing / broken; model.pt and train_summary.json are written, run the SO eval separately") from e_
        EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SO_SETS, train=train)
    else:
        from boxes import inject_aw_eval as EV
        EV.run(a.out, [p_ for p_ in a.eval_parts.split(",") if p_], dev, a.eval_bs, model=(m, g), sets=SETS, train=train)
    if a.keep_final_ckpt and os.path.exists(os.path.join(a.out, "ckpt_last.pt")):   # §244: the final state kept for a resume (renamed: the drain skips ckpt_last.pt)
        os.replace(os.path.join(a.out, "ckpt_last.pt"), os.path.join(a.out, "ckpt_final.pt"))
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("DONE", a.out, flush=True); return 0


if __name__ == "__main__":
    sys.exit(main())
