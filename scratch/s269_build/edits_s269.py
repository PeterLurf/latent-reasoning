"""§262 NAME-KEYED MEMORY build (NK_v1 §2-§4 + Notes 262-1 / 262-2): every edit asserted to apply once; flags off (--nk 0 --ask_p 0, and not both --distract_p and
--perm_p) the computation is the pre-§262 one (gate_s269 offbit).
  scratch/s234_build/inject_aw_head.py  build_aw: InjectNK under --nk (boxes/nk.py); tok_aw: the cue-step AW assert that pass c's name token (6 before ':') == spine[c]
                                        (NK's write key); tok_ask (Note 262-2: ST_v1's class rule via boxes/ask_render.py, one input QUESTION(u, X) + SFX at every pass);
                                        SEED_ASK_* (26210-26212, unused elsewhere).
  E269 (main(), applied by mk_inject_aw.py after E268): --nk / --ask_p / --ask_perm_p; init_model strict except the three nk.* weights (asserted); perm-then-insert
        (--distract_p with --perm_p: §252's coin / permutation, then the §257 inserts; INS_TEXTS + TP_TEXTS asserts); ask items (own stream, same EXCL; NASK = round(ask_p bs)
        last in the batch; perm coin p ask_perm_p before the class; TP / INS disjointness); loss_items = loss_reg (the old body) + loss_ask (':' CE on v_X at every pass);
        ask acc per pass in the 1k bins; the ask stream's rng in the ckpt; nk / ask_render shas + format notes when on.
  E269E (inject_aw_eval run(), applied by mk_inject_aw_eval.py): the nk_pilot part (boxes.nk.part_nk_pilot -> eval_nk_pilot.json).
then regenerates boxes/inject_aw.py / boxes/inject_aw_eval.py.
    python -B scratch/s269_build/edits_s269.py"""
import subprocess
import sys

sys.path.insert(0, "scratch/s268_build")
from edits_s268_fn import edit  # noqa: E402

h = "scratch/s234_build/inject_aw_head.py"; assert "tok_ask" not in open(h, encoding="utf-8").read(), "already applied"
edit(h, [
    ('    m = cls(g, lm, a.mem_write, hm, hm, None).to(dev); m.nopass = False\n',
     '    if getattr(a, "nk", 0):                                                 # §262 NK: InjectNK (the NK block packed into S; boxes/nk.py)\n'
     '        from boxes.nk import InjectNK\n'
     '        assert cls is InjectAW, "--nk needs --write_src answer"; m = InjectNK(g, lm, a.mem_write, hm, hm, None, a.seed).to(dev); m.nopass = False\n'
     '    else:\n'
     '        m = cls(g, lm, a.mem_write, hm, hm, None).to(dev); m.nopass = False\n'),
    ('        assert all(q[-1] == COLON for q in o["think_cues"]) and o["dec"] == o["think"] + SFX, "AW: inputs / decoder"\n',
     '        assert all(q[-1] == COLON for q in o["think_cues"]) and o["dec"] == o["think"] + SFX, "AW: inputs / decoder"\n'
     '        if cue == "step":                                                   # §262 NK: the cue\'s name token (QUESTION\'s A) sits 6 before \':\' and is spine[c] (the write key)\n'
     '            assert all(q[-7] == ENC.encode(" " + it["names"]["spine"][c])[0] for c, q in enumerate(o["think_cues"], 1)), "§262: the cue name token"\n'),
    ('def build_aw(a, dev):\n',
     'SEED_ASK_STREAM, SEED_ASK_COIN, SEED_ASK_PERM = 26210, 26211, 26212          # §262 Note 262-2: the ask items\' stream / perm coin / permutation (unused elsewhere)\n\n\n'
     'def tok_ask(it, aux_target, write_src):\n'
     '    """§262 Note 262-2 one-hop ask item: ST_v1\'s class rule (boxes.ask_render.ask_of) on the (possibly permuted) item; every pass input = [EOT] + enc(\' \' + text\' +\n'
     '    \' \' + QUESTION(u, X)) + SFX (one input); trained by CE(the \':\' logits, v_X) at every pass (no decoder / aux CE); NK key = X (asserted 6 before \':\')."""\n'
     '    import boxes.ask_render as AR\n'
     '    cls, X, v, t2, fb = AR.ask_of(it); o = tok_aw(it, aux_target, "step", write_src)\n'
     '    q = [EOT] + ENC.encode(" " + t2 + " " + OW.QUESTION.format(u=it["unit"], A=X)) + SFX\n'
     '    assert write_src == "answer" and q[-1] == COLON and q[-7] == ENC.encode(" " + X)[0] and 0 <= int(v) < len(NUM_IDS), ("§262 ASK", X, v)\n'
     '    o["think_cues"] = [q]; o["ask"] = 1; o["ask_v"] = int(v); o["ask_X"] = X; o["ask_class"] = cls; o["ask_fallback"] = fb; o["ask_text"] = t2; return o\n\n\n'
     'def build_aw(a, dev):\n')])
FMT_OLD = '+ AW_FMT[a.write_src]}'
E269 = [
    ('    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")\n',
     '    ap.add_argument("--init_model_sha", default="", help="§237: its sha256 prefix (asserted)")\n'
     '    ap.add_argument("--nk", type=int, default=0, choices=[0, 1], help="§262 NK_v1: the name-keyed head packed into S (boxes/nk.py; 0 = off, unchanged)")\n'
     '    ap.add_argument("--ask_p", type=float, default=0.0, help="§262 Note 262-2: the share of each batch that is one-hop ask items (\':\' CE on v_X at every pass; 0 = off)")\n'
     '    ap.add_argument("--ask_perm_p", type=float, default=0.5, help="§262: P(an ask item with d >= 2 is permuted before its class)")\n'),
    ('    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"\n',
     '    assert bool(a.init_trunk) == bool(a.init_sha), "--init_trunk needs --init_sha (G12)"\n'
     '    NASK = int(round(a.ask_p * a.bs))                                      # §262: the ask items per step (last in the batch)\n'
     '    assert not (a.nk or a.ask_p) or (a.arm == "inject" and a.cue == "step" and a.write_src == "answer" and a.src == "ow" and not a.ckpt_passes and not a.cue_ans\n'
     '                                     and NASK < a.bs and a.bs - NASK >= a.hold_far_n), "§262 --nk / --ask_p: LAT-AW (inject, cue step, write_src answer); ask items < bs"\n'),
    ('        params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=True); a.init_model_sha256 = sha_m\n',
     '        mk_ = params.load_state_dict(torch.load(a.init_model, map_location=dev, weights_only=True), strict=not a.nk); a.init_model_sha256 = sha_m\n'
     '        assert not a.nk or (not mk_.unexpected_keys and sorted(mk_.missing_keys) == ["nk.W_n.weight", "nk.W_on.weight", "nk.W_vn.weight"]), ("§262 NK init: only nk.* new", mk_)\n'),
    ('    assert not (a.distract_p and a.perm_p), "--distract_p with --perm_p: not built"\n',
     '    # §262 NK_v1 §2: --distract_p with --perm_p = perm-then-insert (§252\'s coin / permutation first, then the §257 inserts placed by the S labels)\n'),
    ('for it in its_} if a.perm_p else set()',
     'for it in its_} if (a.perm_p or a.ask_p) else set()'),
    ('            raws = [stream.draw(d) for _ in range(a.bs)]; out_ = []\n',
     '            raws = [stream.draw(d) for _ in range(a.bs - NASK)]; out_ = []\n'),
    ('                n_ = 0\n                if np.random.default_rng((a.seed, IR.SEED_DISTRACT_COIN, step, k_)).random() < a.distract_p:\n',
     '                n_ = 0\n'
     '                if a.perm_p and d >= 2 and np.random.default_rng((a.seed, PR.SEED_PERM_TRAIN + 1, step, k_)).random() < a.perm_p:   # §262 perm-then-insert: §252\'s coin / perm\n'
     '                    r_ = PR.train_perm(r_, a.seed, step, k_)\n'
     '                if np.random.default_rng((a.seed, IR.SEED_DISTRACT_COIN, step, k_)).random() < a.distract_p:\n'),
    ('            assert not any(r_["text"] in INS_TEXTS for r_ in out_), "DISTRACT: a training text equals an INSERT eval text"\n',
     '            assert not any(r_["text"] in INS_TEXTS for r_ in out_), "DISTRACT: a training text equals an INSERT eval text"\n'
     '            assert not any(r_["text"] in TP_TEXTS for r_ in out_), "§262: a training text equals a TEST-PERM text"\n'),
    ('            raws = [stream.draw(d) for _ in range(a.bs)]\n',
     '            raws = [stream.draw(d) for _ in range(a.bs - NASK)]\n'),
    ('        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs)]\n',
     '        return [tok_aw(stream.draw(d), a.aux_target, a.cue, a.write_src) for _ in range(a.bs - NASK)]\n\n'
     '    ask_stream = OW.Stream(SEED_ASK_STREAM, EXCL, frames=TRAIN_FR, mul_s_p3=a.mul_s_p3, mul_s_p=a.mul_s_p, **OWKW) if NASK else None   # §262: own stream, same EXCL\n\n'
     '    def ask_items(d):\n'
     '        """§262 Note 262-2: NASK one-hop ask items at the step\'s d (own stream; per item k a perm coin / permutation, then ST_v1\'s class; disjointness asserted)."""\n'
     '        out_ = []\n'
     '        for k_ in range(NASK):\n'
     '            r_ = ask_stream.draw(d)\n'
     '            if d >= 2 and np.random.default_rng((a.seed, SEED_ASK_COIN, step, k_)).random() < a.ask_perm_p:\n'
     '                pit, _, inv = PR.permuted(r_, np.random.default_rng((a.seed, SEED_ASK_PERM, step, k_))); PR.check(r_, pit); assert inv >= 1; pit["perm"] = 1; r_ = pit\n'
     '            o_ = tok_ask(r_, a.aux_target, a.write_src); assert o_["ask_text"] not in TP_TEXTS and o_["ask_text"] not in INS_TEXTS, "§262: an ask text equals an eval text"\n'
     '            out_.append(o_)\n'
     '        return out_\n'),
    ('    def loss_items(items, stat=None):\n        """nested170.loss_items',
     '    def loss_reg(items, stat=None):\n        """nested170.loss_items'),
    ('        return loss\n\n    s_mc = OW.Stream(SEED_MEMCHECK',
     '        return loss\n\n'
     '    def loss_ask(items, stat=None):\n'
     '        """§262 Note 262-2: CE(the \':\' full-vocabulary LM logits, v_X) at every pass 1..K, the mean over passes (batch mean CE per pass); no decoder / aux CE."""\n'
     '        xs, msk = cue_inputs(m, items, dev); assert len(xs) == 1 and all(it.get("ask") for it in items), "§262: one input per ask item"\n'
     '        y = torch.tensor(NUM_IDS[[it["ask_v"] for it in items]], device=dev); ces = []\n'
     '        with ac():\n'
     '            S = m.mem.zero_state(len(items), dev); hl = m.lower(xs[0]) if a.cache_lower else None\n'
     '            for t in range(1, K + 1):\n'
     '                S, h_c, pos_c = m.pass_aw(S, xs[0], msk, hl); lg_c = m.colon_logits(h_c, pos_c); ces.append(Fn.cross_entropy(lg_c, y))\n'
     '                if stat is not None:\n'
     '                    stat.setdefault("ask", []).append((str(t), int((lg_c.detach().argmax(-1) == y).sum()), len(items)))\n'
     '        return torch.stack(ces).mean()\n\n'
     '    def loss_items(items, stat=None):\n'
     '        """§262: the ask items (last in the batch) take loss_ask, the rest loss_reg; the item-weighted mean (no ask items: loss_reg exactly)."""\n'
     '        rg = [it for it in items if not it.get("ask")]; ak = [it for it in items if it.get("ask")]\n'
     '        if not ak:\n'
     '            return loss_reg(items, stat)\n'
     '        assert all(it is r for it, r in zip(items, rg)), "§262: the ask items come last"\n'
     '        l_a = loss_ask(ak, stat)\n'
     '        return l_a if not rg else (loss_reg(rg, stat) * len(rg) + l_a * len(ak)) / len(items)\n\n'
     '    s_mc = OW.Stream(SEED_MEMCHECK'),
    ('    acbin = {}                                                              # Note 251-10: {d: [correct, n, ce sum]} (the open 1k bin; not in the resume ckpt)\n',
     '    acbin = {}                                                              # Note 251-10: {d: [correct, n, ce sum]} (the open 1k bin; not in the resume ckpt)\n'
     '    askbin = {}                                                             # §262: {pass: [correct, n]} (the open 1k bin; not in the resume ckpt)\n'),
    ('            stream.rng.bit_generator.state = ck["stream_rng"]; stream.stats = dict(ck["stream_stats"])\n',
     '            stream.rng.bit_generator.state = ck["stream_rng"]; stream.stats = dict(ck["stream_stats"])\n'
     '        if ask_stream is not None:                                          # §262\n'
     '            ask_stream.rng.bit_generator.state = ck["ask_stream_rng"]\n'),
    ('        its = draw_items(d); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; st_ = {}\n',
     '        its = draw_items(d) + (ask_items(d) if NASK else []); assert len(its) == a.bs; opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro; st_ = {}   # §262: + the ask items\n'),
    ('            ab_ = acbin.setdefault(dd_, [0, 0, 0.0]); ab_[0] += c_; ab_[1] += n_; ab_[2] += s_\n',
     '            ab_ = acbin.setdefault(dd_, [0, 0, 0.0]); ab_[0] += c_; ab_[1] += n_; ab_[2] += s_\n'
     '        for tt_, c_, n_ in st_.get("ask", []):                              # §262: ask acc per pass (per 1k bin)\n'
     '            kb_ = askbin.setdefault(tt_, [0, 0]); kb_[0] += c_; kb_[1] += n_\n'),
    ('                br["acue_by_d"] = {k_: {"acc": v_[0] / v_[1], "ce": v_[2] / v_[1], "n": v_[1]} for k_, v_ in sorted(acbin.items())}\n',
     '                br["acue_by_d"] = {k_: {"acc": v_[0] / v_[1], "ce": v_[2] / v_[1], "n": v_[1]} for k_, v_ in sorted(acbin.items())}\n'
     '            if askbin:                                                      # §262 Note 262-2: the ask items\' \':\' acc per pass (full-vocabulary argmax == v_X)\n'
     '                br["ask_acc_by_pass"] = {t_: v_[0] / v_[1] for t_, v_ in sorted(askbin.items(), key=lambda kv: int(kv[0]))}; br["ask_n"] = askbin["1"][1]\n'),
    ('acbin = {}; wsbin = {}\n', 'acbin = {}; wsbin = {}; askbin = {}\n'),
    ('"stream_rng": stream.rng.bit_generator.state if stream is not None else None,',
     '"stream_rng": stream.rng.bit_generator.state if stream is not None else None, "ask_stream_rng": ask_stream.rng.bit_generator.state if ask_stream is not None else None,'),
    ('"nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")),',
     '"nested_words_sha256": sha_file(os.path.join(ROOT, "boxes", "nested_words.py")), "nk_sha256": sha_file(os.path.join(ROOT, "boxes", "nk.py")) if a.nk else None,\n'
     '            "ask_render_sha256": sha_file(os.path.join(ROOT, "boxes", "ask_render.py")) if a.ask_p else None,'),
    (FMT_OLD, '+ AW_FMT[a.write_src] + (" | §262 NK: one name-keyed head packed into S (k = q = normalize(W_n e(name)); write key = the cue\'s name, v = W_vn ln_w(h_11 at \':\'), '
              'delta rule; read at block 6 at name tokens + W_on S_n q, W_on = 0 at init, no fade)" if a.nk else "") + (" | §262 ASK: round(ask_p bs) one-hop ask items per step '
              '(ST_v1 classes after perm p ask_perm_p; QUESTION(u, X) + SFX at every pass; CE(\':\' logits, v_X) at every pass, no decoder / aux CE)" if a.ask_p else "")}')]
open("scratch/s269_build/edits_s269m.py", "w", encoding="utf-8", newline="\n").write('"""§262 NK: inject_aw main() edits (applied by mk_inject_aw.py after E268)."""\nE269 = %r\n' % (E269,))
E269E = [
    ('"stab_fresh", "stab")\nAW_ONLY = (', '"stab_fresh", "stab", "nk_pilot")\nAW_ONLY = ('),
    ('"c7", "halt", "halthead", "frames")\n', '"c7", "halt", "halthead", "frames", "nk_pilot")\n'),
    ('              "stab": lambda: part_stab(m, g, sets, a, dev, bs)}[p_]()',
     '              "stab": lambda: part_stab(m, g, sets, a, dev, bs),\n'
     '              "nk_pilot": lambda: __import__("boxes.nk", fromlist=["part_nk_pilot"]).part_nk_pilot(m, g, sets, a, dev, bs)}[p_]()   # §262 NK_v1 §4')]
open("scratch/s269_build/edits_s269e.py", "w", encoding="utf-8", newline="\n").write('"""§262 NK: inject_aw_eval run() edits (applied by mk_inject_aw_eval.py)."""\nE269E = %r\n' % (E269E,))
m = "scratch/s234_build/mk_inject_aw.py"; t = open(m, encoding="utf-8").read(); a_ = 'assert "tok_st(" not in main and "inject_st_eval" not in main'; assert t.count(a_) == 1 and "E269" not in t
open(m, "w", encoding="utf-8", newline="\n").write(t.replace(a_, 'for o, n in runpy.run_path("scratch/s269_build/edits_s269m.py")["E269"]:   # §262 NK: --nk / --ask_p / perm-then-insert\n'
                                                               '    assert main.count(o) == 1, ("§262 edit not applied exactly once", main.count(o), o[:90]); main = main.replace(o, n)\n' + a_, 1))
me = "scratch/s234_build/mk_inject_aw_eval.py"; t = open(me, encoding="utf-8").read(); b_ = 'parts = open("scratch/s234_build/inject_aw_eval_parts.py", encoding="utf-8").read()\n'
assert t.count(b_) == 1 and "E269E" not in t
open(me, "w", encoding="utf-8", newline="\n").write(t.replace(b_, 'for o, n in __import__("runpy").run_path("scratch/s269_build/edits_s269e.py")["E269E"]:   # §262 NK: the nk_pilot part\n'
                                                                 '    assert body.count(o) == 1, ("§262 eval edit not applied exactly once", body.count(o), o[:90]); body = body.replace(o, n)\n' + b_, 1))
for g_ in ("scratch/s234_build/mk_inject_aw.py", "scratch/s234_build/mk_inject_aw_eval.py"):
    assert subprocess.call([sys.executable, "-B", g_]) == 0, g_
print("§262 NK edits applied")
