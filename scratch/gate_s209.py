"""§209 gates for boxes/codi_port.py (THEORY §209 / note 209-1). -> scratch/gate_s209/gate.json   python scratch/gate_s209.py [g1|rest]
  g1    the port's eval path (released zen-E/CODI-gpt2 loaded into the port's LoRA structure -> merged_fp32 -> codi_probe.run, cuda:1, bs 16) == the bench's
        §13 probe items (bench/results/gsm8k__codi_L6.json, same card / bs), per item (correct + answer text)
  tok   tiktoken gpt2 == CODI's GPT2Tokenizer on 2,000 training strings (question / CoT / answer)
  g2    per-term gradflow on 8 items (fp32 CPU, dropout off): student CE, teacher CE, distill each -> nonzero LoRA grad; teacher CE -> zero prj grad; the
        distill targets carry no grad (sg); student CE -> nonzero prj grad (latents)
  g3    LoRA trainable-param count / target modules / shapes == the released checkpoint's; prj param count == released
  g4    latents autoregressive: perturbing latent 1 changes latents 2..6 (and the decoder logits); a 200-step smoke is run separately (scratch/s209_smoke.sh)"""
import json
import os
import sys

os.environ.setdefault("BOXES_NB", "5")
sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.codi_port as CP  # noqa: E402

REL = "prosqa/ckpt_ext/hf/codi/pytorch_model.bin"; OUT = "scratch/gate_s209/gate.json"


def load_released(m, prj):
    sd = torch.load(REL, map_location="cpu", weights_only=False); pre = "codi.base_model.model."; mine = dict(m.named_parameters()); n_set = 0
    for k, v in sd.items():
        if k.startswith("prj."):
            continue
        k2 = k[len(pre):].replace(".base_layer.", ".base.").replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        if k2 in mine:
            with torch.no_grad():
                mine[k2].copy_(v.to(mine[k2].dtype)); n_set += 1
    prj.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("prj.")}); return n_set, sd


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"; OUTD = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if which in ("g1", "all"):
        m, prj = CP.build("cuda:1"); n_set, _ = load_released(m, prj); m.eval(); prj.eval(); cp = CP.cp_mod(); data = cp.gsm8k(1319)
        acc, by, rows = CP.evaluate(m, prj, data, "cuda:1", 6, 16); b = json.load(open("bench/results/gsm8k__codi_L6.json"))["items"]
        txt = lambda p: str(int(p)) if p != float("inf") and float(p).is_integer() else str(p)  # noqa: E731
        ce = sum(int(r["ok"]) == int(x["correct"]) for r, x in zip(rows, b)); te = sum(txt(r["pred"]) == x["text"] for r, x in zip(rows, b))
        OUTD["g1"] = {"acc": acc, "bench_acc": sum(x["correct"] for x in b) / len(b), "correct_equal": ce, "text_equal": te, "n": len(b), "params_loaded": n_set,
                      "card": torch.cuda.get_device_name("cuda:1"), "bs": 16, "pass": ce == te == len(b)}
        print("g1", OUTD["g1"], flush=True); del m
    if which in ("rest", "all"):
        import tiktoken
        enc = tiktoken.get_encoding("gpt2"); tok = CP.codi_tok(); D, drop = CP.load_items("valid", enc)
        import boxes.gsm8k_data as G
        R, _ = G.load_aug("train", filter_bad=False); S = []
        for r in R[:700]:
            body = r["_line"].split("||", 1)[1]; c, a_ = body.rsplit("####", 1); S += [r["q"], " ".join(c.strip().split(" ")[:-1]), "The answer is: " + a_.strip()]
        bad = sum(enc.encode(s) != tok(s)["input_ids"] for s in S[:2000]); OUTD["tok"] = {"n": min(2000, len(S)), "mismatch": bad, "pass": bad == 0}
        # g3
        m, prj = CP.build("cpu"); TP = CP.trainable(m, prj); _, sd = load_released(m, prj)
        rel_lora = {k: tuple(v.shape) for k, v in sd.items() if ".lora_" in k}; my_lora = {n: tuple(p_.shape) for n, p_ in TP if ".lora_" in n}
        rel_t = sorted({k.split(".")[-4] for k in rel_lora}); my_t = sorted({n.split(".")[-3] for n in my_lora})
        OUTD["g3"] = {"lora_params": sum(p_.numel() for n, p_ in TP if ".lora_" in n), "released_lora_params": sum(int(torch.tensor(s).prod()) for s in rel_lora.values()),
                      "lora_modules": len(my_lora) // 2, "released_lora_modules": len(rel_lora) // 2, "targets": my_t, "released_targets": rel_t,
                      "shapes_equal": sorted(my_lora.values()) == sorted(rel_lora.values()), "prj_params": sum(p_.numel() for n, p_ in TP if n.startswith("prj.")),
                      "released_prj_params": sum(v.numel() for k, v in sd.items() if k.startswith("prj.")), "trainable_total": sum(p_.numel() for _, p_ in TP)}
        g3 = OUTD["g3"]; g3["pass"] = g3["lora_params"] == g3["released_lora_params"] and g3["shapes_equal"] and g3["targets"] == g3["released_targets"] and g3["prj_params"] == g3["released_prj_params"]
        # g2 (fresh port model, fp32 CPU, dropout off: deterministic per term)
        m, prj = CP.build("cpu"); m = m.float(); m.eval(); prj.eval()
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, CP.LoRAConv1D):
                    mod.lora_B.weight.normal_(0, 1e-3)                      # B = 0 at init gives A no grad; perturb (the gradflow-gate convention)
        TP = CP.trainable(m, prj); b = CP.collate(D[:8], "cpu"); res = {}

        def gnorm(sel):
            v = [p_.grad.double().norm() for n, p_ in TP if sel(n) and p_.grad is not None]; return float(torch.stack(v).norm()) if v else 0.0
        for term in ("ce", "distill", "ref_ce"):
            for _, p_ in TP:
                p_.grad = None
            store = {}
            orig = CP.F.smooth_l1_loss

            def spy(a_, b_, *k, **kw):
                store.setdefault("tgt_req_grad", []).append(bool(b_.requires_grad)); return orig(a_, b_, *k, **kw)
            CP.F.smooth_l1_loss = spy
            _, parts_ = CP.codi_loss(m, prj, b, 6, 2.0); CP.F.smooth_l1_loss = orig
            # recompute the single term with graph
            enc_, em, ref, rl, dec, dl, rp, mp = b
            L, pt = CP.codi_loss(m, prj, b, 6, 2.0)
            # isolate a term by zeroing the others through their factors: rerun pieces
            for _, p_ in TP:
                p_.grad = None
            if term == "ref_ce":
                rog = m(input_ids=ref, output_hidden_states=True); rlg = rog.logits[:, :-1]
                t_ = CP.F.cross_entropy(rlg.reshape(-1, rlg.size(-1)), rl[:, 1:].reshape(-1), ignore_index=-100) * 2.0
            else:
                o = m(input_ids=enc_, attention_mask=em, use_cache=True, output_hidden_states=True); pkv = o.past_key_values; lat = prj(o.hidden_states[-1][:, -1:, :])
                for _ in range(6):
                    o = m(inputs_embeds=lat, use_cache=True, output_hidden_states=True, past_key_values=pkv); pkv = o.past_key_values; lat = prj(o.hidden_states[-1][:, -1:, :])
                so = m(inputs_embeds=m.transformer.wte(dec), use_cache=True, output_hidden_states=True, past_key_values=pkv)
                if term == "ce":
                    lg = so.logits[:, :-1]; t_ = CP.F.cross_entropy(lg.reshape(-1, lg.size(-1)), dl[:, 1:].reshape(-1), ignore_index=-100)
                else:
                    with torch.no_grad():
                        ro = m(input_ids=ref, output_hidden_states=True)
                    ix = lambda h, pos: h.gather(1, pos[:, None, None].expand(-1, 1, h.size(-1)))  # noqa: E731
                    t_ = sum(CP.F.smooth_l1_loss(ix(o_, mp), ix(r_, rp).detach()) / ix(r_, rp).std() for o_, r_ in zip(so.hidden_states, ro.hidden_states)) / len(so.hidden_states)
                    store["teacher_hidden_req_grad"] = any(h.requires_grad for h in ro.hidden_states)
                    assert abs(float(t_) - pt["distill"]) < 1e-4 * max(1, abs(pt["distill"])), (float(t_), pt["distill"])
            t_.backward(); res[term] = {"lora": gnorm(lambda n: ".lora_" in n), "prj": gnorm(lambda n: n.startswith("prj.")), **store}
        res["pass"] = (res["ce"]["lora"] > 0 and res["ce"]["prj"] > 0 and res["distill"]["lora"] > 0 and res["ref_ce"]["lora"] > 0 and res["ref_ce"]["prj"] == 0
                       and res["distill"].get("teacher_hidden_req_grad") is False and not any(res["distill"].get("tgt_req_grad", [True])))
        OUTD["g2"] = res
        # g4 autoregressive latents
        with torch.no_grad():
            o = m(input_ids=enc_, attention_mask=em, use_cache=True, output_hidden_states=True); lat0 = prj(o.hidden_states[-1][:, -1:, :])

            def roll(first, pkv0):
                pkv = pkv0; lat = first; L_ = []
                for _ in range(6):
                    o2 = m(inputs_embeds=lat, use_cache=True, output_hidden_states=True, past_key_values=pkv); pkv = o2.past_key_values; lat = prj(o2.hidden_states[-1][:, -1:, :]); L_.append(lat)
                return L_
            import copy
            A = roll(lat0, copy.deepcopy(o.past_key_values)); B = roll(lat0 + 0.5 * torch.randn_like(lat0), copy.deepcopy(o.past_key_values))
        OUTD["g4_autoregressive"] = {"latent_diffs": [float((x - y).abs().max()) for x, y in zip(A, B)], "pass": all(float((x - y).abs().max()) > 0 for x, y in zip(A, B))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(OUTD, open(OUT, "w", newline="\n"), indent=1, default=str); print(json.dumps(OUTD, default=str)[:3000])


if __name__ == "__main__":
    main()
