"""Share of SERIAL content (DAG level >= 2: not computable from the question in one level) in each per-pass target, recomputable Aug-train items
(20k sample, rng 7). string: R-operand tokens whose referenced result has level >= 2 (they carry that result's identity), as a share of all target
tokens and of content tokens; value target: steps with level >= 2. Output serial_share.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, "D:/Latent")
import boxes.gsm8k_data as G
from step_stats import char_cats, tok_cats, dag_depth
R, _ = G.load_aug("train"); R = [R[i] for i in np.random.default_rng(7).permutation(len(R))[:20000]]
tot = cont = ser_tok = rop_tok = 0; steps = ser_steps = 0; lvl_hist = {}
for r in R:
    tpl, _ = G.parse_recomputable(r)
    if tpl is None:
        continue
    lv = dag_depth(tpl)
    for t, (e, eqt) in enumerate(zip(r["eqs"], tpl["eqt"])):
        steps += 1; ser_steps += int(lv[t] >= 2); lvl_hist[lv[t]] = lvl_hist.get(lv[t], 0) + 1
        s, cc = char_cats(e, eqt)
        for (a, b, tok, kind, ref) in eqt["ops"]:
            if kind == "r" and lv[ref] >= 2:
                for i in range(a, b):
                    cc[3 + i] = "opd_r2"
        ids, tc = tok_cats(s, cc); tc = tc + ["eot"]
        tot += len(tc); cont += sum(c not in ("fmt", "eot", "eq") for c in tc); ser_tok += sum(c == "opd_r2" for c in tc); rop_tok += sum(c in ("opd_r", "opd_r2") for c in tc)
res = {"steps": steps, "step_level_hist": dict(sorted(lvl_hist.items())), "value_target_serial_step_share": ser_steps / steps,
       "string_R_operand_token_share": rop_tok / tot, "string_serial_token_share_all": ser_tok / tot, "string_serial_token_share_content": ser_tok / cont}
json.dump(res, open("serial_share.json", "w", newline="\n"), indent=1); print(res)
