"""THEORY §271 N1 SHUFFLED-OBJECTS tokenization (N1_v1 §3.3-3.4): shared by boxes/inject_so.py (training) and boxes/inject_so_eval.py (eval). No tok_st (F14).
tok_so(it, cue, aux_target='value', write_src='answer') -> the inject_aw item dict:
  think       [EOT] + enc(' ' + text)                    (text ends with the question "At the end, who <verb> <o*>?")
  dec         think + SFX (' The answer is:')            (the decoder / DIRECT input)
  think_cues  pass t = 1..m: [EOT] + enc(' ' + text + CUE(it, t)) + SFX, CUE = so_words.cue_si / cue_en / cue_generic ('index' / 'entity' / 'generic');
              passes t > m read the pass-m input (cue_inputs' min(t, d)); every input of an item has one length (shared padding, asserted)
  ans         VAL index of S_m (NAMES_SO.index); training CE is full-vocabulary on VAL_IDS[ans]
  aux / vidx  aux_t = enc(' ' + R_step[t]) + EOT, R_step[t] = "After swap <L(t)>, <S_t> <verb> <o*>."; vidx_t = the index of S_t's name token (asserted unique, F5)
  people_ids  the item's n name tokens (eval readouts: pick = argmax over these); d = m; sig; raw = the generator item.
VAL_IDS = the 38 NAMES_SO tokens (replaces NUM_IDS as the value vocabulary)."""
import numpy as np
import torch

import boxes.so_words as SO
from boxes.inject_ow import ENC, SFX
from boxes.nested170 import EOT

COLON = SFX[-1]
NAME_TOK = {n: ENC.encode(" " + n)[0] for n in SO.NAMES_SO}
assert all(len(ENC.encode(" " + n)) == 1 for n in SO.NAMES_SO), "NAMES_SO: single tokens"
VAL_IDS = np.array([NAME_TOK[n] for n in SO.NAMES_SO])
CUES = {"index": SO.cue_si, "entity": SO.cue_en, "generic": SO.cue_generic}


def val_index(name):
    return SO.NAMES_SO.index(name)


def tok_so(it, cue="index", aux_target="value", write_src="answer"):
    assert aux_target == "value" and write_src == "answer" and cue in CUES, ("tok_so", cue, aux_target, write_src)
    th = [EOT] + ENC.encode(" " + it["text"]); m = it["m"]
    aux = [ENC.encode(" " + r) + [EOT] for r in it["R_step"]]; vidx = []
    for t, q in enumerate(aux, 1):
        tk = NAME_TOK[it["values"][t]]; hit = [i for i, x in enumerate(q) if x == tk]
        assert len(hit) == 1, ("F5: S_t's name token not unique in R_step[t]", t, it["R_step"][t - 1]); vidx.append(hit[0])
    cues = [[EOT] + ENC.encode(" " + it["text"] + CUES[cue](it, t)) + SFX for t in range(1, m + 1)]
    assert all(q[:len(th)] == th and q[-1] == COLON for q in cues) and len({len(q) for q in cues}) == 1, ("G2: cue rows / shared padding", cue)
    return {"think": th, "dec": th + SFX, "think_cues": cues, "ans": val_index(it["answer"]), "aux": aux, "vidx": vidx, "d": m, "sig": it["sig"],
            "people_ids": [NAME_TOK[p] for p in it["people"]], "raw": it, "cue": cue}


def pick(lg, items):
    """per item the argmax of lg (N, vocab) over that item's own people_ids -> the picked name's VAL index (np.array)."""
    out = []
    for i, it in enumerate(items):
        ids = it["people_ids"]; j = int(torch.as_tensor(lg[i])[ids].argmax()); out.append(val_index(it["raw"]["people"][j]))
    return np.array(out)
