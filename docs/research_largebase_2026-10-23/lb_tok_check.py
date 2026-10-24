"""LB draft check A ($0, CPU, 1 thread): how each candidate base tokenizes the §226 TOPO word programs.

Reads only tokenizer.json files (data; tokenizers.Tokenizer.from_file, no remote code) and boxes/ow_words.py (read-only import).
GPT-2 / GPT-2 medium / GPT-2 large share tiktoken "gpt2" (the repo's encoder).

Checks, per base:
  num   - the token pieces of each value 0..10 in " has <v> cards." and after " The answer is:"; single-token answers; prefix collisions
          between classes (e.g. "1" is a prefix of "10" when digits are split), which break a one-position 11-way read.
  names - the 40 NAMES: tokens of " <Name>" (single-token share), and of " <Name>'s".
  aux   - the aux prompt " Step <t> :" pieces for t = 1..8.
  len   - think length (tokens of [BOS] + " " + text), value-target and R_t (cumulative) target lengths, by d = 1..8 (mean and max over
          256 TOPO items per d from an independent seed), and the ratio to GPT-2 (feeds lb_cost_model.py).
Usage: python -B lb_tok_check.py --tok qwen=<dir>/tokenizer.json --tok llama32=<dir>/tokenizer.json ... --out lb_tok_check.json
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("RAYON_NUM_THREADS", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import boxes.ow_words as OW  # noqa: E402

SEED_LEN = 23101            # independent of every registered OW seed (22900-22997); length statistics only
N_PER_D = 256


class Enc:
    def __init__(self, name, path):
        self.name = name
        if path == "tiktoken:gpt2":
            import tiktoken
            e = tiktoken.get_encoding("gpt2"); self._enc = e.encode; self._dec = lambda ids: e.decode(ids); self.sha = "tiktoken-gpt2"
        else:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(path); self._enc = lambda s: t.encode(s, add_special_tokens=False).ids; self._dec = lambda ids: t.decode(ids)
            self.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            self._tok = t

    def enc(self, s):
        return list(self._enc(s))

    def pieces(self, s):
        return [self._dec([i]) for i in self.enc(s)]


def num_check(E):
    out = {}; first = {}
    for v in range(11):
        ctx = E.enc(" has"); full = E.enc(" has %d cards" % v); tail = E.enc(" cards")
        mid = full[len(ctx):len(full) - len(tail)] if full[:len(ctx)] == ctx and full[-len(tail):] == tail else None
        ans_ctx = E.enc(" The answer is:"); ans_full = E.enc(" The answer is: %d" % v)
        ans = ans_full[len(ans_ctx):] if ans_full[:len(ans_ctx)] == ans_ctx else None
        out[v] = {"in_sentence": None if mid is None else [E._dec([i]) for i in mid], "n_in_sentence": None if mid is None else len(mid),
                  "after_suffix": None if ans is None else [E._dec([i]) for i in ans], "n_after_suffix": None if ans is None else len(ans)}
        if ans is not None:
            first[v] = ans
    # one-position read: the answer token is the first NON-space piece after the suffix; collisions = two classes sharing it
    def key(ids):
        ps = [E._dec([i]) for i in ids]; k = 0
        while k < len(ids) - 1 and ps[k].strip() == "":
            k += 1
        return ids[k], ids[k + 1:]
    keys = {v: key(first[v]) for v in first}
    coll = [[v, w] for v in keys for w in keys if v < w and keys[v][0] == keys[w][0]]
    single = all(len(keys[v][1]) == 0 for v in keys)
    lead_space = sorted({len(first[v]) - 1 - len(keys[v][1]) for v in first})
    return {"by_value": out, "answer_single_token_after_space": single, "first_token_collisions": coll, "separate_space_tokens_before_answer": lead_space}


def names_check(E):
    one = [n for n in OW.NAMES if len(E.enc(" " + n)) == 1]; poss = {n: len(E.enc(" " + n + "'s")) for n in OW.NAMES}
    return {"n_names": len(OW.NAMES), "single_token": len(one), "multi_token": [n for n in OW.NAMES if n not in one],
            "possessive_len_hist": {str(k): sum(1 for v in poss.values() if v == k) for k in sorted(set(poss.values()))}}


def aux_check(E):
    return {t: E.pieces(" Step %d :" % t) for t in range(1, 9)}


def len_check(E, items):
    out = {}
    for d, its in items.items():
        th = [1 + len(E.enc(" " + it["text"])) for it in its]
        val = [len(E.enc(" " + r)) + 1 for it in its for r in it["R_step"]]
        cum = [len(E.enc(" " + it["R"][t])) + 1 for it in its for t in range(1, d + 1)]
        cot = [len(E.enc(" " + it["text"] + " " + it["cot"] + " The answer is %d." % it["answer"])) + 1 for it in its]
        out[d] = {"think_mean": float(np.mean(th)), "think_max": int(max(th)), "value_tgt_mean": float(np.mean(val)), "value_tgt_max": int(max(val)),
                  "cum_tgt_mean": float(np.mean(cum)), "cum_tgt_max": int(max(cum)), "text_cot_mean": float(np.mean(cot)), "text_cot_max": int(max(cot))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--tok", action="append", default=[], help="name=path/to/tokenizer.json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb_tok_check.json")); a = ap.parse_args(argv)
    encs = [Enc("gpt2", "tiktoken:gpt2")] + [Enc(*s.split("=", 1)) for s in a.tok]
    st = OW.Stream(SEED_LEN, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); items = {d: [st.draw(d) for _ in range(N_PER_D)] for d in range(1, 9)}
    res = {"seed_len": SEED_LEN, "n_per_d": N_PER_D, "ow_words_sha256": hashlib.sha256(open(os.path.join(ROOT, "boxes", "ow_words.py"), "rb").read()).hexdigest(), "bases": {}}
    for E in encs:
        r = {"tokenizer_sha256": E.sha, "num": num_check(E), "names": names_check(E), "aux_prompt": aux_check(E), "len": len_check(E, items)}
        res["bases"][E.name] = r
    g = res["bases"]["gpt2"]["len"]
    for n, r in res["bases"].items():
        r["len_ratio_vs_gpt2"] = {d: {k: round(r["len"][d][k] / g[d][k], 3) for k in ("think_mean", "think_max", "value_tgt_mean", "cum_tgt_mean", "text_cot_mean")} for d in g}
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=False); f.write("\n")
    for n, r in res["bases"].items():
        nm = r["num"]
        print("%-10s answer_single=%s collisions=%s space_toks=%s names_single=%d/%d think_mean d1/d4/d8 = %.1f/%.1f/%.1f  ratio d4 %.2f" % (
            n, nm["answer_single_token_after_space"], nm["first_token_collisions"], nm["separate_space_tokens_before_answer"], r["names"]["single_token"],
            r["names"]["n_names"], r["len"][1]["think_mean"], r["len"][4]["think_mean"], r["len"][8]["think_mean"], r["len_ratio_vs_gpt2"][4]["think_mean"]))
        print("           ' 10' in sentence:", nm["by_value"][10]["in_sentence"], "| after suffix:", nm["by_value"][10]["after_suffix"], "| aux t=1:", r["aux_prompt"][1])


if __name__ == "__main__":
    main()
