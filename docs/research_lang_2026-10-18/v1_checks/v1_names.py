"""LANG v1 $0: candidate name sets for ProsQA-SYM v1 (RT F2-4, F2-5; PROOFS 2a).
  PERSON-40 : single-GPT-2-token person names (leading space), ENT17's 15 one-token names first (sigma-symmetric T; PREMISE-SWAP keeps
              every position).
  NONCE-300 : lowercase pseudo-words of 2-3 GPT-2 tokens (leading space) whose FINAL tokens are pairwise distinct, never a non-final
              token of any other name, never a person token, never a template token; no name is a single token (no common word).
A feasibility build for the spec (Experiments builds the frozen list in boxes/prosqa_sym.py; sha recorded there).
    python -B v1_names.py   -> v1_names.json
"""
import hashlib
import json
import os
import random

import tiktoken

enc = tiktoken.get_encoding("gpt2")
HERE = os.path.dirname(os.path.abspath(__file__))
ENT17 = ("Alex", "Bob", "Carol", "Davis", "Eva", "Fae", "Gabriel", "Jack", "Max", "Oliver", "Polly", "Rex", "Sally", "Sam", "Stella", "Tom", "Wren")
MORE = ("Anna", "Ben", "Chris", "Dan", "Emma", "Frank", "George", "Henry", "Ian", "Jane", "Kate", "Leo", "Mark", "Nick", "Paul", "Ray", "Rose",
        "Ruth", "Sara", "Tim", "Victor", "Will", "Adam", "Amy", "Brian", "Carl", "David", "Eric", "Fred", "Gary", "Helen", "Jim", "Joe", "Karen",
        "Lisa", "Mary", "Mike", "Nancy", "Peter", "Steve", "Tony", "Walter")
TEMPLATE = {enc.encode(s)[0] for s in (" is", " a", " Every", "Every", ".", " or", "?", " Is", "###", " ###")}


def ntok(w):
    return enc.encode(" " + w)


persons = [p for p in ENT17 if len(ntok(p)) == 1] + [p for p in MORE if len(ntok(p)) == 1 and p not in ENT17]
persons = persons[:40]
ptoks = {ntok(p)[0] for p in persons}
ONS = ["b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z", "br", "dr", "gl", "gr", "kl", "pr", "sk", "sl", "sn",
       "st", "tr", "vr", "zh", "ch", "sh", "th", "qu", "fl", "pl"]
VOW = ["a", "e", "i", "o", "u", "ai", "ou", "ee", "oo", "ia"]
COD = ["", "", "", "n", "m", "r", "l", "s", "x", "k", "nt", "rk", "sp", "lm", "nd", "st"]
BANNED = ("shit", "fuck", "cunt", "nig", "fag", "rape", "porn", "sex", "cock", "dick", "slut", "whor", "ass", "tit", "cum", "piss", "kill",
          "nazi", "jew", "gay", "hell", "damn", "bitch", "anal", "anus", "rap", "kkk", "isis", "hate")
rng = random.Random(23000)
chosen, finals, nonfinal = [], set(), set()
tries = 0
while len(chosen) < 300 and tries < 400000:
    tries += 1
    w = "".join(rng.choice(ONS) + rng.choice(VOW) + rng.choice(COD) for _ in range(rng.randint(2, 3)))
    if not w.isalpha() or not w.islower() or w in chosen or any(b in w for b in BANNED):
        continue
    t = ntok(w)
    if not (2 <= len(t) <= 3) or (len(t) == 3 and tries < 200000):   # prefer 2-token names (shorter T); 3-token only to fill
        continue
    f, nf = t[-1], set(t[:-1])
    if f in finals or f in nonfinal or f in ptoks or f in TEMPLATE or (nf & finals) or (nf & ptoks):
        continue
    chosen.append(w); finals.add(f); nonfinal |= nf
lens = [len(ntok(w)) for w in chosen]
out = {"persons": persons, "n_persons": len(persons), "person_tokens_all_1": all(len(ntok(p)) == 1 for p in persons),
       "nonce": chosen, "n_nonce": len(chosen), "tries": tries, "nonce_tokens_hist": {k: lens.count(k) for k in sorted(set(lens))},
       "nonce_tokens_mean": round(sum(lens) / len(lens), 3), "final_tokens_distinct": len({ntok(w)[-1] for w in chosen}) == len(chosen),
       "final_disjoint_from_nonfinal": not ({ntok(w)[-1] for w in chosen} & {t for w in chosen for t in ntok(w)[:-1]}),
       "sha256_nonce": hashlib.sha256("\n".join(chosen).encode()).hexdigest(), "sha256_persons": hashlib.sha256("\n".join(persons).encode()).hexdigest()}
assert out["final_tokens_distinct"] and out["final_disjoint_from_nonfinal"] and out["person_tokens_all_1"] and out["n_nonce"] == 300
print({k: v for k, v in out.items() if k not in ("nonce",)})
json.dump(out, open(os.path.join(HERE, "v1_names.json"), "w", newline="\n"), indent=1)
