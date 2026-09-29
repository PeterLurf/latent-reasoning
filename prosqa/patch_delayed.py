"""PROTOCOL6 addendum J: delayed-query Coconut on ProsQA. Apply after patch_coconut.py and patch_mask.py (idempotent).

    python patch_delayed.py /workspace/coconut

dataset.py: a sample's optional "query" (see make_delayed.py) is tokenized as " <query>\\n" and inserted right after
<|end-latent|> in the training sequences (labels -100: it is input, not target) and in the evaluation prompts.
coconut.py (_build_extra from patch_mask.py): besides <|end-latent|> and the answer ("###" onwards), the query tokens
and the newline after its "?" (token 30) may not see anything before <|start-latent|>. Otherwise the position right
after the query could read both the statements and the candidates and hand the answer on (a new bypass). Written
steps (early curriculum stages only) stay open, as in addendum G. Generation: every position after <|end-latent|>.
"""
import os
import sys

d = sys.argv[1]
MARK = "# PROTOCOL6-J delayed query"

dp = os.path.join(d, "dataset.py")
src = open(dp).read()
if MARK not in src:
    old = '''        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
'''
    assert old in src
    src = src.replace(old, old + f'''        {MARK}
        query_tokenized = (tokenizer.encode(" " + sample["query"] + "\\n", add_special_tokens=False)
                           if sample.get("query") else [])
''', 1)
    old = '''            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],'''
    assert old in src
    src = src.replace(old, '''            "answer_tokenized": answer_tokenized,
            "query_tokenized": query_tokenized,
            "idx": sample["idx"],''', 1)
    # evaluation prompt: question + <start> + latents + <end> + query
    old = '''            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )
'''
    assert old in src
    src = src.replace(old, '''            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
            + sample["query_tokenized"]
        )
''', 1)
    # training sequence: ... <end> + query + remaining steps + answer; the query is not a target
    old = '''            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )'''
    assert old in src
    src = src.replace(old, '''            + ([] if no_special_marker else [end_id])
            + sample["query_tokenized"]
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )''', 1)
    old = '''                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],'''
    assert old in src
    src = src.replace(old, '''                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
                + len(sample["query_tokenized"])
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"])
                + len(sample["query_tokenized"]) :
            ],''', 1)
    open(dp, "w").write(src)
    print("J: dataset.py patched (query after <|end-latent|>)")
else:
    print("J: dataset.py already patched")

cp = os.path.join(d, "coconut.py")
src = open(cp).read()
if MARK not in src:
    assert "# PROTOCOL6-G mask" in src, "apply patch_mask.py first"
    old = '''    blockq = (ar[None, :] >= qfrom[:, None]) | (ar[None, :] == endp[:, None])'''
    assert old in src
    src = src.replace(old, f'''    {MARK}: the query (up to its "?", token 30) and the newline after it are blocked too
    isq = (input_ids == 30) & (ar[None, :] > endp[:, None])
    qm = torch.where(isq.any(1), isq.float().argmax(1), endp)
    blockq = ((ar[None, :] >= qfrom[:, None]) | (ar[None, :] == endp[:, None])
              | ((ar[None, :] >= endp[:, None]) & (ar[None, :] <= qm[:, None] + 1)))''', 1)
    open(cp, "w").write(src)
    print("J: coconut.py mask extended to the query")
else:
    print("J: coconut.py already patched")
