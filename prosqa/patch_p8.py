"""PROTOCOL8 training arms for the official Coconut code (apply after patch_coconut.py and patch_mask.py; idempotent).

    python patch_p8.py /workspace/coconut

Adds two config options, both off by default (the standard recipe is unchanged when they are absent):

* `anytime: C` (int). Anytime / minimum-progress supervision (THEORY §9/§18; theory/anytime_loss.py anytime_ce).
  - During training, after the ordinary forward, for every latent step j < K of each example: run a branch where
    <|end-latent|> and the answer tokens follow latent j directly.
  - The branch reuses the final pass's key/value cache. Its attention is limited to <|start-latent|> .. latent j,
    plus the branch itself (the addendum-G answer mask applies as in the ordinary pass).
  - It adds the answer's cross-entropy for examples whose required depth (number of gold steps) is <= j + C.
  - The ordinary loss (all tokens after the latents, including the answer after latent K) is unchanged. The added term
    is the mean over supervised (example, j) pairs of the per-example mean answer-token CE, weight 1.
  - Depth comes from the training JSON (`len(steps)`, looked up by the dataset's `idx`).
* `random_tight: True`. Random tight budgets (THEORY §22.7 rule).
  - At every latent curriculum stage s >= 1, each example gets K ~ U{1..min(s, max_latent_stage)} latents.
  - K is drawn from a hash of (per-call salt, idx), so it is independent of the example's content and length.
  - The first K steps are replaced by K latents; explicit steps resume at step K + 1 (none if K >= n).

`test_p8.py` checks both on a tiny random GPT-2. The anytime branch at j = K reproduces the ordinary pass's answer
logits, and with no supervised pairs the loss equals the ordinary loss.
"""
import os
import sys

d = sys.argv[1]
MARK = "# PROTOCOL8 arms"

# ---------------------------------------------------------------- coconut.py: anytime branch loss
cp = os.path.join(d, "coconut.py")
src = open(cp).read()
if MARK not in src:
    assert "PROTOCOL6-G mask" in src, "apply patch_mask.py first"
    src = src.replace('''        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1''', '''        logits.append(outputs.logits)
        kv_final = outputs.past_key_values                     # PROTOCOL8 arms: full-sequence cache for anytime

        self.gen_forward_cnt += max_n_latents + 1''', 1)
    src = src.replace('''        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
''', '''        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if (getattr(self, "anytime_C", None) is not None and self.base_causallm.training
                and kwargs.get("sample_idx") is not None and max_n_latents > 1):
            at = self._anytime_loss(input_ids, attention_mask, position_ids, latent_lists, kv_final,
                                    kwargs["sample_idx"])
            if at is not None:
                loss = loss + at
''', 1)
    method = f'''
    {MARK}
    DEPTH = {{}}                                   # idx -> number of gold steps (set by run.py from the train JSON)

    def _anytime_loss(self, input_ids, attention_mask, position_ids, latent_lists, kv, sample_idx, only_j=None,
                      return_logits=False):
        """Mean answer CE of branches (<|end-latent|> + answer right after latent j) over examples with depth <= j + C."""
        B, T = input_ids.shape
        dev = input_ids.device
        C = self.anytime_C if self.anytime_C is not None else 0
        start = (input_ids == self.start_latent_id).float().argmax(1)
        ans = []
        for b in range(B):
            last = int(attention_mask[b].nonzero().max())
            hits = (input_ids[b, : last + 1] == ANS_ID).nonzero()
            ans.append(input_ids[b, int(hits.max()): last + 1] if len(hits) else None)
        depth = [Coconut.DEPTH.get(int(i), 10 ** 6) for i in sample_idx]
        Kmax = max(len(l) for l in latent_lists)
        js = [only_j] if only_j is not None else range(1, Kmax)
        terms, keep = [], {{}}
        for j in js:
            sel = [b for b in range(B) if len(latent_lists[b]) >= j and ans[b] is not None
                   and (only_j is not None or (len(latent_lists[b]) > j and depth[b] <= j + C))]
            if not sel:
                continue
            A = 1 + max(len(ans[b]) for b in sel)
            n = len(sel)
            ids = torch.full((n, A), self.eos_token_id, dtype=torch.long, device=dev)
            pos = torch.zeros((n, A), dtype=torch.long, device=dev)
            extra = torch.full((n, 1, T + A, T + A), float("-inf"), device=dev)
            tgt = torch.full((n, A), -100, dtype=torch.long, device=dev)
            for r, b in enumerate(sel):
                La = len(ans[b])
                lj = latent_lists[b][j - 1]
                ids[r, 0] = self.end_latent_id
                ids[r, 1: 1 + La] = ans[b]
                pos[r] = position_ids[b, lj] + 1 + torch.arange(A, device=dev)
                extra[r, 0, T:, int(start[b]): lj + 1] = 0.0
                extra[r, 0, T:, T:] = torch.full((A, A), float("-inf"), device=dev).triu(1)
                tgt[r, :La] = ans[b]
            emb = self.embedding(ids)
            pkv = [(k[sel], v[sel]) for k, v in kv]
            prev = MASK.extra
            MASK.extra = extra
            out = self.base_causallm(inputs_embeds=emb, past_key_values=pkv, position_ids=pos,
                                     attention_mask=torch.ones((n, T + A), dtype=attention_mask.dtype, device=dev))
            MASK.extra = prev
            lg = out.logits
            if return_logits:
                keep[j] = (sel, lg)
            ce = torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt, ignore_index=-100, reduction="none")
            m = (tgt != -100).float()
            terms.append((ce * m).sum(1) / m.sum(1))
        if return_logits:
            return keep
        return torch.cat(terms).mean() if terms else None
'''
    src = src.replace("    def train(self):", method + "\n    def train(self):", 1)
    open(cp, "w").write(src)
    print("P8 anytime patch applied to coconut.py")
else:
    print("P8 anytime patch already in coconut.py")

# ---------------------------------------------------------------- dataset.py: random tight budgets
dp = os.path.join(d, "dataset.py")
src = open(dp).read()
if MARK not in src:
    src = src.replace('''def get_cot_latent_dataset(''', f'''{MARK}
_P8_SALT = [0]


def get_cot_latent_dataset(''', 1)
    src = src.replace('''    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
''', '''    n_additional_tokens = 0 if no_special_marker else 2
    _P8_SALT[0] = random.getrandbits(32)          # PROTOCOL8 random_tight: fresh draws on every call (every epoch)
    salt = _P8_SALT[0]

    def process_dataset(sample):
''', 1)
    src = src.replace('''        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0
''', '''        if getattr(configs, "random_tight", False) and scheduled_stage_to_train >= 1 and not configs.no_cot:
            K_s = min(scheduled_stage_to_train, configs.max_latent_stage)
            K = random.Random(hash((salt, int(sample["idx"])))).randint(1, K_s)
            n_skip_steps, n_latent_tokens = K, K

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0
''', 1)
    assert "random_tight" in src and "import random" in src
    open(dp, "w").write(src)
    print("P8 random_tight patch applied to dataset.py")
else:
    print("P8 random_tight patch already in dataset.py")

# ---------------------------------------------------------------- run.py: pass idx, depth table, flags
rp = os.path.join(d, "run.py")
src = open(rp).read()
if MARK not in src:
    src = src.replace('''        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)''',
                      f'''        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        {MARK}
        model.anytime_C = getattr(configs, "anytime", None)
        Coconut.DEPTH = {{i: len(x["steps"]) for i, x in enumerate(json.load(open(configs.train_path)))}}''', 1)
    old = '''                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps'''
    assert src.count(old) == 1, "training-loop anchor not found exactly once"
    src = src.replace(old, '''                sample_idx = batch["idx"].tolist()   # PROTOCOL8 arms: depth lookup for anytime
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch, sample_idx=sample_idx)

                loss = outputs.loss / configs.gradient_accumulation_steps''', 1)
    if "import json" not in src:
        src = "import json\n" + src
    open(rp, "w").write(src)
    print("P8 patch applied to run.py")
else:
    print("P8 patch already in run.py")
