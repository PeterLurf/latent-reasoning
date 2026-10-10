# Separators and latent carry: verifying 2610.06605, and per-digit separator slots under GPT-2 BPE (2026-10-08)

Literature session, for Audit (the request) and Theory (the lead against §152's carry-resolution failure).
- **ID verified** via the arXiv abstract page (2026-10-08).
- **Tokenisation:** checked locally with GPT2TokenizerFast.

## 2610.06605: what is verified
- **Title:** "Separators Make Carry Propagation Learnable: The Geometry of Latent Carry in a Multiplication Transformer".
- **Authors:** Satariyan, Cousin (SCAI), Biau (LPSM / IUF). Submitted 2026-10-05. Abstract-level reading only.
- **Setup:** **small Llama-style transformers trained from scratch** (not a pretrained LM) on **4×4 multiplication, no CoT**.
- **The headline is verified:** inserting a **space token between digits** raises exact match **from 1% to 89%**.
- Output positions are learned in carry-chain order, with the **middle digits learned last**. That's the same pattern as Bai
  2510.00184 and our §137 / §140 middle-column wall.
- **Mechanism:** the separator token that predicts each digit (its "prediction slot") **encodes the carry-in as an angle on a ring**
  in the residual stream. Examples with more distinct carry values fill more of the ring.
- **The patching number needs qualifying:**
  - Patching the prediction slot alone (between examples matched on the column sum) transfers the source carry in **up to 84% of
    cases after block 4, for *one* middle column of the *best* model**.
  - For other columns the carry is **first assembled at the neighbouring answer slot** before reaching its own slot.
  - So "≤ 84%" is a maximum over one column and one model, not a typical value.
- **Remaining errors are almost always off by one** (a small error in the carry, or in the circular digit code).

## How it relates to what we have
- **The same mechanism family:**
  - Fourier / circular digit codes in pretrained LMs (2406.03445, 2502.00873, 2410.11781);
  - the clock representation for modular addition (2301.05217).
  - The ring-angle carry is that geometry applied to the carry.
- **Why a separator helps: an extra computation position per output digit.** The carry is computed at the slot *before* the digit is
  emitted. That's a targeted pause token: Goyal 2310.02226 (pause tokens help mainly when also used in pretraining); Pfau
  2404.15758 (filler tokens help only on some tasks, with specific training).
- **Format effects on arithmetic:** Shen 2311.14737 (positional description and format decide multiplication generalisation); Lee
  2307.03381; Singh & Strouse 2402.14903 (separator / tokenisation direction changes arithmetic accuracy in frontier LLMs).

## Per-digit separator slots under GPT-2 BPE (checked locally)
- **Our bench format " 5 6 9 4" tokenises as one token per digit with the space merged:** [' 5', ' 6', ' 9', ' 4'] (ids 642, 718,
  860, 604). **There is no separate separator position.** The digit token itself is the only slot, as Audit suspected.
- **Formats that do give a separate slot per digit:**
  - " 5 , 6 , 9" → [' 5', ' ,', ' 6', ' ,', ' 9']
  - " 5| 6| 9" → [' 5', '|', ' 6', '|', ' 9']
  - "5,6,9,4" → ['5', ',', '6', ',', ...]
  - " 5 _ 6 _ 9" → [' 5', ' _', …]
  - **Any of these, or a dedicated new special token, gives the paper's "prediction slot".** Prefer an existing GPT-2 token (' ,' or
    '|') over a new embedding, so the slot starts from a pretrained representation.
- **Precedent for separator slots specifically in GPT-2-style BPE answer formats: not found.**
  - The paper is char-level-style from scratch.
  - Pause / filler work uses generic tokens, not one per digit.

## For §152 / §148 (Theory's lead)
- **The test is cheap:** add one separator slot per answer digit to the answer decode, starting from the §148 FINAL baseline. It
  changes only the output format; the carrier is unchanged.
- **Predicted effect:**
  - it helps carry *resolution at readout* (§152's 4-row failure, where column sums must become digits);
  - it doesn't help carriage across passes.
- **Controls:**
  - (a) the same number of extra positions placed *before* the answer, not interleaved, to separate "a slot per digit" from "more
    compute";
  - (b) the bench format without separators.
- **Diagnostic to borrow:** a ring-angle probe of the carry-in at each separator slot (fit cos / sin of 2π·carry / C), and the
  paper's column-sum-matched slot patching as a C1-style check at readout.
- **Caveat:** the 1% → 89% gain is from scratch. A pretrained GPT-2's digit geometry may already partly give it, so expect a smaller
  gain.

## Addendum (2026-10-10): our pretrained-GPT-2 replication is null
- §156 ran SEP with ' ,' slots on pretrained GPT-2, 4×4, 24k steps: **0.002 / 0.004** (PAUSE 0.002), with the same column profile as PLAIN.
- The from-scratch 1% → 89% of 2610.06605 doesn't transfer to pretrained GPT-2 at our budget. It's consistent with 2604.06374 (pretraining biases
  late layers) and Goyal 2310.02226 (fine-tune-only extra positions give lukewarm gains).
- INTERLEAVE cites §156 as the comparator instead of a new SEP cell.
