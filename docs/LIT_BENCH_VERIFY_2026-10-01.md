# Verification of bench/literature.json (3cbb41e) against the papers (2026-10-01)

Literature session, for Audit. Each value was checked against the paper's table, read from its full text.

**Sources.**
- Coconut 2412.06769, Table 1 (GPT-2; mean ± std over seeds).
- CODI 2502.21074: GPT-2 GSM8k-Aug 43.7% (text and Table 2); CoT-SFT ≈ 44% ("99% of its accuracy").
- Deng et al. 2311.01460, Table 3 (implicit CoT: GPT-2 Small 4×4 0.97 / 5×5 0.10; GPT-2 Medium 0.96 / 0.96).
- Deng et al. 2405.14838, Table 2 (GPT-2 Small: 4×4 / 5×5 / 7×7 / 9×9 — No CoT 0.29 / 0.01 / 0.00 / 0.00; Explicit CoT
  1.00 throughout; ICoT-KD 0.97 / 0.10; ICoT-SI 1.00 / 0.95 / 0.95 / 0.99) and Table 3 (GSM8K, GPT-2 Small: explicit
  0.41, No CoT 0.13, ICoT-KD 0.20, ICoT-SI 0.30).
- bmarti44 Hugging Face model card (not peer-reviewed): ProsQA Coconut 97.0, CoT 83.0, pause-curriculum 96.6,
  pause-multipass 94.8; 7 / 8-hop OOD Coconut 66.0 / 67.5.

**Same items.**
- **ProsQA:** `prosqa/data/prosqa_test.json` is byte-identical to the Coconut repo's `data/prosqa_test.json` (sha256 prefix
  82ddc0afe6ee30eb, 500 items). **Same.**
- **GSM8K:** Coconut, CODI and iCoT use GSM8k-Aug's test file (iCoT repo `data/gsm8k/test.txt`). It has the same 1,319
  problems as openai `test.jsonl`, in the same order, with identical final answers (1,319 / 1,319 question and answer
  matches), only reformatted. **Same items.**
- **Multiplication:** the harness loads the iCoT repo's `data/{n}_by_{n}_mult/test_bigbench.txt` (1,000 items each), the
  papers' test sets. **Same.**
- **Deep ProsQA (bmarti44):** a different generator. **Not the same.**
- ProntoQA is not a harness task; its Table 1 numbers are not entered.

**Corrections.** The placeholder values (ProsQA and GSM8K rows) were all correct. The four multiplication and CODI nulls
are filled. Rows added: Pause Token and Coconut ablations, iCoT GSM8K (30.0, as cited in Coconut), the ICoT-SI/KD rows,
and the bmarti44 8-hop value.

## Addendum (2026-10-01): SIM-CoT GPT-2 GSM8K numbers (for docs/GSM8K_REPRO.md, 25aa883)
- **Values** (SIM-CoT 2509.20317, Table 1, GPT-2, GSM8k-Aug in-domain, i.e. the 1,319-item test above):
  SFT-CoT 42.7; No-CoT 19.1; iCoT 30.1; Coconut 36.6; SIM-CoT (+Coconut) 44.8; CODI 42.0; SIM-CoT (+CODI) 42.6.
  *Note:* their Coconut (36.6) and CODI (42.0) re-runs differ from the original papers' 34.1 and 43.7.
- **Coconut latent setting** (InternLM/SIM-CoT `args/gsm_coconut.yaml`, `gsm_simcot.yaml`, `gsm_simcot_eval.yaml`):
  c_thought 2, max_latent_stage 5, pad_latent_to_max True, so **10 continuous thoughts**; 3 epochs per stage; lr 1e-4.
- **Checkpoint:** the repo's evaluation config loads **`gsm_coconut_simcot/checkpoint_30`**, not checkpoint_28. Confirm
  which file the HF repo's checkpoint corresponds to.
- **CODI:** a fixed implicit length during training (the SIM-CoT text); CODI's GPT-2 default is 6 latents (as in THEORY §13's probe).
