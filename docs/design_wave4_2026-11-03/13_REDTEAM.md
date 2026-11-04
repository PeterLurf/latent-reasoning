# 13 RED TEAM (independent): INTERP TOOLKIT (doc 13), 2026-11-04

Reviewer: an independent red-team sub-agent, read-only. I read doc 13 in full and checked its code claims against `boxes/deltamem.py` (`_forward` L95-130, `read_fixed`, `write_seq`, `_block`, `gpt_lower`), `boxes/inject_aw.py` (`pass_aw`, `tok_aw`, `tok_ask`, the training loop L366-392, argparse L248-268), `boxes/inject_aw_eval.py` (`passes` L54-81, `swap_pairs` / `part_swap_k` / `part_swap_kd` L217-433, `read_mask` L993-1041), `boxes/sc_parts.py` (`_spans`, `part_attn_fade`), `boxes/deltas5.fwd_read`, and the args.json of `bun_s1`, `of_s1` and `gb_s1`. I also read THEORY §263, §269, §270 and §271 and wave-4 doc 18.

The CPU checks are in `C:\Users\peter\AppData\Local\Temp\claude\D--\ffe8dd4f-366e-5d47-92b5-bf5f7ac71716\scratchpad\rt13\chk.py`. They use synthetic tensors only and touch no stored run outputs.

## Verdict: GO-WITH-FIXES, for a reduced scope

| Piece | Verdict |
|---|---|
| Hook core (`z` / `mlp` / `resid` / `memread` / `memstate`), position classes, the exact gates | GO |
| RB3, with the fixes below | GO |
| I1-lite (head noising + split-half joint5) | GO |
| I1 path / pattern-OV, I3 (b)(c) | Conditional |
| I3 (a) | Merge into existing parts |
| I3 (d) probe / Fourier | NO-GO as specified |
| I4 EAP / EAP-IG | NO-GO as specified |
| TH2 | Move out of the toolkit |

**Target.** The primary target must be **of_s1** (of_s2 confirms), with bun_s1 as the paired comparison. Run the parts as a new section, not "§269b".

**Code claims that check out (no finding).**
- Every trunk path goes through the module attribute `DM._block`: `pass_aw` L50, `fwd_read`, and `gpt_lower` as a module global. Rebinding therefore reaches all three.
- `pass_aw` writes from the ':' row only, and `u = LN_w(h_11)` (write_mix 0/0 in all three args files).
- `passes` computes `m.lower` once per distinct input, so hold passes share `xs[d-1]`. The runner trap is real.
- Per-head state patching is exact. A head's write depends only on `S[:, h]` and u. Checked on a synthetic delta rule: the other heads are `torch.equal`, and the patched head equals the donor run's head (chk.py §4).
- of_s1 has the same architecture flags as bun_s1: cue step, write_src answer, add_norm None, fast_add, cache_lower, think_k 8, d_max 4. Only perm_p 0.5 and ask_p 0.25 differ. The toolkit therefore ports to of_s1 unchanged.

## Findings

**F1 HIGH: the parts target the wrong model for the C1 questions.**
- **Evidence.**
  - bun is order-bound (doc 18 §2.3: "perm ... NOT run on bun, order-bound, ~0.1").
  - of_s1 / of_s2 are the current best and order-free over op sentences (Note 263-3: TEST-PERM d 4 0.982 / 0.961).
  - of_s1 is the init for §271 N1, and of is the paired baseline for §270.
  - §270's open hypothesis is H-B CONJECTURE: "§263 selects by name among positions with the trained op layout". I1 (which heads select) and RB3 (which text spans the pass reads) are the direct tests of H-B, but only on of.
  - On bun, I1 would map a position-bound selection circuit that the lineage has already replaced.
- **Fix.**
  - Run I1 / RB3 on of_s1, confirm on of_s2, and run bun_s1 as the paired comparison (same items, same code).
  - Pre-state the bun-vs-of contrast as the routing read. For example: of's top selection heads attend to CUENAME→name-match tokens while bun's attend by position. That reading routes §270b (all-statement perm) vs a name-keyed fix (MC3).
  - Keep the T-length / class asserts per item, because of items carry `perm`.

**F2 HIGH: on of_s1 the ':' WS readout is a trained readout.**
- **Evidence.**
  - of_s1 has `ask_p 0.25`. `tok_ask` and the training loop (inject_aw.py L99-110, L428) train `CE(colon_logits, v_X)` at every pass on one-hop ask items.
  - bun has `cue_ans 0` and `ask_p None`, so on bun the ':' readout is never trained.
  - I1's NE, RB3's EXPOSED / LEAK and I3's follow / stay all read "the pass-t WS readout". On of, an effect at ':' logits can live in the trained ask pathway (ln_f → unembed) without reaching what the pass writes. This is the "trained readings are artefacts" lesson.
- **Fix.** On of, the binding metric for every part is the downstream effect through memory, in one of two forms:
  - the decoder answer at K' 8; or
  - the next pass's WS (`pass t+1` reading the patched `S_t`), with `u_t` captured and reported as ΔLD.

  The pass-t ':' readout stays descriptive on of. It is binding only on bun, where it is not trained, and even there it is reported next to the downstream metric.

**F3 HIGH: duplication with existing parts and with §269, and a section-name collision.**
- **Evidence.**
  - **Name collision.** Doc 13 says the parts are "queued as §269b". Doc 18 *is* §269b (noise / subs / tta / sclat / noop) and defers I1 / TH2 / RB3 to "§269c" (doc 18 L17).
  - **I1 `mass_step`.** This is §265 `attn_fade` (sc_parts): the same conditions (d 4 t 2-4, d 8 t 4/7), the same first 500 TEST2 items, the same per-layer×head ':' mass on S_t's span, plus the op-share.
  - **I3 (a) memory sweep.** This is `part_swap_kd` SWAP-MID, which already reports follow / stay_own with a **MEAN-at-t floor** (the donors' mean `S_t`). I3 (a) has no such floor, so it misses the control lesson: the control must receive the intervention and be informative. The swap_kd results already exist for of_s1 (`eval_swap_kd.json`). I only listed this file; I did not read it.
  - **I3's routing.** "ALGO-COMPLETE + PASS-INVARIANT → C2 failures are not computational" is what §269 triage (I5) decides directly, and triage COMPUTATION already routes to I3.
  - **I4's routing.** DELTA- / ADD-CARRIED informs D10 / D11 / MC3, the same lanes as §269 keys + alphahold. An exact per-family test also already exists: `mem.ablate ∈ {"add", "delta"}` (deltamem L117-120 and read_fixed).
  - **TH2.** TH2 uses no toolkit hook. It is a behavioural render on `passes()`, and it belongs with the P1 / P2 plan-lane designs (gb models), not in an interp toolkit.
- **Fix.**
  - Give the section a fresh number (or call it §269c, as doc 18 does).
  - Have I1 read `mass_step` from §265's `attn_fade` output rather than recompute it.
  - Replace I3 (a) with the SWAP-MID numbers plus one C-VAL0 sweep cell that carries a MEAN-at-t floor.
  - Make I3 (b)(c) conditional on §269 triage = COMPUTATION.
  - Replace I4's channel question with the exact checks: ablate add / delta / both, and per-head state mean-patch.
  - Move TH2 into a P1-lane design.

**F4 HIGH: I4's EAP completeness gate is ill-posed and double-counts through the state carry.**
- **Evidence.**
  - G-GRAD requires "Σ edge scores into the output ≈ the first-order LD change". With the cross-pass carry edges S_{t-1}[h] → S_t[h], the same ΔS is scored once per later pass.
  - For the additive heads (`S_t = S_{t-1} + phi(k)v^T`, deltamem L105) the carry is the identity. Synthetic check (chk.py §3; linear carry, K 8): ΣΔ(write edges) = ΔM exactly, but Σ(write + carry edges) = **4.98 × ΔM**.
  - Completeness holds per cut, not over the whole edge set. The "≤ 5% edges reach faith 0.8" (SPARSE) and "mem_share" labels inherit the overcount.
  - EAP's linearity is also weakest where it matters: softmax-saturated selection heads (2403.00745) and an 8-step recurrence.
  - The 96 state nodes and 144 heads are cheap to patch exactly. Exact patching needs about 0.7M pass-forwards, which §9 already budgets for the validation set alone.
- **Fix.**
  - Drop EAP / EAP-IG from this design.
  - Use exact activation patching on 12 heads × 8 passes of state, 144 z at the two selection passes, and the MLPs.
  - Re-propose EAP only for a model or scale where exact patching costs more than $5. When it returns, its completeness gate must be per-cut ("the sum over the edges crossing the cut after pass t equals ΔLD to first order"), with a G-ATP gate on that cut.

**F5 MED: I3 (d)'s probe and Fourier reads violate the USE-test lesson and have no null.**
- **Evidence.**
  - The linear-probe accuracy at ':' is the exact pattern of the §203 SVR lesson (decodable is not used). No use test is attached.
  - Synthetic null (chk.py §2). For 11 class means of a 768-d residual with **no value information**, the top-2-of-5 frequency-pair power share is 0.412 (p95 0.420). It is the same with 27 items per class.
  - Any value-ordered smooth code, for example a magnitude direction linear in v, also concentrates power at frequency 1 without any Fourier algorithm.
- **Fix.** Either drop (d) or replace it with a USE read:
  - a DAS / subspace intervention on the ':' residual (2303.02536), with the value subspace fitted on half A and interchange-tested on half B;
  - step-0 (init) validation logged, never keeping best-including-init;
  - the share reported against the label-shuffled null (0.41), with a value-permuted (non-cyclic) control.

**F6 MED: RB3's LEAK has no non-specific-disruption control, and "any class" inflates with depth.**
- **Evidence.**
  - LEAK = P(*any* non-licensed occlusion flips). The number of non-licensed classes grows with d and with distractor count, so LEAK rises mechanically with op count. That confounds the registered prediction "the footprint spreads with op count".
  - Knockout removes softmax mass and redistributes it (to EOT and other tokens), so a long span can flip a near-tie readout without being read.
  - `knock` runs on the manual attention path, which is not SDPA-bitwise: fp32 max abs 5.4e-7 on CPU, chk.py §5. Comparing knocked-manual against intact-SDPA therefore lets rounding flip near-tie argmaxes.
  - PREV-flip is read as "recompute from text", but the PREV sentence also introduces S_{t-1}'s name, which the STEP sentence references. A PREV flip may be name binding, not value recompute.
- **Fix.**
  1. Make the intact reference the `knock(∅)` manual run, not SDPA.
  2. Add a null span: one neutral length-matched filler sentence rendered into each item and knocked. Its flip rate is the floor, and LEAK is reported per class minus the floor.
  3. Report the per-class rates. The "any" share is descriptive only, and the label uses the per-class maximum with a Bonferroni-adjusted Wilson bound.
  4. Add the exact arm **TEXT-VALUE-EDIT**:
     - build pass t's input from a text in which PREV's (or an OTHEROPS / GIV) operand is changed;
     - take S_{t-1} from the clean run (`passes(seqs_of_t=…, edit=…)`; bitwise, no manual attention);
     - read follow-text vs follow-state at the downstream metric (F2).

     This is the USE-type test of text recompute, and Note 203-6 INPUT-SWAP is its precedent. Make it binding for SELECT-WITH-LEAKAGE, with knockout as supporting evidence.
  5. Cite 2304.14767 (attention knockout, "Dissecting Recall").

**F7 MED: the fp64 gate cannot run as written.**
- **Evidence.**
  - `read_fixed`, `write_seq` and `_forward` hard-cast with `x.float()` and `S.float()` (deltamem L97-99 and the read / write bodies).
  - On a `.double()` model, `mem.ln(x.float())` raises `RuntimeError: mixed dtype ... expect parameter to have scalar type of Float` (chk.py §1).
  - G-EQ1's "fp64 CPU on 20 items: rel ≤ 1e-9" on end-to-end logits therefore crashes, or, if the model is only partly cast, it measures an fp32 floor.
- **Fix.**
  - State the fp64 gate per trunk block: the manual pattern / knock(∅) / qkv_split vs `_block` with fp64 weights. On CPU I measured rel 3.1e-16 for manual attention vs SDPA (chk.py §5), so the 1e-9 bar is easy.
  - Keep end-to-end logits for the fp32 tolerance arm only.

**F8 MED: the bitwise gates need the deterministic wrap and a pinned source.**
- **Evidence.**
  - Memory rule: default SDPA kernels are not run-to-run bitwise on GPU. G-EQ0 / G-EQ2 assert `torch.equal` on the host.
  - The `z` / `mlp` hooks re-implement `_block`'s body inside the wrapper. A copied body silently diverges if deltamem changes (the copied-module lesson).
  - `IP.hooks` installs `m.pass_aw` / `m.dec_logits` instance attributes, as `read_mask` does. `read_mask` deletes them in `finally`, so nesting the two context managers silently strips the outer wrapper.
- **Fix.**
  - Run G-EQ0 / G-EQ2 and every part under the same det wrap. Gates certify only the kernels that the parts use.
  - Record `sha256(deltamem.py)`, assert it at part start, and re-run G-EQ2 on any change.
  - Assert in `IP.hooks` that there is no pre-existing instance `pass_aw` / `dec_logits`.

**F9 MED: missing positive controls for the patching machinery.**
- **Evidence.** All the gates are identity or no-op checks. None shows that a patch moves the output when it should.
- **Fix.** Add G-POS, with two parts:
  - `IP.patch_state(all 12 heads)` at pass t on `swap_pairs_d` pairs must reproduce `part_swap_kd` SWAP-MID answers **bitwise** on the same pairs and seeds (a known non-zero effect).
  - Zeroing head (l, h)'s z must equal a model whose `c_proj.weight[64h:64h+64, :]` rows are zeroed, bitwise on CPU in fp32.

**F10 MED: normalisation and means for the state and mean-ablation edits.**
- **Evidence.**
  - Additive state grows every pass, including hold passes (`Sa += phi(k)v^T` at ':' each pass).
  - I3 (c) "mean-ablate the rest of blocks 6-11 at COLON" and the `memstate` mean op do not say which mean they use.
  - A mean pooled over passes or positions is off-distribution for additive heads at late t. That is the perturbation-normalisation lesson.
- **Fix.**
  - Every mean is taken at the same pass t, position class and d over the part's own items, as swap_kd does.
  - Any per-head state edit that is not a donor swap reports `||ΔS_h|| / ||S_h||`, matching the scale used by NC (`nc_norms`).
  - For the I4-replacement per-family ablations, report add-only, delta-only and both. If both kills while the singles are ~0, the label is REDUNDANT, not "channel unused" (the ablation-untested-channel lesson).

**F11 MED: most labels do not route, and MIXED, the most likely outcome, routes nothing.**
- **Evidence.**
  - I1 routes only on LOCALIZED + PATTERN + DILUTED (OC: 0.55 × 0.6 × 0.4 ≈ 0.13) or DISTRIBUTED.
  - D1b / FD4 is already being decided by §265 (DENSE-DISTRACT + attn_fade).
  - I3 and I4 route to lanes that §269 triage, keys and alphahold already route.
  - TH2's routing (P1 vs P2) is real but is not interp.
  - RB3's SELECT-WITH-LEAKAGE is the only label that changes a claim's wording (C1).
- **Fix.** Before build, each surviving label must name the one registration or lane it changes, and say what MIXED does: either "no change, logged" or a named tie-break. Proposed:
  - **RB3 / TEXT-VALUE-EDIT on of.** CLEAN-SELECT keeps the C1 hop-by-hop wording for of. LEAKAGE rewords it and routes a read-only-text-at-hold / cue-row-cached arm (E3' / CC).
  - **I1 on of vs bun.** Name-match heads (CUENAME→STEP_SUBJ) route §270b / FP. Position heads route MC3 / NK name-keyed write.
  - Drop the I1 depth (d 8) arm. It is C2, and §265 owns it.

**F12 LOW: NE ratio and censoring.**
- **Evidence.** Per-item NE = (LD_patch − LD_clean) / (LD_corr − LD_clean) explodes when the denominator is near zero. "Readouts differ" is an argmax filter and does not bound LD.
- **Fix.**
  - Aggregate as a ratio of sums with a paired bootstrap CI.
  - Pre-state a denominator floor: items with |LD_corr − LD_clean| < 0.5 nats leave, and their count is reported. The INSUFFICIENT rule (< 30% kept) then applies to this filter as well.

**F13 LOW: the C-VAL0 follow target is mis-defined.**
- **Evidence.** "follow = P(readout = op_t(v'_{t-1}, c_t))" is wrong for `add_s` (which needs v'_j) and for *_v ops (which need G_t).
- **Fix.**
  - Use follow target = the donor item's own `values[t]` from `OW.solve`. The donor and recipient differ only in S_0's number, so this is exact for every op.
  - Drop the pairs whose donor v'_t equals the recipient's v_t. This covers the coefficient-0 mod 11 and mul cases.

**F14 LOW: math-backing tags and citations.**
- **Evidence.** The user 10-13 rule, and §270 / §271 practice, tag each claim PROVED / DERIVED / CONJECTURE. Doc 13 has no tags.
- **Fix.** Tag:
  - per-head state-patch exactness: PROVED from `write_seq` (each head's update uses only `S[:, h]` and u), and verified synthetically;
  - TH2's PAIR-BOTH ≤ QSENS: PROVED, given v^A ≠ v^B;
  - EAP linearity: CONJECTURE (moot after F4);
  - the knockout "licensed set": DEFINITION.

**F15 LOW: position-class details.**
- The decoder pass has no CUE / CUENAME, so the tiling assert must use the decoder's own class set.
- The CUE span "L-11..L-5" depends on the unit's token count. Derive it per item from `len(enc(QUESTION))`; do not hard-code it.
- Of the cited line numbers, `read_mask` is at L993-1041 (not 992-1040). This is trivial.

**F16 LOW: the cost is fine; the risk is build labour.**
- §9's ≈ $1.1 (guard $2) is far below $30. Even with the of + bun pairing and the null-span arm, the staged plan below is under $1.5.
- The [ASSUMED] 2 ms per pass-forward must be replaced by the G-TIME line, as §9 already says.

**On C7 vs C1.** As written, the toolkit is justified mainly by C7 groundwork (the mechanism figure), which is the lowest priority. The parts that serve C1 are:
- RB3 with TEXT-VALUE-EDIT: "step-specific blocking shows serial, hop-by-hop use", and whether the pass reads only its licensed text;
- I1 on of: H-B for §270.

Those parts should be built first. I3, I4 and the d 8 arms serve C2 and C7 and should wait.

## Minimal staged plan I would approve

**Stage 0: build and gates ($0; CPU + one host gate line, ~$0.05).**
- Build `boxes/interp.py` with these hooks only: `z`, `mlp`, `resid`, `memread`, `memstate`, `write_u`, and `knock` (manual path).
- Add the position classes.
- No EAP and no `qkv_split`.
- Gates:
  - G-EQ0 and G-EQ2, under the det wrap with the deltamem sha pinned (F8);
  - G-LIN;
  - G-LOWER;
  - G-CLS (decoder class set, per-item cue span);
  - G-POS (F9: SWAP-MID bitwise reproduction and the zeroed-c_proj equivalence);
  - G-EQ1 per block in fp64 (F7) and end-to-end in fp32.
- CPU smoke at full item scale on of_s1's item sets.
- The scorer `score_ip.py` with fixtures, including a failing fixture for each validity gate.

**Stage 1: C1 reads on of_s1 (+ of_s2 confirmation, bun_s1 paired), ~$0.5, guard $1.**
- **RB3' at d 2-4.**
  - TEXT-VALUE-EDIT is binding (exact; PREV / OTHEROPS / GIV operand edits with the state held).
  - Knockout per class, against the knock(∅) reference and the null-span floor.
  - The downstream metric is binding (decoder K' 8 and pass t+1 WS); the pass-t ':' readout is descriptive on of (F2).
  - The label routes C1 wording for of, and E3' / CC.
- **I1-lite at d 4 t 2-4.**
  - 144-head noising at COLON / CUENAME / ALL under C-SEL.
  - Split-half joint5, with the ratio-of-sums NE (F12).
  - Read `attn_fade`'s mass (no recompute).
  - The label routes §270b vs MC3 / NK by of-vs-bun head type (F1, F11).

**Stage 2: conditional only.**
- I1 path patching and the pattern / OV split, only if Stage 1 says LOCALIZED on of_s1 and of_s2.
- I3 (b)(c) on blocks 6-11, only if §269 triage = COMPUTATION, with pass-matched means (F10).
- The memory-channel read as exact ablation (add / delta / both) plus 96-node per-head state patching. No EAP.
- TH2 is moved to a P1-lane design on gb.
- I3 (d) is dropped, unless it is re-proposed as DAS with a step-0 log and the 0.41 null (F5).
- I4 EAP is deferred until exact patching costs more than $5 (F4).
