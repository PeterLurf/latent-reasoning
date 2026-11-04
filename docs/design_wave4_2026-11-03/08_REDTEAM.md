Status: independent red team of 08_c4_onset_drift_pilots.md, 2026-11-04. Read-only: code re-read in the working tree, args.json of ao_s1-s4 / aw_plaw2_s1 read, one CPU toy in the scratchpad (rt08/toy.py). No run outputs were computed on.

# 08 RED TEAM: C4 onset-drift pilots (ctrl / SG / ANC / LPFT)

## Verdict: GO-WITH-FIXES

The question is well posed. The harness audit is correct: the H0 aux weight is exactly 0 but in the graph, and the fix works. Stage 1 is cheap and is read only where Note 269-1 FD1 allows (drift of an already-present code). Most code citations check out.

Three problems would make the binding readings unsafe as drafted:
- ANC is scored on the very readout it trains.
- SG and LPFT leave a trainable path into S_1 (the write projections, plus W_o's bias at every pass). ws_readout cannot see that path.
- The seed-2 SG cell has no paired control. The seed it is compared with (ao_s2) is the one AO seed that later formed.

All three can be fixed for about $0.3-0.6 more and one small build item.

## Citations verified (2026-11-04 working tree)

These match the doc:
- inject_aw: pass_aw 47-56, colon_logits 61-63, build_aw 109-125, ws_readout 134-151.
- inject_aw argparse 211-249; assert aux_step > 0 at 259; HOLD 289; lam_at 299-302.
- inject_aw loss_reg 361-419: aux block 394-403, la 405, aux 406, loss 407.
- inject_aw: WS_ITEMS 451; make_opt call 508; resume 514-528; g_add calibration 529-541; fade 549; gamma 550; clip 566; opt.step 570; ws log 632-633; torch.save 659-663; stop_at exit 671-672; keep_final rename 686-687.
- inject_ow: tok_ow 63-71; dec_logits 86-90.
- deltamem: read_fixed 158-176; write_seq 179-223.
- fsq120: make_opt 731-737; lr_at 740-743.
- ow_words line 70 (d 1 = mul_v only) and the add_s skip at 72-74.
- inject_aw_eval: part_probe 313-349.
- score_em: wilson 48-50, near 53-54.
- gradflow_check: term 26-28.
- theory/statlib.py does not exist yet; the doc says "once it lands", which is fine.

ao_s1 args match H0: aux_step 1.0, aux_fade "0,0", write_mix 0/0, d_max 2, micro 1 (auto_micro true), excl_n 160598, module 461458f5, RTX 3090, steps 12000. The lam_at algebra is right: with f0 = f1 = 0 and step >= 1, lam = 1.0 * max(0, -step / 1) = 0.0.

## Findings

### 1. HIGH: ANC is scored on the readout it trains, so its NO-DRIFT holds by construction

**Evidence.**
- The ANC student logits are `m.colon_logits(h, pos)[:, NUM_IDS]` at pass 1 on d 2 items (sec. 4.3).
- The binding DRIFT field ws_readout['2']['1'] is `m.colon_logits(h, pos)[:, num].argmax(-1)` at pass 1 on d 2 items (inject_aw 146).
- That is the same function on the same pass and the same depth. ANC is a KL on the logits whose argmax is the metric, toward a teacher that is about 1.000 correct on v_1.
- P3 = 0.85 is therefore close to a tautology, not a test (memories "trained readings are artefacts" and "dynamics targets get gamed"). M5 and risk 4 admit that the readout is held and the write source may not be, but the binding label is still put on the readout.

**Fix.**
- For anc_s1, demote ws_readout['2']['1'] to a manipulation check (it must be >= 0.95, else ANC failed to apply).
- Make ANC's binding stage-1 read the untrained linear probes at 4k: part_probe's u1|v1 (LN_w(h11[':']), the write source) and read1|v1 (the pre-W_o read from S_1). These are fitted post hoc on a separate stream (OW.Stream SEED_PROBE, 3,000 d 2 items, sig-split), and no loss term touches them. This needs finding 2's stop-eval hook.
- NO-DRIFT(ANC) := u1|v1 >= 0.80 and read1|v1 >= 0.80 (Wilson on the test split, n ~600, NEAR rule).

### 2. HIGH: SG and LPFT leave trainable routes into S_1 and into the pass-1 hidden state that ws_readout does not see

**Evidence (code).**
- The write maps u_1 to S_1 through mem.ln_w, the k/v rows of mem.qkv and mem.gate (deltamem 186-191). sg_write detaches only the trunk's ':' rows (sec. 4.2), so these mem parameters still get the full pass-1 write gradient d la / d S_1.
- Pass 1 reads the zero state: add_read and the delta read are 0, so read_fixed adds `fade * wo(0) = fade * wo.bias` to every token at block 6 (deltamem 176; wo is nn.Linear with a bias, 70-72). wo.bias is trained by every other pass and by the decoder read.

**Evidence (toy).** A 2-pass toy with a shared trunk f, write projection P, W_o bias and a decoder on S_2 (scratchpad rt08/toy.py). With SG on the pass-1 write source:
- the grad on P is unchanged (0.519 vs 0.519);
- the trunk grad is 0.342 vs 0.546 live, still nonzero via pass 2;
- the wo.bias grad is 0.043 vs 0.082, still nonzero.

**Consequences.**
- (a) SG does not "remove P-direct exactly". It removes the trunk part of P-direct. The projection part can still rewrite S_1 toward v_2-predictive features of u_1, because u_1 is h11 at ':' on the full text. That is exactly the formation-relevant failure, and ws_readout (the LM-head argmax on h11) is blind to it.
- (b) P-shared has to include the mem read parameters (wo.bias, plus mem.ln and qkv's q rows at passes >= 2). Sec. 3's "same trunk weights" undercounts it.
- (c) lpft_s1's 1k / 2k rows are not "trivially 1.0" (sec. 6). During the freeze the mem group trains at full lr, and wo.bias moves pass 1's hidden state. Those rows measure mem-only drift, which is a useful localiser in its own right.
- (d) A ws_readout drop can also be LM-head drift, because the decoder trains the tied wte / LM head that ws_readout reads, while the code itself stays intact. Note 251-9's addendum separated these two cases only with the probe.

**Fix.**
- Add a default-off `--stop_eval probe,ws` hook. When `step == stop_at`, run the listed eval parts on the live model before `return 9` (inject_aw 671). Gate it with G0 (off = bitwise) and G-smoke.
- Report u1|v1 and read1|v1 at 4k for every stage-1 cell. They are binding for ANC (finding 1). For ctrl / SG / LPFT they are a co-read: a cell is NO-DRIFT only if ws_readout passes and read1|v1 >= 0.80. If ws passes but read1 does not, it is labelled **STORE-DRIFT** and is not resumed under R1's wording.
- Reword R1: "the trunk-side direct path carries the readout drift". Add an R1' row: SG NO-DRIFT on ws but STORE-DRIFT on read1 means the drift relocates to the write projections.
- Cost: one probe pass of 3,000 d 2 items, about 1-2 min per cell.

### 3. HIGH: sg_s2 has no paired control, and the AO seed it is compared with is the one that later formed

**Evidence.**
- ctrl_s1 is the only control (seed 1). sg_s2 is compared with ao_s2 "approximately" (different module and EXCL, sec. 11).
- Note 251-9's 1k drift list is ao_s1 / s3 / s4 / aex0_s1 = 0.22 / 0.09 / 0.16 / 0.27. **ao_s2 is not in it**, and its final pass-1 WS is 0.65, the exception.
- The "AO 4/4" basis for P1 is therefore unverified for seed 2. If seed 2's data order drifts weakly anyway, a NO-DRIFT sg_s2 is uninterpretable, yet R1 (2 / 2) counts it.

**Fix.**
- Add ctrl_s2 (expected +$0.29, ceiling +$0.46). It is exactly paired with sg_s2 (same d sequence, stream and DeltaMem init).
- R1 then needs ctrl DRIFT in both seeds. If ctrl_s2 is not DRIFT, sg_s2 is VOID-PAIR and R1 cannot pass.
- Also pre-state ao_s2's ws_readout['2']['1'] at 1k / 2k / 4k from its existing log in the stage-0a $0 read (pushed, Audit ack first).

### 4. MED: ANC's C4 status. It meets Note 269-4(d) cond. 1 in letter, but the target is v_1 by proxy

**Evidence.**
- 269-4(d) cond. 1 permits "an earlier checkpoint".
- Here, though, that checkpoint (askname) got its pass-1 competence from value-labelled one-hop training (CE on v_X; the same law as §262 ask_p). It emits v_1 at about 1.000 (G-AW0), and the design's own ANC-TGT guard voids the arm below 0.95.
- A KL to a near-one-hot teacher at 1.000 accuracy is, in effect, CE on v_1 at the pass-1 ':' of d 2 items: a per-pass value target on the seeded pass. Note 266-1 ruled K3 out because prefix answers are "per-step targets by proxy". The same argument applies here.
- PAR's earlier checkpoint is a checkpoint of the same answer-only training, not of a value-supervised one.
- Note 251-9's consolidated finding already says per-pass targets on the seeded passes let a new depth form from answers: AEX 2/2 at d 4 accuracy (Note 251-9), EXTEND 1/4 (Note 255-1). So question (b) under ANC is partly answered already.

**Fix.**
- Have U1 ruled before build. Red-team reading: ANC should carry no C4 tier, not even "C4-SD".
- Keep ANC as the **positive control for question (b)**: "with pass 1 held, can pass 2 form from answers in this harness by 12k?" That is what makes an SG NOT-FORMED interpretable. If anc_s1_r also fails to form, R5's reading is "holding is not sufficient" no matter what SG does.
- Remove "C4-SD" from the sec. 8 wording unless Audit rules otherwise.

### 5. MED: SG is not self-defeating for formation, but stage 2 should also test release

**Analysis (question 2).**
- With the trunk source detached, pass 2 can still form. G-AW0 shows v_1 linearly decodable from u_1 at init (u1|v1 1.000). The write projections (ln_w, k/v rows, gate) and the whole pass-2 reader (trunk at pass 2, read_fixed q / W_o) remain trainable (toy: P and trunk grads nonzero).
- So pass 2 can learn to store and read v_1 through the projections without the source changing.
- What SG forbids is adapting u_1 to pass 2's needs. Under P-shared, u_1 still moves, but with no gradient toward usefulness.
- A permanent SG is the clean mechanism test. As a recipe it is the stricter case, and R5 pre-names SG-release without running it.

**Fix.** If an SG cell is NO-DRIFT, stage 2 also resumes **sg_s1_rel** from the same ckpt_4000.pt with sg_write 0. It is exactly paired with sg_s1_r and is a one-flag diff. Its 5k / 6k ws and read1 rows show whether the drift force persists once the cut is lifted, and its 12k label tests the recipe. Cost: +$0.58 expected, within the $4.0 guard if anc_s1_r and lpft_s1_r are capped at the 4-cell limit; sec. 12 should state the priority order (sg_r, sg_rel, anc_r, lpft_r).

### 6. MED: LPFT at 0.1x cannot discriminate at stage 1 and confounds two changes

**Evidence.**
- M6 itself concedes that DRIFT vs "slower" is ambiguous at 4k. After unfreeze, 2k steps at 0.1x lr equal roughly 200 full-lr steps. ctrl already drifts by 1k.
- The trunk's Adam moments also keep accumulating during the freeze: AdamW updates exp_avg / exp_avg_sq even at lr 0. So at unfreeze there is no fresh-moment shock. The arm thus removes the onset shock and lowers the lr at once.

**Fix.**
- Use `--trunk_lr_mult 1.0` (the single change is freeze-until-2000 plus the existing 300-step warm_from ramp). Read the 3k / 4k rows against ctrl's 1k / 2k.
- This is a direct onset-shock test: the trunk meets the answer gradient at full lr, after the mem has adapted, with warmed moments.
- Report the 1k / 2k rows as mem-only drift (finding 2c).
- Alternatively drop LPFT; it is the lowest-information cell.

### 7. MED: censoring a ctrl cell as DRIFT validates the harness

**Evidence.** Sec. 10 scores a stage-1 cell with a second failure as DRIFT, "the unfavourable label". For ctrl, though, DRIFT is the label that validates the family (sec. 8, ctrl validity).

**Fix.** A ctrl cell with a missing binding row after one rerun is **VOID-HARNESS**, and no localisation claim follows. Add a scorer fixture for it.

### 8. MED: ao_s3 / ao_s4 are not 12k comparators

**Evidence.**
- Their args.json have steps 18000, so their cosine runs over 18k. Their eval_* files are 18k evals.
- At 12k their lr is about 0.25x base, not about 0.
- Sec. 2 computes their s/step as train_sec "over 12k". If train_sec covers 18k steps, the rates are about 0.51 / 0.56 s/step, not 0.77 / 0.85. The 1.65 ceiling covers either case.

**Fix.**
- Name only ao_s1 (12k) as an exact-schedule comparator. Label s2 / s3 / s4 as "18k-schedule logs at the 12k bin, descriptive".
- Restate M7's "0 / 4 by 12k" as log-bin based.
- Take the rate basis from train_summary last_step.

### 9. LOW: the c4_noaux fix is sound, but the gate text is self-contradictory and one log changes

**Checks.**
- The finding is correct: lam = 0.0 exactly, and the term is in the graph.
- Before the fix, the only channel by which R_step could affect training was a non-finite aux skipping a step (inject_aw 567-568).
- The toy shows 0 * aux, in-graph zero, and no aux give bitwise-identical grads. The flag is default-off, so default behaviour is preserved by construction, and G0 / G-C4a are the right gates.

**Problems.**
- G-C4a asserts `aux.requires_grad is False`, but sec. 3 sets `aux := la * 0.0` in-graph, which does require grad. Assert instead that no node in loss.grad_fn's walk descends from aux_logits.
- `stat["aux"] = float(aux.detach())` (inject_aw 411) would log 0, so aux_ce changes. Keep the no_grad CE in a separate variable for the logs.

### 10. LOW: gate details

- G-ANC: L_anc ~ 0 at step 0 because wo is zero-initialised (deltamem 70-72), not because of fade, which is already 1/500 at step 1.
- G-ANC: L_anc does reach wo.bias (pass 1's read of S_0 is wo.bias). The list of zero-grad mem parameters should name only the qkv k/v rows, gate and ln_w.
- G-SG: after a detach, the retain_grad on pass-1 h11[':'] is None, not 0.
- Build g_0 after build_aw, and assert the DeltaMem init equals ctrl's (the teacher load must not consume torch RNG before the mem is built).

### 11. LOW: priority relative to §267

- Nothing is shared with §267: no compute, no cells, no module.
- Stage 0 + 1 (about $1.7 with the fixes) do not need §267's result. Stage 2 does not either, since R5 already routes to §267.
- The only real constraint is that this build must not delay §267's (U6, same lane-B owner); §267 is the registered C4 route.
- Decisiveness: stage 1 is a solid mechanism localiser. Only an SG AO-FORMED (P7b 0.40, single seed) would open a new C4 route, and it would need stage 3 (3 seeds in total) before any rate wording. That matches the design.

### 12. LOW: OC, tags and seeds are otherwise fine

- The OC simulation reproduces the stated rule.
- The NEAR dead zone at 0.80-0.88 is conservative, which is acceptable.
- Math tags are honest: M3 is CONJECTURE, M2 is assumption-named.
- M2's claim that d la / d v_1 != 0 regardless of pass 2's use is correct for the fast_add heads: S_a,K is the sum of the writes.
- Formation claims are single-seed existence checks with stage 3 pre-named, which is right.
- Guards change with the fixes (see below).

## Minimal staged plan I would approve

| Stage | Content | Expected | Guard |
|---|---|---|---|
| 0a ($0) | Pre-state, push, Audit ack, then read: AO ws_op1 split and v_2 share; ao_s2 ws['2']['1'] at 1k / 2k / 4k (finding 3). | $0 | - |
| 0 | Build sg_write, anc, freeze_trunk_until (mult 1.0), c4_noaux, `--stop_eval`, ws_op1 / ws_bleed. Gates G0, G-C4a (fixed assert), G-C4b, G-SG, G-ANC (fixed lists), G-LPFT, G-mem, G-smoke, G-resume, G-host. U1 ruled before build. | $0.25 | $0.5 |
| 1 | ctrl_s1, ctrl_s2, sg_s1, sg_s2, anc_s1, lpft_s1 (mult 1.0, or dropped). 4k, with probe + ws at 4k. Binding: ws DRIFT label plus read1|v1 co-read (STORE-DRIFT); ANC binding on the probes only. ctrl censoring = VOID-HARNESS. | 6 x $0.30 = $1.8 | **$3.0** |
| 2 | Only if an SG seed is NO-DRIFT on both ws and read1. In priority order: sg_sX_r (SG on), sg_s1_rel (SG off from 4k, same ckpt), anc_s1_r (positive control for (b)), lpft_s1_r. At most 4 cells, 12k, §244 binding set + FORM-WS2. | <= 4 x $0.58 | **$4.0** |
| 1b / 3 | As drafted (1b only on R2; 3 registered separately). | | $1.2 / $3.0 |

Family guard: 0 + 1 + 1b + 2 = **$8.7**, within the $30 auto-approve.
