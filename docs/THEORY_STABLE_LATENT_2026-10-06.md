# A stable latent method: design synthesis (2026-10-06) — candidate 1 registered as §134; helper's data in

The user's directive (via Audit): "ask literature and helper to work together with theory to invent and synthesize a stable latent
method". Theory leads. Literature maps the prior art (4da2f672 / e0909d0b / ebc2a039); the Audit helper mined our raw runs (report 2026-10-06 15:10, `scratchpad/report_early_separators.md` in its session).
**Goal:** one task-agnostic method that is
(1) **stable to train** (enters without restarts),
(2) **stable to run** (contractive, exact over long chains; C6 / C2),
(3) **causally serial** (C1), and
(4) **expressive** (large or superposed states: the multiplication / ProsQA path),
with the same code and settings across families.

## What we know (one line each; details in THEORY)
* **The narrow FSQ-8 carrier works once entered.** §128 TASK-AGNOSTIC: 5 families; held-out families 12/12 slots with no
  restart; every learner has ρ₁ 0.03–0.27.
* **Entry is the weak point, on S5 only:** 2 restarts in 4 slots, and 1 accepted run escaped without learning.
* **Wide or continuous carriers drift.** RAW-H 768-d: ρ_ff 0.5–0.8, 0/4 exact. §119's published latent methods: 0/9 learned.
* **Exception, with per-step supervision:** §118 LONG s81–83 (Audit-reproduced) is a **wide, unrounded** latent that stays exact
  with no snap at test, at 512 and 2,048 (ext 1.0 / 1.0 / 0.998), and is SERIAL.
  - Its carried 768-d latents have effective rank 149 / 197 / 173, above S5's 120 states.
  - It uses §111's reader (per-step state labels), the TF bootstrap, training snaps every k_train, and K-CURR to 32.
  - So **a high-dimensional latent can be stable without rounding**. Here the attractor comes from label-supervised reading plus a
    long curriculum (not C4).
  - Candidate 2 asks whether the model's own decode can take the reader's place, label-free. Comparators: §133's CONT-64 / SOFT-64
    (FINAL-only) and §118 LONG (labelled).
* **Entry fixes escape without learning** (§127: STD2 / LR10 / VAR / SKIP s2 all end at 0). §129: a fresh head escapes in 2/4 and learns
  in 0/4. SKIP did rescue one basin seed (s4).
* **Early markers fail** (§130: the code-variance collapse at about 650 is common and transient). The 4k flag catches escaped
  failures 4/4 so far (§131, interim).
* **A weak attractor suffices** (§101: β_min 0.01 passes with a given codebook).
* **Reading a given state is the strongest entry aid we have:**
  - §111 READ (a per-step reader, with labels): 5/5 reliable hand-off.
  - §124 COPY (label-free: decode the given start state back): acquisition at L5 in **4/4** (EXPOSE L5–L8; BASE 2/4); serial in 2/4,
    against 0/4.
* **In-context rules** (§132 pilot): near-exact (0.9865 on held-out tables), serial, with an absolute-state code. The k6 prototype holds
  the op history, not the state, at 2k.
* §133 (running) tests SOFT / NOISE / PIN / FSQ-W error correction on a wide carrier, plus an S5×4 capacity probe.

## What the helper's mining shows (n in brackets; hard-rounded S5 unless stated)
* **Nothing at or before 2k separates learners from escape-without-learning:**
  - the 2k ratio is 0.78 (exact, n 12) against 0.79 (escape-fail, n 18);
  - the 2k probe matches: the state is read (R²_state 0.25–0.47) and the current op is gone, in 5/5 learners and in 6/11 escape-fails;
  - the carried norm matches.
  - Only the **plateau** group shows at 2k: an unread code (abl about 0, R²_state ≤ 0.2) in 10/11, and no or tiny decoder pulls in the
    replays (no pull or dmax < 0.5 → flagged 26/26 on the 4060 and 3070).
* **The earliest separator is the 2.6k → 4k window: loss(4k) / loss(2.6k)** at 0.71 (exact) against 0.997 (escape-fail).
  - At 0.9: 0/12 learners against 14/18 escape-fails flagged (§131's FLAG4).
  - By 4.5k, exact hard-rounded runs reach 0.37–2.75 and failures 3.42 and up (Audit's recomputation; the helper's first figure said
    ≤ 2.49 / ≥ 3.45).
  - **That window is the soft → hard rounding ramp (p_hard 0 → 1 over 2k–4k) together with L_max 4,** two changes at once.
* **The 2.6k → 4k window separates outcomes only on S5.** FF / CYC / MOD / RDFA have already solved L ≤ 2 by 2k, before p_hard
  rises (loss 0.000 at 1k–2k). Their 2.1k and 5k L-steps give transient bumps that recover fully, so their ratio is 0.00–0.21. S5
  learners are still at 2.1–3.9, and failures at 3.8–4.4, when rounding begins to harden. **The ramp quantises an unseparated code
  only on S5,** which is candidate 3's mechanism.
* **ρ₁ is necessary, not sufficient:** the exact band is 0.12–0.22 on S5, but 5 non-exact runs share it. ρ₁ = 0 marks the
  collapsed-code null (9 runs).
* **COPY (§124) gives the earliest, seed-uniform, sharpest acquisition event** (4/4 at the same round, jumps 0.99, no precursor), against
  BASE (2/4) and EXPOSE (precursors, later).
* **Partial outcomes:** a state-bearing code at 2k is not enough (3070 s17: R²_next 0.86 → the 3.2 basin).
* **Exact runs self-correct after late collapses; failures do not** (helper; Theory's recount on the 12 exact runs plus s12_a4000, n 13).
  - After the L_max 8 step (5k), 11/13 rise by at least 1.0 above their 4.9k–5k loss and revisit the null level (≥ 3.5). Some
    excursions last up to about 8k steps. All 13 recover to ≤ 0.01 by 24k.
  - **The depth before the step separates:** loss at 5k is 0.23–2.55 in the exact runs, against 3.34 (s17_a4000) and 4.26
    (s7_4060ti) in the failures, which re-enter and never recover. This is the same cut as 134-2(b)'s 4.5k statistic, one step
    later.
  - So the trained carrier already corrects large late errors once its basin is deep. The open problem is reaching that depth
    before the L_max 8 step.
* **End-of-training probes** (§128 S5, kept models):
  - learners hold the state at every length (R² 0.98–1.00);
  - the escape-fail s3 **reverts**: state R² 0.44 at 2k, then about 0 at L 2 / 4 / 8 by 24k, with the current op back at L 8 (0.83).
  - So F2 looks like an unanchored state code decaying (n = 1 probed), not a short-chain op-history shortcut. That fits candidates 1
    and 3.

## Failure taxonomy (what a method must fix)
* **F1, entry basin:** the first decoder pull fails and the code's variance collapses (127-7). Card-dependent, and often transient
  (§130).
* **F2, escape without learning (dominant; it shows only in the 2.6k → 4k window, during the rounding ramp at L 4):** the run passes the 2k flag but never composes. The loss rises at the L_max 2 → 4 → 8
  switches. The code is read by the decoder, but its content is not the state (VAR: R²_state −0.05; LR10 / SKIP s2: 0.25–0.28).
* **F2 has at least two sub-types at the end of training** (Audit; on-host and post-hoc end probes, n small):
  - **F2a, reversion** (s128_s5_s3): the code goes back to the op (R²_state about 0, last op 0.83 at L 8), and the decoder barely uses
    it (abl 0.89). COPY (candidate 1) targets this.
  - **F2b, state present but wrong** (s130chk_s5_s17_a4000: R²_state 0.80, abl 8.04, long_acc 0, loss 4.11; §129 s301: R²_state
    0.96, abl 14.7). The code carries the state linearly and the decoder depends on it, yet the chain fails.
    - Candidate causes: approximate encoding compounding with L (contraction, ρ₁); a wrong transition; or the decoder reading another
      mapping than the transition writes (a within-model dialect: §134's dialect probe; 2b ties the writers).
    - Discriminating measures on F2b models: accuracy at prefixes 1 / 2 / 4 / 8, R²_state by L (opprobe), and the fp32 ρ₁.
  - **End-probe profiles of the six cloud reruns** (opprobe R²_state at L 2 / 4 / 8; abl_dloss; Theory-verified from the probe files):
    - s12 exact: 0.997 / 0.997 / 0.997, abl 25.6;
    - **F2b flat** (s6): 0.82 / 0.81 / 0.81, abl 6.0, long_acc 0. Not compounding, so a wrong transition or a dialect mismatch;
    - **F2b compounding** (s17): 0.72 / 0.33 / −0.02, abl 8.0. The state decays with L.
      - It is a **non-contracting code**: acc at prefix 1 / 2 / 4 / 8 = 1.000 / 0.939 / 0.344 / 0.025, so the first transition is
        right.
      - Per-position survival is about 0.6 per step after that.
      - fp32 ρ₁ (ε 0.1) is 0.80, against 0.13–0.22 in the exact runs. 127-12's clause gives 0.42. The file's top-level "rho" is the
        vacuous hard-rounded summary.
    - **F2a** (s7, s5): about 0 at every L, abl 1.8–1.9.
      - s5 is wrong **from the first hand-off** (acc at prefix 1 / 2 / 4 / 8 = 0.097 / 0.0035 / 0 / 0; chance 0.008) while
        **strongly contractive** (fp32 ρ₁ 0.072): a stable map to the wrong attractor, not drift;
    - **F1** (s30): an unread 4-code buffer of the last op (last1 R² 0.69 / 0.99 / 0.99, abl about 0).
  - **Reading so far: two mechanisms, two fixes.**
    - F2a (s5; s128 s3 on a CPU preview) is **wrong from the first transition** with a contractive code (ρ₁ 0.07). That is a
      transition fix: COPY / SELF-ANCHOR / a reader.
    - F2b-compounding (s17) is **right at the first transition but not contractive** (ρ₁ 0.80). That is a contraction fix: §133's
      error correction / candidate 2.
    - s6 (F2b flat) is pending, with prefix accuracy and ρ₁ on the 3070. It decides whether F2b also has a wrong-transition
      sub-type.
    - **Positions 1 and 2 together separate the classes** (helper's and Experiments' 14-model prefix batch, Theory-verified;
      acc_by_prefix 1 / 2; chance 0.008).
      - Position 1 is segment(START, op₁) → code → decode: a write and a read of one state. Position 2 is the first transition on a
        self-written code.
      - **Exact** (s12, skip_s1, s102, s104, final_s1): 1.0 / 1.0 at every prefix, ρ₁ 0.13–0.22.
      - **F2b-compounding** (s17, s129 s202): 1.0, then 0.94 / 0.60, then decay; ρ₁ 0.80 / 0.48.
      - **F2b-transition** (lr10 s2): 0.957 / 0.037. The write and read work, but the transition on its own code fails. ρ₁ 0.
      - **F2a** (s5, s7, skip_s2_rerun, ctrl_s4_full, s128 s3): 0.04–0.10 / under 0.05. The first write or read already fails, even
        with ρ₁ in band (skip_s2_rerun 0.175; ctrl_s4_full 0.14).
      - **F2b-dialect** (s6, provisional): 0.048 / 0.021, then flat at about 3× chance, with no decay. Yet the end probe reads the
        state at every L (R² 0.81) and the decoder depends on the code (abl 6.0); ρ₁ 0.12, in band.
        - The code carries the state, but the decoder maps it wrong: a writer / reader dialect. That is what §134's dialect probe
          measures.
        - s6 was a 2k-flagged plateau that reorganised late (131-8).
      - **F1** (s30): chance at both positions.
    - **The discriminator is a triple:** positions 1 and 2 plus the end probe (R²_state with abl_dloss). F2b-dialect and F2a share
      the low prefix profile, and only the end probe separates them (state 0.81 / abl 6.0, against ≤ 0.31 / ≤ 1.9).
      - Counts: 5 exact, 2 compounding, 1 transition, 1 dialect, 5 F2a, 1 F1.
      - **ρ₁ is a measure for one class, not a classifier** (fp32, ε 0.1):
        - it is 0.07–0.29 within F2a (s5 0.072, s7 0.294), 0.12 in the dialect case, and 0 in F1;
        - only compounding stands apart (0.48, 0.80);
        - every failing model has ρ_k = 0 by the second or third hand-off. Perturbed and clean chains re-merge, in F2 on a **wrong**
          code: a contractive map to a wrong attractor, not §128-style drift.
    - **Ranking that follows** (Audit): a wrong transition, write/read or dialect dominates (7 of 10 failures; contraction 2). So the
      transition-side fixes come first for S5 entry: COPY (§134) for F2a's write/read, and SELF-ANCHOR (2b) for F2b-transition's
      reading of its own code. §133 / 2 stay for compounding and capacity. ρ₁ in band is not sufficient (skip_s2_rerun).
* **F3, drift:** a continuous or wide carrier has no attractor (RAW-H, CONT).
* **F4, capacity:** 8⁸ codes (24 bits) cannot hold large states (S5×4, multiplication, graph sets).
  - On the §128 families the cap is far from binding (helper's growth reading, fp32, 500 rollouts to 2,048, acc 1.0):
    - MOD's code_growth distinct is still rising at 2,048 (214–456), at codes_per_state_2048 43–91;
    - but by count it stays at or below the BFS-reachable set: fp32 for s1 (383 against 386), bf16 only for s2–s4 (391 / 214 / 456
      against 407 / 226 / 511). fp32 certificates for s2–s4 are queued with 128-4's extension.
    - So the 8-d code spreads one state over many codes and stays exact. F4 bites only where the state space itself exceeds 24 bits
      (§133 stage 2).
  - The S5 learners (s1 / s102 / s104; fp32, 500 × 2,048) are the same in kind, with an open code set:
    - distinct codes 2,946 / 2,683 / 2,344 at 2,048, still up about 6 % over the last doubling;
    - 20–31 codes per state, accuracy 1.0 / 0.996 / 1.0;
    - no certified bound, since the S5 CERT is inconclusive by time.
    - The code wanders within each state's basin without error at this length. The stability measure for S5 is per-step survival
      (s12: 1.0 at every position), not the code count.

## Candidate 1 (rank 1): COPY-ANCHOR, a label-free state-reading auxiliary from step 0 (targets F1 + F2)
* **Mechanism.** A fraction of each batch (copy rows, e.g. 10 %) is **autoencoding through the carrier:** [EOT, state text] →
  segment → code → answer decoder → the same state text (CE).
  - The text is either the problem's given start state (S5), or **the final answer text of another problem**. That is a
    final-answer-only target, so it is task-agnostic and C4-compatible for every family, including FF / CYC / MOD / RDFA, whose start is
    a code.
  - It gives the decoder a direct gradient to read the state from the code from step 0, with no failed first pull to wait for (F1).
  - It anchors the code's content to **state identity**, not op identity or op history (F2: the code that escaped-but-failed runs
    carry is not the state).
* **Evidence it can work:** §124 COPY (uniform L5 acquisition, 4/4, in self-training); §111 READ (reliable hand-off); 127-7 (failure
  starts with a failed first pull).
* **Falsifiable predictions** (S5, §128 recipe, RTX 3090):
  - (a) The known plain failures become learners: seed 2 (plateau), seed 3 (escaped then failed), seed 4 (plateau). Plain: 0/3.
  - (b) R²_state at 2k is above R²_item in every seed (plain basin seeds: below).
  - (c) No escaped-and-failed run.
  - (d) FAMILY-PASS on all five families is preserved: copy rows do not hurt CYC / MOD / RDFA / FF.
* **Cheapest decisive test** (about $5): a COPY arm on S5 seeds 1–4 (the s2 / s3 / s4 failures plus the s1 learner), with no restarts
  allowed.
  - **PASS = 4/4 exact at 2,048 and SERIAL, with no restart.** The plain recipe is 1/4 without restarts.
  - Then FF / CYC / MOD / RDFA, 1 seed each, as the no-harm guard.
* **Composes with §133:** COPY is carrier-agnostic, so it adds onto whichever carrier §133 shows is stable.

## Candidate 2 (rank 2): ROUND-TRIP ATTRACTOR, a label-free auto-snap toward the model's own decoded state (targets F3 + F4)
* **Mechanism.** At every hand-off, s = (1 − β)·z + β·sg(enc(dec(z))):
  - dec is the answer decoder's greedy text for the current code;
  - enc is Candidate 1's text → code path;
  - the attractor's fixed points are exactly **the codes of decodable states**: the "text manifold", not a grid.
* **What it buys:**
  - It works on a **wide continuous** carrier (F3, an error-correcting pull).
  - Its capacity is the space of decodable texts (F4): multi-token states such as S5×4's four registers, partial products or node sets
    are not bounded by 8⁸.
  - It is the user's "auto snap without hard-coded design": no schedule, no template, only the model's own decoder.
* **Risks:**
  - It locks onto a wrong or constant decode at init, so β ramps from 0 after entry, or is gated by decoder confidence.
  - Cost: one extra decode and encode per hand-off, about 2–3× compute.
* **Falsifiable predictions:**
  - (a) CONT-64 + RT reaches ρ ≤ 0.28 and is exact at 2,048 where CONT-64 drifts (§133's control).
  - (b) On S5×4 it CAP-PASSES where FSQ-8 fails by capacity (§133's reading).
* **Cheapest decisive test:** as a §133-style arm (CONT-64 + RT, β 0.1 from 2k) on the S5 proof seed, then S5×4. Its comparator is §133's
  CONT-64 / SOFT-64.
* **Variant 2b, SELF-ANCHOR (training only; added after §118 LONG / ANCHOR).** §118 ANCHOR made a wide 768-d latent exact with no snap
  to 2,048 in 5/5. It did that by pulling the free-running latent a = [EOT, latent_{j−1}, op_j] toward the text path
  b = [EOT, **gold** state text, op_j] (sg on b; 1 − cos + log-norm gap) in training only.
  - 2b swaps the gold text for **the model's own greedy decode** of latent_{j−1} ("the answer if queried now"). That makes it
    label-free, so C4-compatible.
  - **Zero test-time cost:** the attractor is learned, not applied at inference, unlike 2's per-hand-off mixing.
  - Guards as for 2: weight 0 until decodes are valid (copy accuracy or decoder confidence), then constant. Report the share of anchors
    built on wrong decodes.
  - **Comparators:** §118 ANCHOR (gold text: the labelled upper bound), §133 CONT-64 and §126 RAW-H (unanchored).
  - Prediction: CONT-64 + 2b reaches ρ ≤ 0.28 and is exact at 2,048 where CONT-64 drifts.
  - **Literature 5a416fff** (`docs/LIT_SELF_ANCHOR_2026-10-06.md`) finds it new as scoped, and adds three design guards:
    - (a) train the text path b too, label-free: roll out from the self-decoded text to the final answer (§116's mechanism), so b
      cannot go dead (the §121 / §122 lesson);
    - (b) gate on decode diversity / active units as well as validity, against constant-decode collapse;
    - (c) use a modest weight, optionally anchoring only final-answer-correct chains (§110's filter).
* **Composes with Candidate 1** (enc is Candidate 1's copy path). Candidate 1 alone fixes entry; Candidates 1 and 2 together are the
  proposed full method: a **COPY-anchored, round-trip-stabilised wide carrier.**

## Candidate 3 (new, from the helper's data): ROUNDING-AWARE ENTRY (targets F2)
* **Hypothesis.** Escape without learning happens when FSQ rounding hardens (2k–4k) before the code has separated the states. The soft code
  that carried state at 2k (R²_state 0.3–0.4) loses it when it is quantised.
  - Rival: the L_max 2 → 4 step (composition demand) is the cause. The two are confounded in §120's schedule.
* **$0 disentangling prototype first** (LOCAL_BACKLOG): the same seeds, plain vs **delayed rounding ramp** (p_hard 0 → 1 over 6k–8k, L_max
  schedule unchanged), short runs to 6k, reading loss(4k) / loss(2.6k) and loss(7k) / loss(5.6k).
  - If the failure window moves with the ramp, rounding is the cause.
* **If so, the fix candidates:**
  - a soft-snap carrier (§133 SOFT, a constant weak pull instead of a hard ramp);
  - or a label-free gated ramp, which hardens only when the code is separable: grid margin, active units (1509.00519), or the decoder's
    agreement between soft and rounded codes.
  - Registered only after the prototype.

## Candidate 4 (the user's idea, via Audit): TEACHER-STATE DISTILLATION (TSD), from a CoT model's hidden states into the carrier
* **The user's point:** natural-language CoT is a poor training artifact for a latent reasoner. Use the CoT **model's internal
  state**, not its text.
* **Mechanism.**
  - Teacher: a text-CoT GPT-2 on the same task, frozen. Text CoT learns S5 and multiplication (§98 TC 1.0).
  - Student: §128's carrier.
  - At every hand-off j, an auxiliary loss pulls the student's carried embedding (W_up(code_j), 768-d) toward the teacher's hidden
    state at the matching step boundary: the last token of CoT step j, at one layer chosen by a held-out state probe on the
    teacher and fixed before any student run.
  - It passes through a learned linear map, with §118 ANCHOR's loss (1 − cos + log-norm gap) and sg on the teacher.
* **Changes from Audit's draft (Theory):**
  - **Constant weight, no anneal.** Inference never touches the teacher, so the model is pure latent at test whatever the weight.
    Annealing to 0 buys only a FINAL-only last phase, and fading an anchor after entry is the §126 ANNEAL / α → 1 collapse pattern.
    The anneal is a later ablation.
  - **S5 is a weak test of TSD's specific claim.** Per-step state supervision already solves S5 (§111 READ, §118 ANCHOR 5/5), so
    the S5 stage checks the mechanism and the entry fix. The question "teacher hidden state beats step text" is decided on
    **multiplication**, where the latent methods hit the k = 4 wall (§98 GEN-M 0.163 / 0.005) and text CoT is 1.0.
  - **A text-target control** with the same per-step information isolates the user's claim: the student's own text-path latent of
    the teacher's step text (§118 ANCHOR in this harness).
* **Why it may succeed where §119 CODI did not** (0/3; Audit):
  - alignment at every hand-off, not one token;
  - it acts through the carrier;
  - it supervises the **transition** directly, which is the dominant S5 failure (F2a + F2b-transition + dialect, 7 of 10 above).
* **Scope (honest):** not C4. The teacher's CoT carries step states, as §118's reader does. It is a C1 / C6 / C2 and multiplication
  tool, not a label-free result. Shortcut check: SERIAL plus the prefix profile (positions 1 and 2 = 1.0).
* **Cheapest decisive staging:**
  - **(A)** S5 on §128's carrier, seeds 1–4, no restarts: TSD vs the text-target control. PASS = 4/4 exact at 2,048 + SERIAL,
    against plain 1/4. Teacher models: reuse a kept S5 TC model if one exists.
  - **(B)** only after A: 4×4 multiplication, against §98's latent arms and ICoT-SI 0.998.
  - Literature's check comes first (it is new as scoped only if the per-hand-off hidden-state target through a discrete carrier
    is). Registration after that, with Audit's budget.

## Not proposed
* More entry levers of §127's kind (escape without learning).
* A global contraction penalty (it fights S5's isometric carry; Literature 7dbdb426).
* Learned codebooks (§101's SELF-SNAP premise failed).

## Status
* **Candidate 1 is registered as §134** (26bc216e; scorer `theory/score_s134.py`), with Literature's design checks:
  - constant f;
  - copy decoded from the code alone;
  - a "two dialects" identity probe and a copy-code donor injection (134-1);
  - no copy → one-op rows (they would label transitions);
  - the FF caveat (copy anchors only 2 codes there).
* **Candidate 2** waits for §133's stage 1. Literature's guards apply:
  - dec reads the code only;
  - β ≈ 0.01, with β = 0 until decodes are valid;
  - decode "the answer if queried now" (= the state for S5 / CYC / MOD / RDFA, not FF);
  - arm order COPY → COPY + RT → COPY + RT + SKIP.
* Literature's $0 test (does escape without learning carry the op history?) **is done** (3070, eval_probe_posthoc.json; n = 1 failure):
  - learners s1 / s102 / s104 hold the state at L 2 / 4 / 8 (R² 0.995–0.997), with no op history (last-op R² < 0);
  - the escape-fail s3 holds neither the state (R² ≤ 0.04) nor the last op at L 2 / 4 (0.04), but does hold the last op at L 8 (0.83).
  - Last-2 / last-3 R² cannot be read: probe coverage is 0.14 / 0.002 of those classes.
  - So the failure is a decayed state code, not a stored op history, as in the "What the helper's mining shows" section above.

## Order
* Register Candidate 1 first (cheap, decisive, targets the dominant failure).
* Register Candidate 2 after §133's stage 1 reads, so its comparators exist.
* Literature's check comes before each registration. Spending is staged with Audit's approval, cloud only.
