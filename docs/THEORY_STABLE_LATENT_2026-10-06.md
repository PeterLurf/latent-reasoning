# A stable latent method: design synthesis (2026-10-06) — DRAFT, pending Literature and helper inputs

The user's directive (via Audit): "ask literature and helper to work together with theory to invent and synthesize a stable latent
method". Theory leads. Literature maps the prior art (@@LIT@@); the Audit helper mines our raw runs (@@HELPER@@).
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

## Failure taxonomy (what a method must fix)
* **F1, entry basin:** the first decoder pull fails and the code's variance collapses (127-7). Card-dependent, and often transient
  (§130).
* **F2, escape without learning (dominant):** the run passes the 2k flag but never composes. The loss rises at the L_max 2 → 4 → 8
  switches. The code is read by the decoder, but its content is not the state (VAR: R²_state −0.05; LR10 / SKIP s2: 0.25–0.28).
* **F3, drift:** a continuous or wide carrier has no attractor (RAW-H, CONT).
* **F4, capacity:** 8⁸ codes (24 bits) cannot hold large states (S5×4, multiplication, graph sets).

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
* **Composes with Candidate 1** (enc is Candidate 1's copy path). Candidate 1 alone fixes entry; Candidates 1 and 2 together are the
  proposed full method: a **COPY-anchored, round-trip-stabilised wide carrier.**

## Not proposed
* More entry levers of §127's kind (escape without learning).
* A global contraction penalty (it fights S5's isometric carry; Literature 7dbdb426).
* Learned codebooks (§101's SELF-SNAP premise failed).

## Order
* Register Candidate 1 first (cheap, decisive, targets the dominant failure).
* Register Candidate 2 after §133's stage 1 reads, so its comparators exist.
* Literature's check comes before each registration. Spending is staged with Audit's approval, cloud only.
