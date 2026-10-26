# Research digest: what worked, what didn't (§0–§228)

Index for the project's research history, for future inspiration and to avoid repeating dead ends. Written 2026-10-23 by Audit. Per-section rows (question → verdict → key number → lesson):
- [DIGEST_s000-096.md](DIGEST_s000-096.md): toy Coconut reachability → GPT-2 state tracking (2026-09-29 to 10-03)
- [DIGEST_s097-170.md](DIGEST_s097-170.md): S5 / multiplication hand-off, task-agnostic recipe, additive memory (10-03 to 10-09)
- [DIGEST_s171-228.md](DIGEST_s171-228.md): INJECT think passes, STEP-AUX, RMS×WINDOW, GSM8K/CODI, VERSION-BANK/SLVA, self-depth, language (10-09 to 10-23)

Each digest was condensed from section headers, result blocks and Notes, not from every line. The authoritative text is docs/THEORY.md (and docs/archive/THEORY_s0-s96.md); check the source before citing a number.

## North-star status (C1 > C6 > C4 = C2 > C3 > C5 > C7)

| C | status | best evidence | what blocks it |
|---|---|---|---|
| C1 latents do the work | MET on synthetic; language MET scaffolded (plan given) at d ≤ 3, 1 seed (§234); chain breaks at d4 | §184 (nested arithmetic, 2/2), §211 VERSION-BANK, §217 causal use preserved; §227 sets a language band | GSM8K latents recompute from input (§207, §208); word tasks not one-pass learnable (§223, §226); §228/§229 failed at GPT-2 small (d2 at chance); larger base next |
| C6 stable | MET in scope; hold without depth cost (§230) | §193 RMS×WINDOW 2/2 to K'32; §224 trained far pass holds to K'128 | seed-3 failures (§192, §205); §224's from-scratch hold cost depth; §230 post-formation FT holds to K'128 at depth parity (1 confirmed + 1 motivating seed) |
| C4 scalable supervision on a pretrained LM | not met (strict) | self-labelled chains (§215–§219); short-label foothold (§96, §112) | strict C4 barrier from final answers alone (§103–§117); CoT warm start forgotten (§202) |
| C2 more thinking helps past training | not met | written-state / restart length generalisation in toys (§26, §71, §92); in-support self-depth (§215, §216) | boundary still binds (§216 LIT-C2-2 0.018); depth limited by pass count / basin entry |
| C3 competitive with text CoT | MET in scope (synthetic) | §225: within 0.02, ~20–50x fewer steps, ~0.65x FLOPs; §159 0.80x FLOPs; CODI 0.379 GSM8K at published budget (§210) | language untested; GSM8K latent still below text at our budget (§200) |
| C5 decides how long to think | partial | learned halt stops overthinking 7/7 in range (§201) | answer-settling halts early (§173); premature at the edge |
| C7 readable | weak | pointer/written-state readouts (§2, §45, §71); probes on VB | readings decode-dependent out of range (§213); §221 context-bound |

## What worked (cross-era)

- **Per-step supervision is the master key.** Loop-aligned per-iteration targets (§88, §91), per-pass aux from text CoT (§175), equal per-pass weight (§184), per-step relabelling (§191). With final-only loss, the passes go unused (§171, §176, §177).
- **Re-anchoring / restarts / written state** give length generalisation: §26 restarts, §71 at 16x, §92 to 2,048, §97/§118 S5 to 2,048.
- **Normalised reads and multiplicative/standardised cells**: §70 multiplicative 10/10 vs additive 0/10; §72b/§73 standardised readout; §185/§189 RMS read.
- **Sampled pass count (WINDOW)** for stability: §182, §193.
- **Small footholds**: 3–10% short labels or a COPY objective unlock acquisition (§85, §87, §112, §124); self-labels then reproduce the teacher (§96, §215).
- **Task-agnostic recipe and staged growth** across five families (§128–§143).
- **Additive / write-after-read memory** breaks the multiplication middle-column wall; latent beats text on length (§147–§164).
- **Budget before architecture**: CODI's GSM8K gap was budget (§209, §210); §86 → §92 the same.

## What didn't (cross-era dead ends)

- Identical-input think passes with final-only loss, including pass-index, fade, random-K and interleave variants (§171–§186).
- Mechanism-targeted seed rescues: Grokfast §196, CSR §199 (target gamed), SVR §203 (unused), IIT §205, fixed-point penalty §224.
- Strict C4 from final answers; RL for discrete latents (§69); learned halt from reward (§50, §51).
- Inference-time repair of public CODI/SIM-CoT (§13); teacher hidden-state distillation (§135); CoT warm start (§202); CODI L1 coupling (§206).
- Sets/beams in one episode (§44–§48); paraphrase transfer (§74, §79); label thinning (§93, §95).
- Language learnability without a pilot: §218 ProsQA-SYM at chance for both arms.

## Traps that recur (check before each registration)

- **Basin-entry timing and the seed lottery** drive most "recipe" differences; two-seed claims broke at seed 3 (§187, §192, §205). GPU class flips basins (§108, §125).
- **Ablation choice**: zero/mean ablation overstates C1 — use donor patching (§104). Decodes are unfaithful out of range (§213).
- **Delivery bugs voided results**: no_grad (§116), std not in force (§125), test exclusion removed a table (§178), the curriculum gate advanced on step caps (§214, §218).
- **Uninformative controls**: a control that never receives the intervention (§178, §212, §220).
- **One-sided dynamics targets get gamed** (§199).
- **Outcome-based triggers need absolute floors** (§162); post-hoc choices must be disclosed (§215, §224).

## Open threads with the most promise

- **C2 past the boundary**: self-depth lineage plus a hold that costs no depth (§216, §224 RD); F3 self-synthesis of deeper data and F5 verify-then-extend (§52), mostly untried.
- **Language C1**: §227's band on a ~0.5–1B base (§228/§229 failed at GPT-2 small); fresh procedurally generated data (§168 ProsQA-FRESH lead).
- **GSM8K depth**: CODI's m ≥ 3 weakness (0.265, §210); latents that carry state instead of recomputing it (§207).
- **C4**: short-label teacher (p ≤ 0.05) plus final-answer self-training (§96).
- **C7**: function-vector / mid-layer readouts (§79-6); readout of restart latents.
