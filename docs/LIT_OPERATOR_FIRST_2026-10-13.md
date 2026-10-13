# Check of Zhang et al. 2607.20594's operator-first curriculum, for Audit's easy-first arm (2026-10-13)

Literature session, for Audit (the request) and Theory. ID verified via the arXiv abstract page; details from its HTML (Sec. 6–7, Tables 4 and 7, App.)
(2026-10-13).

**The paper:** Tong Zhang, Junhao Hu, et al., "When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers"
(2026-07-22). Weight-tied looped transformers on **group word problems** (S4, Z60, A5, S5).

## CORRECTION to the cited claim
- **"A5 seed lottery 8/22 → 100% with operator-first" mixes two different results:**
  - **Operator-first fixes S5, not A5.** S5 is *never* learned by free training under the standard length curriculum. Operator-first cracks it in
    **10/10 seeds** (two machines; exact match ≥ 0.92 at 2× the training length).
  - **The A5 seed lottery is 8/22 seeds** cracked from scratch at T = n (Table 7). It's bypassed by **budget annealing** (warm-starting a solved model across
    loop contracts: **4/4**), **not** by operator-first.
  - A5 itself "trains like the solvable groups". The wall the paper names is S5's 120×120 operator.

## The four questions
1. **What "operator-first" trains first:** "an **extended n = 2 stage** that **masters the composition table before any chaining**". That is, length-2
   sequences (single pairwise compositions) until the operator is learned, then the usual length curriculum. Single operator applications, i.e. length-2
   items.
2. **The schedule** (operator-first S5, at model scale "m"): **stage lengths 2 : 20k steps, 4 : 10k, 8 : 10k, 16 : 10k, 32 : 12k** (62k total). Free training
   used T = ⌈log₂ n⌉ + 3 (jittered) or fixed T, with a length curriculum.
3. **Optimiser changes: none.** Same AdamW (lr 3 × 10⁻⁴; 2 × 10⁻⁴ at scale "l"; with weight decay). The fix is the curriculum alone.
4. **Timing:**
   - **Table 4** reports the **optimisation cost of the operator as training steps to per-token accuracy ≥ 0.98 on pair composition (n = 2):** at scale s,
     S4 ≈ 0.5k, Z60 2.5k, A5 ≈ 4k, **S5 > 40k (never)**; at scale m, Z60 2.0k and S5 6.5k.
   - **There's no before / after convergence-time comparison for the full chained task.** Without operator-first, S5 never converges, so success rate is the
     reported effect.
   - Their τ(n, i) is convergence over *loops* within a forward pass, not training time.

## Relevance to us
- **The closest external analogue of our S5 basin lottery.** Their diagnosis is that **joint learning of operator and chain deadlocks**, while sequenced learning
  doesn't.
- Our fsq120 length curriculum starts at L = 2 but moves on at a threshold. Their fix is an **extended** n = 2 stage (20k steps, about a third of training) to
  master the full 120 × 120 table first.
- **A faithful easy-first arm:** hold L = 2 until pair-composition accuracy ≥ 0.98 (their criterion), *then* the normal length curriculum.
- **Two more findings bear on our designs:**
  - **"Imposed seriality fails":** streaming input release (token i at loop i) fails on every group task (including A5) that free scheduling solves. Adding
    per-loop supervision on the revealed prefix didn't rescue it (Table 5). Relevant to our segment-per-item designs.
  - **Budget annealing** transfers a found algorithm across loop contracts (4/4) and bypasses the from-scratch lottery. That's a candidate fix for
    non-entering seeds: warm-start from an entered seed's model.
