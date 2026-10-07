# ProsQA-NA on the task-agnostic carrier: design draft (2026-10-07; Audit's steward idea; design only, $0, nothing registered)

**Target.** ProsQA-NA (§59: the adjacency / proximity cues neutralised; `theory/prosqa_na.py`, data regenerated into `prosqa/p8data/`).
* p8_na is clean on every shallow-cue check (0.50).
* **Earlier bar:** every GPT-2 latent arm was at chance (0.486–0.508), and text CoT reached 0.767 (PROTOCOL8 amendment 3).
* Coconut's published 97.0 on the original ProsQA rides the proximity cue (0.91, §57).

**Mapping onto fsq120 (one family adapter, the same harness as §128 / §132 / §137).**
* **Segment t** = [EOT, carried code, the problem's statements (the graph, in context every segment, as §132's tables), the two options, a hop token t].
* **The carried code holds the search frontier:** the set of concepts reached from the root person after t hops (BFS).
  - It is a set over at most about 25 concepts, so up to 2²⁵ subsets (25 bits). FSQ-8's 24 bits are borderline; SOFT-16 (§133's PASS, 48-bit grid) or
    a wider passing carrier fits.
* **FINAL only:** after T hops (T = the generator's depth bound) the answer decoder reads the code and emits the option that is reachable. There are
  no frontier labels.
* **Curriculum over T** (hop count), as the length curriculum. The answer for a shorter T is "reachable within T hops", computable from the graph.
  Shorter-hop problems are §110b-1-qualified prefix labels, as in 139-3. A strict arm with T fixed at the maximum is the C4-stretch comparator.
* **SERIAL:** inject a donor problem's frontier code into a recipient at hop t. The answer must follow the donor's frontier expanded over the
  recipient's graph (exact).
  - **C1 is unforced** here, as in §137: the whole graph is in context every segment, so an in-pass search is possible. SERIAL separates the latent
    route from in-pass search (137-1 (2)'s IN-PASS reading).

**Gates before registering.**
* §132 rung 2 (k 6, m 3 in-context tables) reads PASS: the latent carrier tracks state from larger in-context rules. Rung 2 is staged after §136.
* §137 or §140 reads: whether a segment can consume in-context operands.
* A wide-carrier PASS (SOFT-16 already) for the frontier's size.
* Literature's check on graph-search latent reasoning (Coconut's BFS reading 2412.06769; the ProsQA shortcut note `docs/LIT_PROSQA_SHORTCUT_2026-09-30.md`).

**Rules sketch.** PASS = ≥ 3/4 seeds ≥ 0.95 on p8_na_test (balanced, chance 0.5), with SERIAL and ρ in band. Bars: text CoT 0.767 and the latent arms at about
0.5 on the same data.

**Cost estimate.** Segments are long (the graph is about 150–250 tokens) and T ≤ 6, so about 1.5–2 h per 24k run on a 3090, about $0.5 each. About $4–6 for 2
arms × 4 seeds.
