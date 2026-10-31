# REVIEW_PG: red team of PG_draft.md (Opus subagent, 2026-10-30). Verdict: NO-GO as drafted. Theory: WITHDRAWN (Note 251-14)

Scripts: theory/read_s251_pg_pw.py (word → op probabilities; re-run by Theory, reproduces), read_s251_pg_perop.py, read_s251_pg_perseed.py (per-frame per-op computed op, §250 sets regenerated, sigs equal to eval_frames.json).

1. **BLOCKER (premise):** content words are not the shortcut.
   - In frames 0-5, "more" / "fewer" are 0.50 add / 0.50 sub (frame 4 inverts roles), "as many" is 0.40 / 0.20 / 0.40 and "than" 0.45 / 0.55.
   - Only operator words are pure, and the 251-6 failures ignore them: f6 sub_v "less those of C" contains "less" (P(sub) 1.0) and still computes add.
   - So the shortcut is a **construction default** (skeleton → op, operator word ignored). The draft's word bound is ill-defined.
2. **BLOCKER:** fd_div (§250 DIV) already decorrelated "by" (P(mul|by) 1.0 → 0.8), with this result:
   - f7 "exceeds … by c" stays mul in s1 and flips to sub in s2. Argument order is still ignored.
   - "once" (only in sub_v frame 8) drives held-out mul_v to sub: f12 1.00, f14 0.59 / 0.70.
   - f6 is unchanged.
   - Extra frames re-route which cue is used; they do not induce composition.
3. **BLOCKER (procedure):** Note 251-5 pre-stated "DIFFUSE → phrasing parked", and 251-6 read DIFFUSE. A third phrasing iteration must disclose that it overrides this.
4. **BLOCKER (build):** frames are drawn per step regardless of op (ow_words.py:116), so per-op held-out cells need new sampling. Uniform sampling over ~200 cells drops frames 0-5 to ~3% of steps (NOLOSS risk).
5. **BLOCKER (comparators):** the design confounds template volume with decorrelation. It needs a GRAM-CORR arm (same templates, each skeleton tied to one op).
6. **BLOCKER:** Stage 0 is likely VACUOUS. Recombined trained words already pass at step 0, failures cluster on a sub_v prior and "by" → mul, and verbs carry no op information (GT-LEX is empty).
7. **Minor:** censoring / LP0 missing; eval size too heavy; P(COMPOSE) optimistic.

**Literature:** SCAN / COGS: lexical recombination is easier than structural / role generalisation, and diversity helps when it covers the held-out structure. Here the hard part is role order and new operator lexemes.

**If re-opened:** a skeleton × operator-lexeme × role grammar, with a $0 pre-read first: does a masked-skeleton classifier trained on the training templates predict the op the model *computes* on held-out frames? If yes, decorrelation only re-routes cues.
