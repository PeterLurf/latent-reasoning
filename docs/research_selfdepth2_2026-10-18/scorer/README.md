# score_s216 fixtures (G-S12)

fixtures_s216.py builds synthetic run trees under ./_fx/ next to itself, so run it from a scratch copy next to a copy of theory/score_s216.py, never in the repo:
copy both files to a scratch dir, then python -B fixtures_s216.py.

- 2026-10-18 02:3x: 39 / 39 PASS (pre-landing).
- 2026-10-18 after note 216-7 (Audit): 40 / 40 PASS. The two pre-landing real-tree snapshots became state-independent (synthetic panels with the D2 launch gating checked both ways; real-tree no-crash / no-write), and args_ref_content_rule_216-7 was added (ok-content, one-key-diff, missing file, R2 one-key-diff).
