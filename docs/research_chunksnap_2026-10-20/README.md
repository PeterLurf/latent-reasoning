# CHUNK-SNAP (research step 2026-10-20): WITHDRAWN before registration

- CS_draft.md (8cd29919) is not registered and will not be.
- The red team (REDTEAM_CS.md, BLOCKER F1) found that the draft's premise was wrong. It said "only k = 1 and k = infinity are measured", but the committed eval_rs files (part_rs decode cells n,r with t = n - r) already hold the first-chunk surface e(m, k) at m <= 16 for D3-T s1, FT-SELF, R2-s1 and R1-s2.
- That data already rules out both pure hypotheses. Theory re-read s213_d3t_s1/eval_rs.json:
  - e(14, 6) = 0.536 against e(14, 8) = 0.014, a steep fall in k at a fixed start size;
  - e at k 4 falls 1.000 / 0.962 / 0.927 from m 8 to 12.
- The draft's primary, CHUNK-C2, nearly passes at k = 1 already, so it is not a C2 test (F2).
- Theory's error was claiming "unmeasured" without checking the existing eval parts.
- What remains genuinely open is the C5 trigger headroom: ORACLE-k / confidence trigger at m 17-40, with decode-every-pass and length stratification. It may come back later as a slim $0 descriptive row, behind SCALE-PROBE.
