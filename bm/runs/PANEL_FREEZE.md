# §60 Phase A — settings frozen for the seed panel (2026-10-01 02:26 EDT, before any panel run)

* Code: `bm/` at the commit of this file (data/model/run as of 1ffaeff + dev-only eval option + --kmax diagnostic flag, unused by the panel).
* Data pool: `bm/data_pool` (seed 0, train 50,000; evaluation sets as generated).
* **λ_order = 1** for BM-oc. The development pilot gave identical dev metrics for λ = 1 and λ = 0.1 (both collapse to the
  identity map, dev probe 0.874); tie broken towards the registered primary concept (stronger order consistency). Disclosed.
* **Steps 10,000, batch 128, lr 3e-4 (cosine, 200 warm-up), AdamW wd 0.01**, identical for both arms, from the pilot
  (0.47 s/step with two schedules). The pilot's train accuracy was flat from step 2,000, so more steps were not chosen.
* Seeds 1–4 in both arms. Failed or unfinished seeds are kept, with no extra training.
* **Reference line (Audit):** the identity map f(a) = a scores probe 0.874 on dev and satisfies all laws; panel probe
  accuracies and law validity are reported against it.
