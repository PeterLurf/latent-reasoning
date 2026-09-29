import pathlib

p = pathlib.Path(r"D:\Latent\report\render2.js")
r = p.read_text(encoding="utf-8")
R = []
# findings bullet 1
i0 = r.index("  F2.push(`<b>Only the recurrent models solve the deepest queries;")
i1 = r.index("  F2.push(`<b>Readable ≠ usable.</b>")
R_new = ("  F2.push(`<b>Recurrence buys reliability on the deepest queries, not a broad capability.</b> With depth-matched decoys (chance 50%), "
         "recurrent seeds score ${pct(mat(seeds, \"6\")[1])}–${pct(mat(seeds, \"6\")[2])} at six hops. All ${ctrlN.length} pause and no-latent control seeds score "
         "${pct(mat(ctrlN, \"6\")[1])}–${pct(mat(ctrlN, \"6\")[2])}. Below that, controls vary enormously between seeds. Some never pass two hops. One no-latent seed, with no latent steps at all, "
         "reaches ${pct(mat(ctrlN, \"4\")[2])} at four hops and ${pct(mat(ctrlN, \"5\")[2])} at five, matching the weakest recurrent seed. Both of Part I's single-seed control conclusions were seed luck.`);\n\n")
r = r[:i0] + R_new + r[i1:]

old_prose_start = r.index('    P("prose-competence").innerHTML = `<p>Every recurrent seed clears the 85% competence bar on held-out families (${pct(accMain[1])}–${pct(accMain[2])} overall balanced accuracy).')
old_prose_end = r.index("    <p>Truncating the latent budget at inference is not a clean ablation.", old_prose_start)
new_prose = ('    P("prose-competence").innerHTML = `<p>Every recurrent seed clears the 85% competence bar on held-out families (${pct(accMain[1])}–${pct(accMain[2])} overall balanced accuracy). '
             'With depth-matched decoys as negatives, they score ${pct(lo(R_, "5")[1])}–${pct(lo(R_, "5")[2])} at five hops and ${pct(lo(R_, "6")[1])}–${pct(lo(R_, "6")[2])} at six. '
             'At six hops every control seed is at or near chance (pause ${pct(lo(Pz, "6")[1])}–${pct(lo(Pz, "6")[2])}, no-latent ${pct(lo(Kz, "6")[1])}–${pct(lo(Kz, "6")[2])}).</p>\n'
             '    <p>Below six hops the controls differ enormously between seeds. Two of three pause seeds and one no-latent seed never pass two hops. '
             'Others solve 3–4 hops through computation in the prefix, and one no-latent seed reaches ${pct(lo(Kz, "5")[2])} at five hops, as good as the weakest recurrent seed, with no latent steps at all. '
             'Part I\'s single pause seed happened to be a strong one and its single no-latent seed a weak one, so both control conclusions there were seed luck. '
             'What replicates is narrower: recurrent training reached the deepest queries in every seed, and no control seed reached six hops.</p>\n')
r = r[:old_prose_start] + new_prose + r[old_prose_end:]

R.append(('      ["Recurrent latent computation is needed for the deepest (5–6 hop) queries", `All ${["pause","pause1","pause2","k0","k0_1","k0_2"].filter(n => (S.competence_matched||{})[n]).length} control seeds near chance at 5–6 hops (depth-matched)`, yes("supported"), "Synthetic task only; controls are 2-layer like the recurrent models"],',
          '      ["Recurrent latent computation is needed for the deepest queries", "Six hops: every recurrent seed above, every control seed at or near chance (depth-matched); five hops: one no-latent seed matches the weakest recurrent seed", midc("supported at six hops, partial at five"), "Three seeds per model type; controls are 2-layer like the recurrent models"],'))
R.append(('      ["Extra positions or feedback are needed for 3–4 hop queries", "Some pause and no-latent seeds solve 3–4 hops through prefix computation; others fail beyond two", no("not supported"), "Competence of controls is strongly seed-dependent"],',
          '      ["Extra positions or feedback are needed for 3–5 hop queries", "Some no-latent seeds solve 3–5 hops through prefix computation; other control seeds fail beyond two", no("not supported"), "Competence of controls is strongly seed-dependent"],'))
for old, new in R:
    assert r.count(old) == 1, old[:60]
    r = r.replace(old, new)
p.write_text(r, encoding="utf-8")
print("ok")
