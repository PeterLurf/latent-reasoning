"""§264 / §265: boxes/insert_render.py distract / distract_mix gain nmax (default 2) and n (None = drawn): the draw n ~ U{0..nmax} is ALWAYS consumed from the item's rng
(nmax 2: rng.integers(0, 3), bitwise the pre-edit draw), a forced n (§264 op_count) replaces the drawn value; §265 name budget: n is capped at floor(free names / 2)
(both distract_mix and draw_pairs use 2 new names per insert) and the item records distract_n_drawn / distract_capped. Each edit asserted once.
    python -B scratch/s271_build/edits_ir.py"""
import sys

sys.path.insert(0, "scratch/s268_build")
from edits_s268_fn import edit  # noqa: E402

p = "boxes/insert_render.py"; assert "nmax" not in open(p, encoding="utf-8").read(), "already applied"
edit(p, [
    ('def distract(it, seed, step, k):\n',
     'def _budget(it, n):\n'
     '    """§265: the inserts that fit the 40-name budget (2 new names each) -> (n capped, capped?)."""\n'
     '    used = set(it["names"]["spine"]) | set(it["names"]["givens"]) | set(it["names"]["distractors"]); free = sum(1 for x in OW.NAMES if x not in used)\n'
     '    return min(n, free // 2), n > free // 2\n\n\n'
     'def distract(it, seed, step, k, nmax=2, n=None):\n'),
    ('    rng = np.random.default_rng((seed, SEED_DISTRACT, step, k)); n = int(rng.integers(0, 3))\n    if n == 0:\n        return it, 0\n    giv, ops, meta = draw_pairs(it, n, rng)',
     '    rng = np.random.default_rng((seed, SEED_DISTRACT, step, k)); n_draw = int(rng.integers(0, nmax + 1)); n = n_draw if n is None else n   # §265 nmax / §264 forced n\n'
     '    n, capped = _budget(it, n)\n    if n == 0:\n        return it, 0\n    giv, ops, meta = draw_pairs(it, n, rng)'),
    ('o = assemble(it, giv, [(gaps[int(rng.integers(0, len(gaps)))], op) for op in ops], meta); o["distract_n"] = n; return o, n',
     'o = assemble(it, giv, [(gaps[int(rng.integers(0, len(gaps)))], op) for op in ops], meta); o["distract_n"] = n\n'
     '    if capped:\n        o["distract_capped"] = n_draw\n    return o, n'),
    ('def distract_mix(it, seed, step, k, types=MIX_TYPES):\n', 'def distract_mix(it, seed, step, k, types=MIX_TYPES, nmax=2, n=None):\n'),
    ('    rng = np.random.default_rng((seed, SEED_MIX, step, k)); n = int(rng.integers(0, 3))\n    if n == 0:\n        return it, 0\n    d = it["d"]; u = it["unit"]',
     '    rng = np.random.default_rng((seed, SEED_MIX, step, k)); n_draw = int(rng.integers(0, nmax + 1)); n = n_draw if n is None else n   # §265 nmax / §264 forced n\n'
     '    n, capped = _budget(it, n)\n    if n == 0:\n        return it, 0\n    d = it["d"]; u = it["unit"]'),
    ('o["distract_n"] = n; o["distract_types"] = ty; o["distract_meta"] = meta\n',
     'o["distract_n"] = n; o["distract_types"] = ty; o["distract_meta"] = meta\n    if capped:\n        o["distract_capped"] = n_draw\n')])
print("insert_render: nmax / n / name budget")
