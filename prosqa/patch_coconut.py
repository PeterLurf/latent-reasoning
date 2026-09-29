"""Speed patch for the official coconut.py (PROTOCOL6 addendum D amendment): replace the per-position Python rebuild of
inputs_embeds after each latent pass with one out-of-place index_put. Same values and gradients (test_patch.py checks
this against the unpatched code). Keeps the original as coconut_orig.py.   python patch_coconut.py /workspace/coconut
"""
import os
import shutil
import sys

d = sys.argv[1]
src = open(os.path.join(d, "coconut.py")).read()
if "vectorised feedback" in src:
    print("PATCH already applied"); sys.exit(0)
start = src.index("            # to avoid in-place operations")
end = src.index("        # final pass")
new = '''            # vectorised feedback (PROTOCOL6 speed patch; same values as the official per-position list rebuild)
            if filling_indices:
                bi = torch.tensor([b for b, _ in filling_indices], device=inputs_embeds.device)
                ti = torch.tensor([t for _, t in filling_indices], device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.index_put(
                    (bi, ti), hidden_states[bi, ti - 1 - hidden_states_offset, :])

'''
shutil.copy(os.path.join(d, "coconut.py"), os.path.join(d, "coconut_orig.py"))
open(os.path.join(d, "coconut.py"), "w").write(src[:start] + new + src[end:])
print("PATCH applied")
