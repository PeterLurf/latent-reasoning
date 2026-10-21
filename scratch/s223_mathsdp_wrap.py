"""§223 Note 223-4 learning pilot on the RTX 2070 (Audit 10-21: accepted, both cells on the same card / kernel): Turing has no bf16 memory-efficient
SDPA kernel, so the math kernel is forced (flash / mem-efficient off). Unlike scratch/s222_det_wrap.py, deterministic algorithms are NOT forced
(a learnability pilot, not a bitwise gate). The target runs unchanged as __main__ with the remaining argv.
    python -B scratch/s223_mathsdp_wrap.py boxes/codi_ps.py --direct ..."""
import runpy
import sys

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_math_sdp(True)

if __name__ == "__main__":
    target = sys.argv[1]; sys.argv = [target] + sys.argv[2:]
    print("MATHSDP_WRAP math SDP only; target", target, flush=True)
    runpy.run_path(target, run_name="__main__")
