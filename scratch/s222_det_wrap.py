"""§222 gate-only determinism wrapper (Audit 10-21 (a'), engine equivalence + the t >= 2 resume test): force the math SDP kernel (memory-efficient and
flash off) and torch.use_deterministic_algorithms(True) (strict; CUBLAS_WORKSPACE_CONFIG :4096:8), then run the target script unchanged as __main__ with
the remaining argv. GATES ONLY: the registered §222 runs keep the default (memory-efficient) path, as §218.
    python -B scratch/s222_det_wrap.py scratch/s222_cert.py --mode cert ...      (the target's own sha is unchanged; its resume sha check still holds)"""
import os
import runpy
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True)


def apply():
    """for in-process callers (s222_engine_equiv.py --det): the same flags, idempotent."""
    torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    target = sys.argv[1]; sys.argv = [target] + sys.argv[2:]
    print("DET_WRAP math SDP, deterministic algorithms strict; target", target, flush=True)
    runpy.run_path(target, run_name="__main__")
