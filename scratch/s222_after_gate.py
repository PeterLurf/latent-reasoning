"""§222 Note 222-3 (b) / Audit 10-21: run a target script only if a gate JSON passed. b320's group line runs scratch/s222_cert.py through this, so a
failed registered-kernel envelope / resume test (s222_envelope_3090/envelope.json proposed_PASS false, or the file missing) stops the host before any
10k-step group. The target runs unchanged as __main__ with the remaining argv (gen_vast_job's appended --device / --out pass through).
    python -m scratch.s222_after_gate --gate boxes/runs_fsq/s222_envelope_3090/envelope.json --target scratch/s222_cert.py --mode cert ...   (exit 3 = gated)"""
import json
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)


def main():
    av = sys.argv[1:]; gate = av[av.index("--gate") + 1]; target = av[av.index("--target") + 1]
    rest = [x for i, x in enumerate(av) if x not in ("--gate", "--target") and (i == 0 or av[i - 1] not in ("--gate", "--target"))]
    try:
        ok = bool(json.load(open(gate)).get("proposed_PASS"))
    except (OSError, ValueError) as e:
        ok = False; print("GATE unreadable", gate, e, flush=True)
    if not ok:
        print("GATE_FAIL", gate, "- the target is NOT run (Audit 10-21: no 10k-step group after a failed resume / envelope test)", flush=True); sys.exit(3)
    print("GATE_PASS", gate, "-> running", target, flush=True)
    sys.argv = [target] + rest; runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
