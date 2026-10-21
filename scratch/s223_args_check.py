"""§223 ARGS-DIFF check for the filler (PS2_v1 Addendum A item 3; Note 223-3 frozen refs): one host args.json (stdin, or a path) against the cell's
frozen args (scratch/s223_args_frozen_<cell>.json). Every key outside CONTROL must be equal. CONTROL = codi_ps.META (host / version / micro keys) +
the derived 'cell' label + the run-control flags that change how a run is driven, never what it trains (args_only, args_ref, resume, auto_micro,
mem_check, time_steps, stop_at, mem_budget). One line on stdout:  ARGS-CHECK OK | BAD <keys> | UNREADABLE <why>; exit 0 / 1 / 2.
    ssh ... "cat /workspace/Latent/boxes/runs_fsq/s223_direct_s1/args.json" | python scratch/s223_args_check.py - scratch/s223_args_frozen_direct.json"""
import json
import sys

sys.path.insert(0, "D:/Latent")
META = ("device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "micro", "accumulation", "out", "resume_from")
CONTROL = set(META) | {"cell", "args_only", "args_ref", "resume", "auto_micro", "mem_check", "time_steps", "stop_at", "mem_budget"}


def main():
    src, fro = sys.argv[1], sys.argv[2]
    try:
        h = json.load(sys.stdin if src == "-" else open(src)); f = json.load(open(fro))
    except Exception as e:
        print("ARGS-CHECK UNREADABLE", e); sys.exit(2)
    bad = {k: [f.get(k), h.get(k)] for k in sorted(set(h) | set(f)) if k not in CONTROL and h.get(k) != f.get(k)}
    print("ARGS-CHECK OK" if not bad else "ARGS-CHECK BAD " + json.dumps(bad)[:600], "(frozen %s; transformers %s; micro %s)" % (fro, h.get("transformers_version"), h.get("micro")))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
