"""§224 args checks (FH_v1 §6; Theory 10-21: Stage-1 args diffs are exactly seed / init_from / out / --far / --fh).
  check:  one host args.json (stdin '-' or a path) vs the cell's frozen args (scratch/s224_args_frozen_<run>.json): every key outside CONTROL equal, and
          micro == 1 (FT-CTRL s1's micro; G-FH-EQ needs equal micro) -> one line ARGS-CHECK OK | BAD <keys> | UNREADABLE <why>; exit 0 / 1 / 2.
          ssh ... "cat .../s224_fhft_s1/args.json" | python scratch/s224_args_check.py check - scratch/s224_args_frozen_s224_fhft_s1.json
  diffs:  the single-change diffs over the frozen files -> scratch/gate_s224/args_diffs.json: FH vs RD (same seed) = {fh}; RD vs FT-CTRL (s1: vs
          s215_ftctrl_s1's args; s2: vs s224_ftctrl_s2's frozen) = {far} plus, against the pre-§224 s215 file only, the §224 keys absent there at their
          defaults; FT-CTRL s2 vs s215_ftctrl_s1 = {seed, init_from} (+ the same default keys). CONTROL keys are ignored throughout.
          python scratch/s224_args_check.py diffs"""
import json
import sys

CONTROL = {"device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "cudnn_version", "image_tag",
           "numpy_version", "platform", "hostname", "micro", "out", "resume", "auto_micro", "mem_check", "mem_check_steps", "time_steps", "stop_at",
           "mem_budget", "args_ref", "keep_ckpt_steps", "log_every", "ckpt_every"}
NEW_DEFAULTS = {"far": False, "fh": False, "far_n": 16, "far_t": "13,48", "far_w": 1.0, "fh_dose": "250,500", "fh_target": 0.2, "fh_lam_cap": 10.0}
REF = "scratch/s224_args_ref_ftctrl_s1.json"; FZ = "scratch/s224_args_frozen_%s.json"


def diff(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in CONTROL and a.get(k) != b.get(k)}


def check(src, fro):
    try:
        h = json.load(sys.stdin if src == "-" else open(src)); f = json.load(open(fro))
    except Exception as e:
        print("ARGS-CHECK UNREADABLE", e); sys.exit(2)
    bad = diff(f, h)
    if h.get("micro") != 1:
        bad["micro"] = [1, h.get("micro")]
    print("ARGS-CHECK OK" if not bad else "ARGS-CHECK BAD " + json.dumps(bad)[:600], "(frozen %s; torch %s; micro %s)" % (fro, h.get("torch_version"), h.get("micro")))
    sys.exit(0 if not bad else 1)


def diffs():
    ld = lambda r: json.load(open(FZ % r))  # noqa: E731
    ref = json.load(open(REF)); o = {}

    def only(d, want, allow_new=False):
        extra = {k: v for k, v in d.items() if k not in want and not (allow_new and k in NEW_DEFAULTS and v[0] is None and v[1] == NEW_DEFAULTS[k])}
        return set(want) <= set(d) and not extra, extra
    for s in ("s1", "s2"):
        d = diff(ld("s224_rdft_" + s), ld("s224_fhft_" + s)); ok, ex = only(d, {"fh"}); o["FH_vs_RD_" + s] = {"diff": d, "pass": ok and d["fh"] == [False, True], "extra": ex}
    d = diff(ref, ld("s224_rdft_s1")); ok, ex = only(d, {"far"}, True); o["RD_vs_FTCTRL_s1(s215)"] = {"diff": d, "pass": ok and d["far"] == [None, True], "extra": ex}
    d = diff(ld("s224_ftctrl_s2"), ld("s224_rdft_s2")); ok, ex = only(d, {"far"}); o["RD_vs_FTCTRL_s2"] = {"diff": d, "pass": ok and d["far"] == [False, True], "extra": ex}
    d = diff(ref, ld("s224_ftctrl_s2")); ok, ex = only(d, {"seed", "init_from"}, True); o["FTCTRL_s2_vs_s215"] = {"diff": d, "pass": ok, "extra": ex}
    o["pass"] = all(v["pass"] for v in o.values() if isinstance(v, dict))
    json.dump(o, open("scratch/gate_s224/args_diffs.json", "w", newline="\n"), indent=1); print("ARGS-DIFFS pass", o["pass"])


if __name__ == "__main__":
    check(sys.argv[2], sys.argv[3]) if sys.argv[1] == "check" else diffs()
