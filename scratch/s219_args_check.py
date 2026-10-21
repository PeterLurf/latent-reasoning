"""§219 ARGS-DIFF check for the filler (TS_v1 §7 G-T5 / §4.6; Notes 219-1 (3), 219-3 (3)): one host args.json (a path, or - for stdin) ->
  (1) SCORER: theory/score_s219.args_diff(lin s1, role t1|t2, that dict) - the registered rule itself (imported read-only from the file; score_s219's main()
      is never called and nothing is written). The host smokes (G-T9 mem, G-T10 gtime) use smoke=True: stop_at / mem_check_self free (score_s219.SMOKE_FREE);
      self_pool keeps the round's REGISTERED value (the host smokes run on the registered pool);
  (2) FROZEN: every key outside score_s219.META_T equal to the frozen args (Theory's sha-frozen copy of the cell's CPU args.json, scratch/s219_args_draft.py),
      except, for the smokes only, the probe keys they set (mem: mem_check_self, mem_check_steps; gtime: none beyond META time_steps).
One line on stdout:  ARGS-CHECK <role> OK | BAD scorer_ok <bool> scorer_bad [...] frozen_bad [...] | UNREADABLE <why>   exit 0 OK / 1 BAD / 2 unreadable.
    sq ... "cat /workspace/Latent/boxes/runs_fsq/s219_t1_s1/args.json" | python scratch/s219_args_check.py cell - [--round t1] [--frozen scratch/s219_args_ref_t1.json]"""
import argparse
import hashlib
import importlib.util
import json
import sys

ROOT = "D:/Latent"
SMOKE_PROBE_KEYS = {"mem": {"mem_check_self", "mem_check_steps"}, "gtime": set(), "cell": set()}


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_s219_ro", ROOT + "/theory/score_s219.py"); S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S); return S


def check(role, a, frozen, rnd="t1", S=None):
    """-> (ok, line). role mem | gtime | cell; a = the host args dict; frozen = the frozen args dict (or None: FROZEN skipped and the result BAD)."""
    S = S or load_scorer(); smoke = role != "cell"
    d = S.args_diff("s1", rnd, a=a, smoke=smoke)
    if d is None:
        return False, "ARGS-CHECK %s UNREADABLE scorer args_diff returned None (base args.json unreadable?)" % role
    sbad = sorted(set(d["shared_diffs"]) | set(d["dropped_base_keys"]) | set(d["neutral_bad"]) | set(d["unexpected_new_keys"]) | set(d["required_bad"])
                  | (set() if d["args_ref"]["ok"] else {"args_ref"}))
    if frozen is None:
        fbad = ["<no frozen ref>"]
    else:
        skip = set(S.META_T) | SMOKE_PROBE_KEYS[role]
        fbad = sorted(k for k in set(frozen) | set(a) if k not in skip and frozen.get(k) != a.get(k))
    ok = bool(d["ok"]) and not sbad and not fbad
    det = {k: [frozen.get(k) if frozen else None, a.get(k)] for k in fbad if frozen}
    return ok, "ARGS-CHECK %s %s scorer_ok %s scorer_bad %s frozen_bad %s%s" % (role, "OK" if ok else "BAD", d["ok"], sbad, fbad,
                                                                              "" if ok else " detail %s args_ref %s" % (json.dumps(det, default=str)[:600], d["args_ref"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("role", choices=["mem", "gtime", "cell"]); ap.add_argument("args_json")
    ap.add_argument("--round", default="t1", choices=["t1", "t2"]); ap.add_argument("--frozen", default="scratch/s219_args_ref_t1.json")
    a = ap.parse_args()
    try:
        txt = sys.stdin.read() if a.args_json == "-" else open(a.args_json, encoding="utf-8").read(); host = json.loads(txt)
        assert isinstance(host, dict)
    except Exception as e:                                                  # noqa: BLE001
        print("ARGS-CHECK %s UNREADABLE host args (%r)" % (a.role, e)); sys.exit(2)
    fp = a.frozen if a.frozen.startswith(("/", "D:", "C:")) else ROOT + "/" + a.frozen
    try:
        raw = open(fp, "rb").read(); frozen = json.loads(raw); fsha = hashlib.sha256(raw).hexdigest()[:12]
    except Exception as e:                                                  # noqa: BLE001
        print("ARGS-CHECK %s UNREADABLE frozen ref %s (%r)" % (a.role, a.frozen, e)); sys.exit(2)
    ok, line = check(a.role, host, frozen, a.round)
    print(line + " (frozen %s sha %s)" % (a.frozen, fsha)); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
