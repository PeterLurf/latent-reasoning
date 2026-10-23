"""§228 args checks (Note 228-1; LAT_v1 §8 frozen args + single-change diffs).
  check:  one host args.json (stdin '-' or a path) vs the cell's frozen args (scratch/s228_args_frozen_<run>.json): every key outside CONTROL equal
          (micro / eval_bs are host-sized by --auto_micro and are CONTROL) -> ARGS-CHECK OK | BAD <keys> | UNREADABLE <why>; exit 0 / 1 / 2.
  diff:   two frozen files -> the non-CONTROL diff (LAT vs DIRECT must be {arm, think_k, aux_step, aux_reweight, fast_add, cache_lower, mem_write,
          write_mix_start, write_mix_end, format} + any key named in --allow).
    python scratch/s228_args_check.py check - scratch/s228_args_frozen_s228_plat2_s1.json
    python scratch/s228_args_check.py diff <lat.json> <dir.json> [--allow out]"""
import json
import sys

CONTROL = {"device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "cudnn_version", "image_tag",
           "numpy_version", "platform", "hostname", "hostprobe_sps", "micro", "eval_bs", "out", "resume", "args_only", "args_ref", "auto_micro", "mem_check",
           "time_steps", "stop_at", "mem_budget", "log_every", "ckpt_every", "set_build_sec", "module_sha256", "eval_module_sha256", "nested170_sha256",
           "deltamem_sha256", "ow_words_sha256", "nested_words_sha256"}
SHAS = ("module_sha256", "eval_module_sha256", "nested170_sha256", "deltamem_sha256", "ow_words_sha256", "nested_words_sha256")
LAT_VS_DIR = {"arm", "think_k", "aux_step", "aux_reweight", "fast_add", "cache_lower", "mem_write", "write_mix_start", "write_mix_end", "format"}


def diff(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in CONTROL and a.get(k) != b.get(k)}


def check(src, fro):
    try:
        h = json.load(sys.stdin if src == "-" else open(src)); f = json.load(open(fro))
    except Exception as e:
        print("ARGS-CHECK UNREADABLE", e); sys.exit(2)
    bad = diff(f, h)
    for k in SHAS:                                                          # the shipped code must be the committed code the freeze saw
        if f.get(k) != h.get(k):
            bad[k] = [f.get(k), h.get(k)]
    print("ARGS-CHECK OK" if not bad else "ARGS-CHECK BAD " + json.dumps(bad)[:600], "(frozen %s; torch %s; micro %s; eval_bs %s)" % (fro, h.get("torch_version"), h.get("micro"), h.get("eval_bs")))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "diff":
        al = set(sys.argv[sys.argv.index("--allow") + 1].split(",")) if "--allow" in sys.argv else set()
        d = diff(json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))); extra = set(d) - LAT_VS_DIR - al
        print(json.dumps({"diff": d, "outside_registered": sorted(extra), "pass": not extra}, indent=1)); sys.exit(0 if not extra else 1)
