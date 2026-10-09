"""THEORY §176 fixed-K reference (2a1c3be5): post-hoc eval of a §169 K8 model (eval only) -> <run>/eval_s176_ref.json
    acc_by_kprime / ext_4x8_by_kprime at K' in {24, 32, 48, 64} (the same items, seeds and predict path as inject169's eval: the 4x4 bench test_n, ext_4x8
    from gen_operands seeds 14004 / 14005), mem_norm_by_pass to 64 (median per-row norms of the additive / delta blocks on the first 256 bench items).
    python scratch/s176_ref.py <run dir> [device]"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.inject169 as I  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import gen_operands, load_bench  # noqa: E402
from boxes.fixedchunk import product_digits, amp_ctx  # noqa: E402

KPS = (24, 32, 48, 64)


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; t0 = time.time()
    try:
        from boxes import memcap
        memcap.apply(dev)
    except ImportError:
        pass
    a = json.load(open(os.path.join(run, "args.json"))); assert a.get("arm") == "inject" and int(a["think_k"]) == 8, "a §169 K8 run"
    DM.FAST_ADD = bool(a.get("fast_add"))
    amp = None if (a.get("fp32") or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    m = I.Inject(s5.load_gpt2(dev), 6, a.get("mem_write", "top")).to(dev)
    if a.get("pass_index"):
        m.pidx = torch.nn.Embedding(int(a["pass_index"]), 768).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    bench = a.get("bench") or os.path.join("bench", "data", "mult_4x4_test.txt"); teA, teB = load_bench(bench if os.path.isabs(bench) else os.path.join(ROOT, bench))
    teA, teB = teA[:a.get("test_n", 1000)], teB[:a.get("test_n", 1000)]; n = a.get("ext_n", 1000); bs = a.get("eval_bs", 250)
    xA, xB = gen_operands(np.random.default_rng(14004), n, 4), gen_operands(np.random.default_rng(14005), n, 8); P4, P8 = product_digits(teA, teB), product_digits(xA, xB)
    res = {"run": run.replace(os.sep, "/"), "kprimes": list(KPS), "acc_by_kprime": {}, "ext_4x8_by_kprime": {}}
    for kp in KPS:
        res["acc_by_kprime"][str(kp)] = float((I.predict(m, teA, teB, kp, dev, amp, bs)[0] == P4).all(1).mean())
        res["ext_4x8_by_kprime"][str(kp)] = float((I.predict(m, xA, xB, kp, dev, amp, bs)[0] == P8).all(1).mean()); print(kp, round(time.time() - t0), flush=True)
    mn = {}
    with torch.no_grad():
        nA, nB = teA[:256], teB[:256]; S = m.mem.zero_state(len(nA), dev); ids = m.ids(torch.tensor(I.think_text(nA, nB), device=dev))
        for t in range(1, 65):
            with amp_ctx(amp):
                S = m.write(S, ids, t)
            mn[str(t)] = {"additive": float(S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()), "delta": float(S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())}
    res.update({"mem_norm_by_pass": mn, "eval_dtype": str(amp), "fast_add": DM.FAST_ADD, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                "sec": round(time.time() - t0)})
    json.dump(res, open(os.path.join(run, "eval_s176_ref.json"), "w", newline="\n"), indent=1); print("DONE", run, flush=True)


if __name__ == "__main__":
    main()
