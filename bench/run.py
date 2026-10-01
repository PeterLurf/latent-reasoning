"""Run one model on one task and write a self-describing result file.

    python -m bench.run --task prosqa_na --model "coconut:prosqa/ckpt/checkpoint_50?K=6" --device cuda:0
    python -m bench.run --task prosqa --model shortcut
    python -m bench.run --list

The result file (bench/results/<task>__<tag>.json) holds accuracy with a 95% Wilson interval, accuracy by depth, mean
compute, the no-reasoning baselines on the same items, every per-item outcome, and provenance (git commit, data and
checkpoint fingerprints, command line), so any number in a report can be traced and recomputed.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

from bench import models, tasks

ROOT = tasks.ROOT


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fingerprint(path):
    """Size, mtime and a hash of the first and last 4 MB: cheap enough for 500 MB checkpoints, distinct in practice."""
    if os.path.isdir(path):
        path = next((os.path.join(path, f) for f in ("model.pt", "pytorch_model.bin") if os.path.exists(os.path.join(path, f))), path)
    if not os.path.isfile(path):
        return None
    st = os.stat(path); h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 << 20)); f.seek(max(0, st.st_size - (4 << 20))); h.update(f.read(4 << 20))
    return {"path": os.path.relpath(path, ROOT).replace("\\", "/"), "bytes": st.st_size, "head_tail_sha256": h.hexdigest()[:16]}


def select(items, n, per_depth, seed):
    if per_depth:
        rng = np.random.default_rng(seed); out = []
        for d in sorted({it["depth"] for it in items}):
            grp = [it for it in items if it["depth"] == d]
            idx = rng.permutation(len(grp))[:per_depth]
            out += [grp[i] for i in sorted(idx)]
        items = out
    if n and n < len(items):
        rng = np.random.default_rng(seed)
        items = [items[i] for i in sorted(rng.permutation(len(items))[:n])]
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task"); ap.add_argument("--model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=0, help="random subset of this size (0 = all)")
    ap.add_argument("--per_depth", type=int, default=0, help="at most this many items per depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", help="name for the result file (default: derived from the model spec)")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "bench", "results"))
    ap.add_argument("--threads", type=int, default=3, help="CPU threads (keeps local GPU queues fed)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for t in tasks.TASKS.values():
            print(f"{t.name:15s} {t.source}  [depth: {t.depth}]")
        return
    import torch
    torch.set_num_threads(a.threads)
    task = tasks.TASKS[a.task]
    items, files = task.load()
    items = select(items, a.n, a.per_depth, a.seed)
    model = models.load(a.model, task, a.device)
    items = [it for it in items if model.supports(it) or isinstance(model, models.Baseline)]
    if not items:
        raise SystemExit(f"model {a.model} supports no items of task {a.task}")
    tag = a.tag or re.sub(r"[^A-Za-z0-9_.=-]+", "_", a.model.replace("prosqa/ckpt/", "").replace("prosqa/runs_ail/", ""))[:80]
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        p = model.predict(it)
        if "score" in p:
            ok = p["score"]
        elif it["kind"] == "choice":
            ok = 1.0 if p["margin"] > 0 else 0.0
        else:
            ok = 1.0 if task.normalise(p["text"]) == task.normalise(it["answer"]) else 0.0
        r = {"id": it["id"], "depth": it["depth"], "correct": ok, **p["compute"]}
        if "margin" in p:
            r["margin"] = round(float(p["margin"]), 4)
        if "text" in p:
            r["text"] = p["text"][:200]
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(items)} acc {np.mean([x['correct'] for x in rows]):.3f} {time.time() - t0:.0f}s", flush=True)
    acc = lambda rs: float(np.mean([x["correct"] for x in rs]))
    depths = sorted({r["depth"] for r in rows if r["depth"] is not None})
    base = {}
    if task.baselines and items[0]["kind"] == "choice":
        for name, f in task.baselines.items():
            base[name] = round(float(np.mean([f(it["raw"]) for it in items])), 4)
    if task.chance is not None:
        base["chance"] = task.chance
    k = sum(r["correct"] for r in rows)
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    _, path, _ = models.parse_spec(a.model)
    out = {
        "task": task.name, "model": a.model, "tag": tag, "n": len(rows),
        "accuracy": round(acc(rows), 4), "ci95": wilson(k, len(rows)),
        "by_depth": {str(d): {"acc": round(acc([r for r in rows if r["depth"] == d]), 4),
                              "n": sum(1 for r in rows if r["depth"] == d)} for d in depths},
        "baselines_same_items": base,
        "mean_compute": {c: round(float(np.mean([r[c] for r in rows])), 2) for c in ("latent_passes", "gen_tokens")},
        "provenance": {"commit": commit, "when": datetime.datetime.now().isoformat(timespec="seconds"),
                       "argv": sys.argv[1:], "device": a.device, "seconds": round(time.time() - t0, 1),
                       "data": [{"path": os.path.relpath(f, ROOT).replace("\\", "/"), "sha256": tasks.sha256(f)[:16]} for f in files],
                       "checkpoint": fingerprint(os.path.join(ROOT, path)) if path else None,
                       "selection": {"n": a.n, "per_depth": a.per_depth, "seed": a.seed}},
        "items": rows,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    fn = os.path.join(a.out_dir, f"{task.name}__{tag}.json")
    with open(fn, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"{task.name} | {a.model} | n {len(rows)} | acc {out['accuracy']:.3f} {out['ci95']} | baselines {base}")
    print("WROTE", os.path.relpath(fn, ROOT))


if __name__ == "__main__":
    main()
