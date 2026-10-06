"""THEORY §130 local runner (one card): python scratch/run130.py --card 4060 --device cuda:0   |   --card 3070 --device cuda:1
Runs scratch/replay_basin.py unchanged (--seed S --steps 2000 --every 50 --amp bf16 --threads 3) into
theory/results/replay130/{card}_s{seed}.jsonl, one seed at a time, skipping complete outputs (restart-safe).
4060: FIT 3..10 (+ 35..42 if FIT has < 3 flagged or < 3 non-flagged), then TEST-SAME 19..26, 59..82 until >= 8 flagged and >= 8
non-flagged. 3070: TEST-CROSS 11..18, 27..34, 43..58 until >= 8 / >= 8 or the list ends.
BLINDING: block completion counts only the 123-3 flag computed from each replay's own LOSS lines (score_s130.features' formula);
no zb_std_in / L1_dmean / marker value is read here. Logging (Audit, blinding): logs/c1_local.log gets the flag for FIT seeds
only; TEST seeds are logged there as "complete", their flags only in theory/results/replay130/.blind_runner.log (not opened before
note 130-1). Stops when scratch/run130.stop exists."""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "theory", "results", "replay130")


def flag_from_loss(path):
    """123-3 flag on the replay's own loss (nearest logged steps to 100 / 1k / 2k), as score_s130.features; None if incomplete."""
    if not os.path.exists(path):
        return None
    L = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if "step" in r and r.get("loss") is not None:
                L[r["step"]] = r["loss"]
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


FIT = set(range(3, 11)) | set(range(35, 43))                              # 4060 FIT seeds: the only ones whose flag is logged openly


def run(card, seed, dev, log):
    out = os.path.join(D, "%s_s%d.jsonl" % (card, seed))
    f = flag_from_loss(out)
    if f is not None:
        return f
    if os.path.exists(out):
        os.replace(out, out + ".void_incomplete_%d" % int(time.time()))   # an interrupted replay: kept, rerun
    t0 = time.time()
    rc = subprocess.call([sys.executable, "-u", os.path.join(ROOT, "scratch", "replay_basin.py"), "--seed", str(seed), "--steps", "2000",
                          "--every", "50", "--amp", "bf16", "--threads", "3", "--device", dev, "--out", out],
                         stdout=open(os.path.join(ROOT, "logs", "c1_s130_%s_s%d.log" % (card, seed)), "w"), stderr=subprocess.STDOUT, cwd=ROOT)
    f = flag_from_loss(out); fit = card == "4060" and seed in FIT
    log.write("%s s130 %s s%d rc %d %s (%s, %ds)\n" % (time.strftime("%H:%M:%S"), card, seed, rc, ("flag %s" % f) if fit else "complete", dev, time.time() - t0)); log.flush()
    if not fit:                                                             # TEST: the flag goes to the blind log only (Audit)
        with open(os.path.join(D, ".blind_runner.log"), "a") as bl:
            bl.write("%s %s s%d flag %s\n" % (time.strftime("%H:%M:%S"), card, seed, f))
    return f


def block(card, seeds, dev, log, need=None):
    """Run seeds in order; with need, stop once >= need flagged and >= need non-flagged."""
    res = {}
    for s in seeds:
        if os.path.exists(os.path.join(ROOT, "scratch", "run130.stop")):
            break
        res[s] = run(card, s, dev, log)
        if need and sum(v is True for v in res.values()) >= need and sum(v is False for v in res.values()) >= need:
            break
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--card", required=True, choices=["4060", "3070"]); ap.add_argument("--device", required=True)
    a = ap.parse_args(); os.makedirs(D, exist_ok=True); log = open(os.path.join(ROOT, "logs", "c1_local.log"), "a")
    if a.card == "4060":
        fit = block(a.card, list(range(3, 11)), a.device, log)
        if sum(v is True for v in fit.values()) < 3 or sum(v is False for v in fit.values()) < 3:
            fit.update(block(a.card, list(range(35, 43)), a.device, log))
        log.write("%s s130 4060 FIT done: %d flagged / %d non-flagged\n" % (time.strftime("%H:%M:%S"), sum(v is True for v in fit.values()),
                                                                              sum(v is False for v in fit.values()))); log.flush()
        block(a.card, list(range(19, 27)) + list(range(59, 83)), a.device, log, need=8)
    else:
        block(a.card, list(range(11, 19)) + list(range(27, 35)) + list(range(43, 59)), a.device, log, need=8)
    log.write("%s s130 %s RUNNER_DONE\n" % (time.strftime("%H:%M:%S"), a.card)); log.flush()


if __name__ == "__main__":
    main()
