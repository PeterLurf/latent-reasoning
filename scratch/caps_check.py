"""Template v6 items 1 + 5 (Audit 11-03 LAUNCH_POSTMORTEM §3.1 / §3.5): the queue carries its caps; the filler never takes hand-typed ones.
  mkq side   caps_line(wallw, eval_h, gates_h, sps_est, hostprobe, gpu, fetch_gb) -> '# CAPS wallw=.. eval_h=.. gates_h=.. sps_est=.. hostprobe=.. class=.. fetch_gb=..'
             fetch_gb(queue text, base) -> the GB the drain fetches (model.pt of every dir that writes one and is not '# NOFETCH'ed; + ckpt_final under --keep_final_ckpt).
  filler     python scratch/caps_check.py <queue> <STEPS> <CAPC> <PMAX> <SETUP_H>   (env WALLW / EVAL_H / GATES_H / SPS_EST / GPU optional)
             -> stdout 'WALLW=..; EVAL_H=..; GATES_H=..; SPS_EST=..; FETCH_H=..; WALLW_G=..' (eval'd by the filler; WALLW_G = WALLW with FETCH_H added to the last
             line's weight, for the wall guard) or exit 2 with 'REFUSED: <why>' when: the header is missing / incomplete; an env value disagrees with it; the weight count
             != the queue's lines; hostprobe != the queue's '# HOSTPROBE <lim>' (default 0.854); PMAX > CAPC / (STEPS x SPS_EST / 3600 + SETUP_H + EVAL_H + GATES_H + FETCH_H).
FETCH_H = fetch_gb / FETCH_GBPH (env, default 10 GB/h)."""
import os
import re
import sys

KEYS = ("wallw", "eval_h", "gates_h", "sps_est", "hostprobe", "class", "fetch_gb"); ENV = {"wallw": "WALLW", "eval_h": "EVAL_H", "gates_h": "GATES_H", "sps_est": "SPS_EST", "class": "GPU"}
GB = {"gpt2": 0.51, "gpt2-medium": 1.42}; CKPT = {"gpt2": 1.52, "gpt2-medium": 4.3}


def caps_line(wallw, eval_h, gates_h, sps_est, hostprobe, gpu, fetch):
    return "# CAPS wallw=%s eval_h=%s gates_h=%s sps_est=%s hostprobe=%s class=%s fetch_gb=%s" % (",".join(str(w) for w in wallw), eval_h, gates_h, sps_est, hostprobe, gpu, fetch)


def fetch_gb(q, base="gpt2"):
    nof = {ln.split()[2] for ln in q.split("\n") if ln.startswith("# NOFETCH ")}; dirs = {}
    for ln in q.split("\n"):
        t = ln.split()
        if len(t) < 2 or ln.startswith("#"):
            continue
        d, mod = t[0], t[1]
        if mod in ("boxes.inject_aw", "scratch.s257_guarded", "scratch.s259_guarded", "scratch.s268_stopland") and "--mem_check" not in t and "--time_steps" not in t:
            dirs[d] = dirs.get(d, 0.0) + (0.0 if d + "/model.pt" in nof or d in dirs else GB[base]) + (CKPT[base] if "--keep_final_ckpt" in t else 0.0)
    return round(sum(dirs.values()), 2)


def check(qp, steps, capc, pmax, setup_h):
    q = open(qp, encoding="utf-8").read(); hdr = [ln for ln in q.split("\n") if ln.startswith("# CAPS ")]
    if len(hdr) != 1:
        return None, "no single '# CAPS' header in %s (template v6: the caps come from the queue)" % qp
    kv = dict(x.split("=", 1) for x in hdr[0][len("# CAPS "):].split()); miss = [k for k in KEYS if k not in kv]
    if miss:
        return None, "CAPS lacks %s" % ",".join(miss)
    for k, e in ENV.items():
        v = os.environ.get(e, "")
        if v and v != kv[k]:
            try:
                same = k not in ("wallw", "class") and abs(float(v) - float(kv[k])) < 1e-9
            except ValueError:
                same = False
            if not same:
                return None, "env %s=%s != CAPS %s=%s (template v6: drop the env value or rebuild the queue)" % (e, v, k, kv[k])
    w = [float(x) for x in kv["wallw"].split(",")]; nl = len([ln for ln in q.split("\n") if ln.strip() and not ln.startswith("#")])
    if len(w) != nl:
        return None, "CAPS wallw has %d weights, the queue %d lines" % (len(w), nl)
    hp = re.search(r"^# HOSTPROBE ?([0-9.]*)", q, re.M); hp = (hp.group(1) or "0.854") if hp else None
    if hp is not None and float(hp) != float(kv["hostprobe"]):
        return None, "CAPS hostprobe %s != the queue's HOSTPROBE %s" % (kv["hostprobe"], hp)
    fh = round(float(kv["fetch_gb"]) / float(os.environ.get("FETCH_GBPH", "10")), 3)
    h = steps * float(kv["sps_est"]) / 3600 + setup_h + float(kv["eval_h"]) + float(kv["gates_h"]) + fh
    if pmax > capc / h + 1e-9:
        return None, "PMAX $%s > admit $%s / %.2f h (steps x sps_est + setup + eval_h + gates_h + fetch %.2f h)" % (pmax, capc, h, fh)
    wg = w[:-1] + [round(w[-1] + fh, 3)]
    return ("WALLW=%s; EVAL_H=%s; GATES_H=%s; SPS_EST=%s; GPU=%s; FETCH_H=%s; WALLW_G=%s" % (kv["wallw"], kv["eval_h"], kv["gates_h"], kv["sps_est"], kv["class"], fh,
                                                                                         ",".join(str(x) for x in wg))), None


if __name__ == "__main__":
    out, err = check(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    if err:
        print("REFUSED: " + err); sys.exit(2)
    print(out)
