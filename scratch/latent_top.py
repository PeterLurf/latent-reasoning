"""latent_top: an nvtop-style terminal dashboard for the whole Latent plant (user 2026-10-06: "include the current jobs lined up in local
and in cloud, as well as the cloud's gpu usage, all in one place"). nvtop.exe itself is a compiled third-party binary, so this runs beside it.
Panels:
  LOCAL GPUs   nvidia-smi every 2 s: util / VRAM / temp / power bars per card, CUDA index, compute processes;
  LOCAL JOBS   per v2 backlog worker (c4060 / c3070 / c2070): the running job and its age (logs/c1_local.log), then the next runnable lines
               of scratch/c1_backlog_<card>.txt (not done, not claimed, under the card's memory gate - the worker's own rules);
  CLOUD        Vast instances every 30 s (status, GPU util / VRAM / temp from the API, $/h, uptime) + one parallel SSH pass every 90 s per
               host (current run, its last logged step / loss / L, queue progress);
  CLOUD QUEUE  queues held by running fillers (fill_generic / fill135) not deployed yet, and staged gen_q files (< 48 h old, never deployed).
Read-only: it never starts, stops or deploys anything. The Vast API key stays inside the Vast CLI (never read or printed here).
    python scratch/latent_top.py            (Ctrl+C quits)        options: --once (print one frame and exit), --no-ssh"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAST_LIB = "C:/Users/peter/.local/vastai/lib"
SSH = r"C:\Program Files\Git\usr\bin\ssh.exe" if os.path.exists(r"C:\Program Files\Git\usr\bin\ssh.exe") else "ssh"
KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
WORKERS = (("c4060", "4060", "cuda:0"), ("c3070", "3070", "cuda:1"), ("c2070", "2070", "cuda:2"))
NOWIN = 0x08000000 if os.name == "nt" else 0                                # CREATE_NO_WINDOW: no console flashes from child processes
STATE = {"vast": [], "vast_t": 0, "vast_err": None, "credit": None, "ssh": {}, "ssh_t": 0, "fillers": [], "fill_t": 0, "workers": {}}
LOCK = threading.Lock()


def sh(cmd, timeout=20, env=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, creationflags=NOWIN, encoding="utf-8", errors="replace").stdout
    except Exception:
        return ""


def bar(frac, width=18, warn=0.75, crit=0.92):
    frac = max(0.0, min(1.0, frac or 0.0)); n = int(round(frac * width)); col = "red" if frac >= crit else "yellow" if frac >= warn else "green"
    return Text("█" * n, style=col) + Text("░" * (width - n), style="grey30")


def ago(sec):
    sec = int(max(0, sec)); return "%dh%02dm" % (sec // 3600, sec % 3600 // 60) if sec >= 3600 else "%dm%02ds" % (sec // 60, sec % 60)


# ---------------------------------------------------------------- local
def local_gpus():
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,pci.bus_id"
    rows = []
    for ln in sh(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"], 10).splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 10:
            rows.append(dict(zip(q.split(","), p)))
    procs = {}
    for ln in sh(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader,nounits"], 10).splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 4:
            procs.setdefault(p[0], []).append((p[1], os.path.basename(p[2].replace("\\", "/")), p[3]))
    return rows, procs


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def worker_jobs():
    """The running job per worker tag from logs/c1_local.log (its last 'start' with no later 'done')."""
    out = {t: None for t, _, _ in WORKERS}; lp = os.path.join(ROOT, "logs", "c1_local.log")
    try:
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()[-600:]
    except OSError:
        return out
    for ln in lines:
        m = re.match(r"(\d\d:\d\d:\d\d) (c\d{4}) (start|done|IDLE|HALT|backlog worker)(.*)", ln)
        if not m:
            continue
        t, tag, kind, rest = m.groups()
        if tag in out:
            if kind == "start":
                out[tag] = {"t": t, "cmd": rest.lstrip(": ").strip()}
            elif kind in ("done", "IDLE", "HALT") or (kind == "backlog worker" and "exit" in rest):
                out[tag] = {"t": t, "state": kind + rest[:60]} if kind != "done" else None
    return out


def since_hms(hms):
    try:
        h, m, s = map(int, hms.split(":")); now = time.localtime(); d = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec) - (h * 3600 + m * 60 + s)
        return d + 86400 if d < 0 else d
    except ValueError:
        return 0


def short(cmd):
    """A readable label for a job command: its --out (or a done file / --run), plus the module."""
    mo = re.search(r"-m (\S+)", cmd) or re.search(r"python -u (\S+)", cmd); mod = mo.group(1).split(".")[-1].replace(".py", "") if mo else cmd.split()[0]
    for k in ("--out", "--run"):
        mm = re.search(k + r" (\S+)", cmd)
        if mm:
            o = mm.group(1).rstrip("/"); o = o if "theory/results" in o else os.path.basename(o)
            return "%s %s" % (mod, o.replace("theory/results/", ""))
    fl = [w for w in cmd.split() if w.startswith("--")][:3]
    return "%s %s" % (mod, " ".join(fl))


def backlog_next(card, total_mib, n=4):
    """Next runnable lines of a backlog file under the worker's rules (skip done, claimed, over the card's memory gate)."""
    p = os.path.join(ROOT, "scratch", "c1_backlog_%s.txt" % card); nxt = []; left = 0
    try:
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        return nxt, 0
    limit = total_mib - 1536
    for ln in lines:
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split(" ", 2)
        if len(parts) < 3:
            continue
        mem, done_f, cmd = parts; df = done_f if os.path.isabs(done_f) or re.match(r"^[A-Za-z]:", done_f) else os.path.join(ROOT, done_f)
        if os.path.exists(df) and os.path.getsize(df) > 0:
            continue
        if os.path.isdir(os.path.join(ROOT, "c1", "runs", "locks", "bl_" + hashlib.md5((ln + "\n").encode()).hexdigest()[:12])):
            continue
        if num(mem, 99) * 1024 > limit:
            continue
        left += 1
        if len(nxt) < n:
            nxt.append((mem, short(cmd)))
    return nxt, left


# ---------------------------------------------------------------- cloud
def vast(*args, timeout=40):
    env = dict(os.environ, PYTHONPATH=VAST_LIB, PYTHONIOENCODING="utf-8")
    return sh([sys.executable, "-c", "import sys; from vastai.cli.main import main; sys.argv[0]='vastai'; sys.exit(main())", *args], timeout, env)


REMOTE = r"""cd /workspace/Latent 2>/dev/null || { echo NODIR; exit; }; q=$(ls -t logs/queue_*.out 2>/dev/null | head -1)
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | head -1 | sed 's/^/GPU /'
[ -n "$q" ] && echo "Q $(grep -c ' start ' $q) $(grep -c ' evaluated ' $q) $(grep -c QUEUE_DONE $q)"
cur=$( [ -n "$q" ] && grep ' start ' $q | tail -1 | awk '{print $3}'); echo "CUR $cur"
[ -n "$cur" ] && [ -f "$cur/log.jsonl" ] && tail -c 6000 "$cur/log.jsonl" | grep '"loss"' | tail -1 | sed 's/^/LAST /'
ls /workspace/Latent/DRAINED* 2>/dev/null | head -1 | sed 's/^/DRAINED /'"""


def ssh_host(ip, port):
    out = sh([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", "-i", KEY, "-p", str(port), "root@" + ip, REMOTE], 25)
    r = {"t": time.time(), "ok": bool(out.strip())}
    for ln in out.splitlines():
        if ln.startswith("GPU "):
            p = [x.strip() for x in ln[4:].split(",")]; r["gpu"] = p
        elif ln.startswith("Q "):
            r["q"] = [int(x) for x in ln.split()[1:4]]
        elif ln.startswith("CUR "):
            r["cur"] = ln[4:].strip()
        elif ln.startswith("LAST "):
            try:
                r["last"] = json.loads(ln[5:])
            except ValueError:
                pass
        elif ln.startswith("NODIR"):
            r["nodir"] = True
    return r


def queue_lines(tag):
    try:
        return [l for l in open(os.path.join(ROOT, "scratch", "gen_q_%s.txt" % tag), encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("#")]
    except OSError:
        return []


def fillers():
    """Running filler scripts (fill_generic.sh <tag> <n...> / fill135.sh) and their not-yet-deployed queues."""
    ps = "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -match 'fill_generic|fill135' } | % { $_.CommandLine }"
    out = sh(["powershell", "-NoProfile", "-Command", ps], 20); res = []
    try:
        dep = open(os.path.join(ROOT, "logs", "gen_deploy.log"), encoding="utf-8", errors="replace").read()
    except OSError:
        dep = ""
    seen = set()
    for ln in out.splitlines():
        m = re.search(r"fill_generic\.sh\s+(\S+)\s+([\d\s]+)", ln)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); ns = m.group(2).split()
            pend = [n for n in ns if not re.search(r"gen host b%s offer \S+: .*STARTED" % n, dep)]
            res.append(("fill_" + m.group(1), pend))
        elif "fill135" in ln and "fill135" not in seen:
            seen.add("fill135"); res.append(("fill135", []))
    return res, dep


def cloud_loop(use_ssh):
    while True:
        t0 = time.time()
        try:
            raw = vast("show", "instances", "--raw"); inst = json.loads(raw) if raw.strip().startswith("[") else None
            if inst is not None:
                with LOCK:
                    STATE["vast"] = inst; STATE["vast_t"] = time.time(); STATE["vast_err"] = None
            else:
                with LOCK:
                    STATE["vast_err"] = "vast CLI: no instance list"
            u = vast("show", "user", "--raw")
            try:
                with LOCK:
                    STATE["credit"] = json.loads(u).get("credit")
            except ValueError:
                pass
            fl, dep = fillers()
            with LOCK:
                STATE["fillers"] = fl; STATE["deploy_log"] = dep; STATE["fill_t"] = time.time()
            if use_ssh and inst and time.time() - STATE["ssh_t"] > 85:
                hosts = []
                for o in inst:
                    pp = (o.get("ports") or {}).get("22/tcp") or []
                    if o.get("actual_status") == "running" and o.get("public_ipaddr") and pp:
                        hosts.append((o["id"], o["public_ipaddr"].strip(), pp[0]["HostPort"]))
                with cf.ThreadPoolExecutor(16) as ex:
                    res = dict(zip([h[0] for h in hosts], ex.map(lambda h: ssh_host(h[1], h[2]), hosts)))
                with LOCK:
                    STATE["ssh"].update(res); STATE["ssh_t"] = time.time()
        except Exception as ex:                                             # the dashboard must survive any transient failure
            with LOCK:
                STATE["vast_err"] = repr(ex)[:120]
        time.sleep(max(5, 30 - (time.time() - t0)))


# ---------------------------------------------------------------- render
def render(gpus, procs):
    now = time.time()
    # LOCAL GPUs
    tg = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    for c, j in (("GPU", "left"), ("CUDA", "left"), ("util", "left"), ("", "right"), ("VRAM", "left"), ("", "right"), ("temp", "right"), ("power", "right"), ("fan", "right"), ("processes", "left")):
        tg.add_column(c, justify=j, no_wrap=True)
    for g in gpus:
        cuda = next((d for _, n, d in WORKERS if n in g["name"]), "?"); mu, mt = num(g["memory.used"]), num(g["memory.total"], 1)
        pr = procs.get(g["pci.bus_id"], []); ptxt = ", ".join("%s %s" % (p[1].replace(".exe", ""), p[0]) for p in pr[:3]) or "-"
        tg.add_row("%s %s" % (g["index"], g["name"].replace("NVIDIA GeForce ", "")), cuda, bar(num(g["utilization.gpu"]) / 100, 14), "%3d%%" % num(g["utilization.gpu"]),
                   bar(mu / mt, 14, 0.80, 0.90), "%.1f/%.0fG" % (mu / 1024, mt / 1024), "%d°C" % num(g["temperature.gpu"]),
                   "%.0f/%.0fW" % (num(g["power.draw"]), num(g["power.limit"])), "%s%%" % g["fan.speed"].replace("[N/A]", "-"), ptxt)
    # LOCAL JOBS
    jobs = worker_jobs(); tj = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    tj.add_column("worker", no_wrap=True); tj.add_column("running", ratio=3); tj.add_column("age", justify="right", no_wrap=True); tj.add_column("next up (runnable)", ratio=4); tj.add_column("left", justify="right")
    for tag, card, cuda in WORKERS:
        g = next((x for x in gpus if card in x["name"]), None); total = int(num(g["memory.total"], 8192)) if g else 8192
        nxt, left = backlog_next(card, total); j = jobs.get(tag); alive = STATE["workers"].get(tag)
        if j and "cmd" in j:
            run, age = Text(short(j["cmd"]), style="bold cyan"), ago(since_hms(j["t"]))
        elif j:
            run, age = Text(j["state"], style="yellow"), ago(since_hms(j["t"]))
        else:
            run, age = Text("-", style="grey50"), ""
        flags = []
        for f_, lab in (("stop", "STOP"), ("halt", "HALT")):
            if os.path.exists(os.path.join(ROOT, "scratch", "%s.%s" % (tag, f_))):
                flags.append(lab)
        if alive is False:
            flags.append("NO WORKER")
        wt = Text("%s %s" % (tag, cuda)) + (Text(" " + " ".join(flags), style="bold red") if flags else Text(""))
        tj.add_row(wt, run, age, "\n".join("%s  %sG" % (s, m) for m, s in nxt) or "-", str(left))
    # CLOUD
    with LOCK:
        inst = list(STATE["vast"]); sshd = dict(STATE["ssh"]); vt, st, credit, err = STATE["vast_t"], STATE["ssh_t"], STATE["credit"], STATE["vast_err"]
        fl = list(STATE["fillers"]); dep = STATE.get("deploy_log", "")
    tc = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    for c, j in (("host", "left"), ("GPU", "left"), ("status", "left"), ("util", "left"), ("", "right"), ("VRAM", "right"), ("temp", "right"), ("$/h", "right"), ("up", "right"),
                 ("current run", "left"), ("step", "right"), ("loss", "right"), ("L", "right"), ("queue", "right")):
        tc.add_column(c, justify=j, no_wrap=True)
    burn = 0.0
    for o in sorted(inst, key=lambda o: o.get("label") or ""):
        burn += o.get("dph_total") or 0; s = sshd.get(o["id"], {}); lab = (o.get("label") or str(o["id"])).replace("gen-", ""); stt = o.get("actual_status") or "?"
        util = num(s["gpu"][0], o.get("gpu_util") or 0) if s.get("gpu") else (o.get("gpu_util") or 0)   # ssh nvidia-smi first, else the API
        if s.get("gpu"):
            vu, vtot = num(s["gpu"][1]) / 1024, num(s["gpu"][2], 1) / 1024
        else:
            vu, vtot = o.get("vmem_usage") or 0, (o.get("gpu_totalram") or 0) / 1024
        last = s.get("last") or {}; q = s.get("q"); ql = len(queue_lines(lab)); cur = os.path.basename(s.get("cur") or "") or ("(no ssh)" if not s else "-")
        qtxt = "%d/%d%s" % (min(q[1], ql or q[0]), ql or q[0], " DONE" if q[2] else "") if q else "-"   # restart slots evaluate several attempts per line
        sty = "green" if stt == "running" else "yellow" if stt in ("loading", "created") else "red"
        tc.add_row(lab, "%s %s" % (o.get("gpu_name", "").replace("RTX ", ""), (o.get("driver_version") or "")[:6]), Text(stt, style=sty), bar(util / 100, 10), "%3d%%" % util,
                   "%.1f/%.0fG" % (vu, vtot), "%d°C" % (o.get("gpu_temp") or 0), "%.3f" % (o.get("dph_total") or 0), ago(now - (o.get("start_date") or now)),
                   Text(cur, style="cyan"), "%s%s" % (last.get("step", ""), " eval" if q and q[1] < q[0] and last.get("step") == 24000 else ""), "%.4g" % last["loss"] if isinstance(last.get("loss"), (int, float)) else "", str(last.get("L_max", "")), qtxt)
    # CLOUD QUEUE
    tq = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False); tq.add_column("source", no_wrap=True); tq.add_column("lined up (not deployed yet)", ratio=1)
    live = {(o.get("label") or "").replace("gen-", "") for o in inst}; held = set()
    for name, pend in fl:
        held.update("b" + n for n in pend)
        tq.add_row(name, ", ".join("b%s %s" % (n, short_q("b" + n)) for n in pend) or "(all deployed)")
    staged = []
    try:                                                                    # every vdeploy create is logged in pods.txt as "label gen-<tag>"
        pods = open(os.path.join(ROOT, "scratch", "pods.txt"), encoding="utf-8", errors="replace").read()
    except OSError:
        pods = ""
    for f in glob.glob(os.path.join(ROOT, "scratch", "gen_q_*.txt")):
        tag = os.path.basename(f)[6:-4]
        if tag in live or tag in held or time.time() - os.path.getmtime(f) > 48 * 3600:
            continue
        if re.search(r"gen host %s offer \S+: .*STARTED" % re.escape(tag), dep) or re.search(r"label gen-%s," % re.escape(tag), pods):
            continue
        outs = [l.split()[0] for l in queue_lines(tag)]                    # older deploys predate these logs: lined up = some run has no result yet
        if not outs or all(glob.glob(os.path.join(ROOT, o.rstrip("/") + "*", "eval.json")) for o in outs):
            continue
        staged.append((tag, time.time() - os.path.getmtime(f)))
    for lab, sel, sty in (("staged", lambda a_: a_ <= 24 * 3600, ""), ("stale >24h", lambda a_: a_ > 24 * 3600, "grey50")):   # never deployed, maybe superseded
        ts = sorted((t for t, a_ in staged if sel(a_)), key=lambda t: (len(t), t))
        if ts:
            tq.add_row(Text(lab, style=sty), Text(", ".join("%s %s" % (t, short_q(t)) for t in ts), style=sty))
    hdr = Text.assemble(("LATENT TOP ", "bold white on blue"), ("  %s   " % time.strftime("%Y-%m-%d %H:%M:%S"), "bold"),
                        ("cloud %d hosts  $%.2f/h  credit $%s   " % (len(inst), burn, "%.2f" % credit if isinstance(credit, (int, float)) else "?"), "bold yellow"),
                        ("vast %s ago, ssh %s ago" % (ago(now - vt) if vt else "-", ago(now - st) if st else "-"), "grey50"), ("   " + err if err else "", "red"))
    return Group(hdr, Panel(tg, title="LOCAL GPUs", border_style="blue"), Panel(tj, title="LOCAL JOBS (v2 backlog workers)", border_style="blue"),
                 Panel(tc, title="CLOUD (Vast)", border_style="magenta"), Panel(tq, title="CLOUD QUEUE", border_style="magenta"))


def short_q(tag):
    ql = queue_lines(tag)
    if not ql:
        return "(empty)"
    names = [os.path.basename(l.split()[0]) for l in ql]
    return "%s%s" % (names[0], " +%d" % (len(names) - 1) if len(names) > 1 else "")


def worker_loop():
    ps = "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -match 'c1_backlog_worker' } | % { $_.CommandLine }"
    while True:
        out = sh(["powershell", "-NoProfile", "-Command", ps], 20)
        with LOCK:
            STATE["workers"] = {t: t in out.split() for t, _, _ in WORKERS} if out.strip() else {}   # empty output: unknown (no flag)
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--no-ssh", action="store_true"); a = ap.parse_args()
    threading.Thread(target=cloud_loop, args=(not a.no_ssh,), daemon=True).start(); threading.Thread(target=worker_loop, daemon=True).start()
    con = Console()
    if a.once:                                                              # one frame once the first cloud pass is in (or 60 s)
        t0 = time.time()
        while time.time() - t0 < 90 and (not STATE["fill_t"] or (not a.no_ssh and not STATE["ssh_t"])):
            time.sleep(1)
        con.print(render(*local_gpus())); return
    with Live(render(*local_gpus()), console=con, screen=True, refresh_per_second=1) as lv:
        try:
            while True:
                time.sleep(2); lv.update(render(*local_gpus()))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
