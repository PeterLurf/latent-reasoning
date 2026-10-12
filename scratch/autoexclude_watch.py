"""Auto-exclude failed Vast hosts (Audit 2026-10-10): follows logs/gen_deploy.log and appends the IP of every BAD HOST / CREATE FAILED deploy to
scratch/vast_exclude_ips.txt as "<ip> # auto <epoch> <reason>", so fill_generic (which re-reads the file at each offer search) skips it. Auto lines
expire after 24 h; hand-written lines are never touched. Heartbeat: logs/autoexclude.hb (epoch).
2026-10-12 (Audit; offer 32750800 was BAD HOST for b265 at 77.28.255.194 and again for b274 at 77.28.206.92): each failure also excludes
"oid:<offer id>" and, when the offer lookup returns it, "mid:<machine_id>" (host_id noted in the reason); fill_generic skips offers matching either.
    nohup python scratch/autoexclude_watch.py > logs/autoexclude.out 2>&1 &"""
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
LOG, EX, HB, TTL = "logs/gen_deploy.log", "scratch/vast_exclude_ips.txt", "logs/autoexclude.hb", 86400
V = "C:/Users/peter/.local/vastai/vast.sh"
BAD = re.compile(r"gen host (b\d+) offer (\d+): INSTANCE \d+ (\d+\.\d+\.\d+\.\d+) \S+ BAD HOST")
CF = re.compile(r"gen host (b\d+) offer (\d+): CREATE FAILED")
AUTO = re.compile(r"#\s*auto\s+(\d+)")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def offer_info(o):
    """-> (ip, machine_id, host_id) of offer o, or Nones when it is no longer listed."""
    try:
        r = subprocess.run(["bash", V, "search", "offers", "id=%s" % o, "--raw"], capture_output=True, text=True, timeout=60)
        rows = json.loads(r.stdout or "[]")
        return (rows[0].get("public_ipaddr"), rows[0].get("machine_id"), rows[0].get("host_id")) if rows else (None, None, None)
    except Exception as e:  # noqa: BLE001
        say("offer lookup failed", o, e); return (None, None, None)


def exclude_all(ip, offer, reason):
    _, mid, hid = offer_info(offer)
    for key in ([ip] if ip else []) + ["oid:%s" % offer] + (["mid:%s" % mid] if mid else []):
        rewrite((key, reason + (" host %s" % hid if hid else "")))


def rewrite(add=None):
    lines = open(EX, encoding="utf-8").read().splitlines(); now = time.time(); keep = []
    for l_ in lines:
        m = AUTO.search(l_)
        if m and now - int(m.group(1)) > TTL:
            say("expire", l_); continue
        keep.append(l_)
    if add and add[0] not in {l_.split()[0] for l_ in keep if l_.strip() and not l_.startswith("#")}:
        keep.append("%s # auto %d %s" % (add[0], int(now), add[1])); say("exclude", add[0], add[1])
    if keep != lines:
        tmp = EX + ".tmp"; open(tmp, "w", encoding="utf-8", newline="\n").write("\n".join(keep) + "\n"); os.replace(tmp, EX)


def main():
    pos = os.path.getsize(LOG); say("start at byte", pos); last_exp = 0
    while True:
        if os.path.getsize(LOG) > pos:
            with open(LOG, encoding="utf-8", errors="replace") as f:
                f.seek(pos); new = f.read(); pos = f.tell()
            for l_ in new.splitlines():
                m = BAD.search(l_)
                if m:
                    exclude_all(m.group(3), m.group(2), "BAD_HOST %s offer %s" % (m.group(1), m.group(2))); continue
                m = CF.search(l_)
                if m:
                    exclude_all(offer_info(m.group(2))[0], m.group(2), "CREATE_FAILED %s offer %s" % (m.group(1), m.group(2)))
        if time.time() - last_exp > 600:
            rewrite(); last_exp = time.time()
        open(HB, "w").write("%d\n" % time.time()); time.sleep(30)


if __name__ == "__main__":
    main()
