"""§211 descriptive watch (Theory 211-9): read-only ssh polls of the cells' log.jsonl every 2 min. Emits one line per event:
b304 rows at steps 10000 / 12000 (answer_ce, rec_ce, c, off-self, gather_gn, gn_preclip), and every cur_stage advance in b304 / b305 / b306 with
the answer_ce records from 200 before to 1000 after it (emitted once the 1000-after record exists). Exits after b304's 12k row and N1's first advance."""
import json, os, re, subprocess, sys, time
os.chdir("D:/Latent"); KEY = os.path.expanduser("~/.ssh/runpod_ed25519")
CELLS = {304: "s211_vb_pa_s1", 305: "s211_vb_carry_s1", 306: "s211_vb_pamct_s1"}; done = set()
def host(n):
    h = None
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m = re.search(r"gen host b%d offer \d+: INSTANCE \d+ (\S+) (\d+) .*STARTED" % n, l)
        if m: h = (m.group(1), m.group(2))
    return h
def recs(n):
    hp = host(n)
    if not hp: return []
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", hp[1], "root@" + hp[0], "cat /workspace/Latent/boxes/runs_fsq/%s/log.jsonl" % CELLS[n]], capture_output=True, text=True, timeout=90).stdout
        return [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]
    except Exception:
        return []
fmt = lambda r: "%d ans %.3f rec %.3f c %.2f off %s ggn %s gn %.1f" % (r["step"], r["answer_ce"], r["rec_ce"], r["c"], ("%.2f" % r["gather_offself"]) if "gather_offself" in r else "-", ("%.2f" % r["gather_gn"]) if "gather_gn" in r else "-", r["gn_preclip"])
while True:
    for n in CELLS:
        R = recs(n); main = {r["step"]: r for r in R if "loss" in r and r.get("answer_ce") is not None}
        if n == 304:
            for s in (10000, 12000):
                if ("row", n, s) not in done and s in main:
                    done.add(("row", n, s)); print("b304 @%d: %s" % (s, " | ".join(fmt(main[k]) for k in sorted(main) if s - 1000 <= k <= s and k % 200 == 0)), flush=True)
        for a in (r for r in R if "reason" in r):
            key = ("adv", n, a["step"]); s0 = a["step"]
            if key not in done and max(main or [0]) >= s0 + 1000:
                done.add(key); print("b%d ADVANCE @%d -> n_max %d (%s): %s" % (n, s0, a["n_max_train"], a["reason"], " | ".join("%d %.3f" % (k, main[k]["answer_ce"]) for k in sorted(main) if s0 - 200 <= k <= s0 + 1000 and k % 100 == 0)), flush=True)
    if ("row", 304, 12000) in done and any(k[0] == "adv" and k[1] == 305 for k in done):
        print("WATCH DONE", flush=True); sys.exit(0)
    time.sleep(120)
