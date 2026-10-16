"""§211 note 211-10 bins (Theory 01:2x 10-16): fixed 1k-bin means (bin b = steps 1000b+1 .. 1000(b+1); records deduped by step, last occurrence
wins = basin_entry's binning) of answer_ce for N1 (b305) and N2 (b304), bins 5..9 (5k-10k; 5 = context), plus N2's gather_offself / gather_gn.
Read-only ssh; waits until both cells log step >= 10000. -> scratch/gate_s211/bins_211_10.json"""
import json, os, re, subprocess, time
os.chdir("D:/Latent"); KEY = os.path.expanduser("~/.ssh/runpod_ed25519"); CELLS = {304: "s211_vb_pa_s1", 305: "s211_vb_carry_s1"}
def host(n):
    h = None
    for l in open("logs/gen_deploy.log", encoding="utf-8", errors="replace"):
        m = re.search(r"gen host b%d offer \d+: INSTANCE \d+ (\S+) (\d+) .*STARTED" % n, l)
        if m: h = (m.group(1), m.group(2))
    return h
def recs(n):
    h, p = host(n)
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-i", KEY, "-p", p, "root@" + h, "cat /workspace/Latent/boxes/runs_fsq/%s/log.jsonl" % CELLS[n]], capture_output=True, text=True, timeout=120).stdout
    d = {}
    for l in out.splitlines():
        if l.strip().startswith("{"):
            r = json.loads(l)
            if "step" in r and "loss" in r:
                d[int(r["step"])] = r                                   # last occurrence wins
    return d
while True:
    R = {n: recs(n) for n in CELLS}
    if all(max(R[n] or [0]) >= 10000 for n in CELLS):
        break
    time.sleep(120)
def bins(d, key):
    b = {}
    for s, r in d.items():
        if r.get(key) is not None:
            b.setdefault((s - 1) // 1000, []).append(float(r[key]))
    return {"%dk-%dk" % (k, k + 1): round(sum(v) / len(v), 4) for k, v in sorted(b.items()) if 5 <= k <= 9}
out = {"def": "bin b = steps 1000b+1..1000(b+1), dedup by step (last wins), mean of logged records", "N1_b305_answer_ce": bins(R[305], "answer_ce"),
       "N2_b304_answer_ce": bins(R[304], "answer_ce"), "N2_b304_gather_offself": bins(R[304], "gather_offself"), "N2_b304_gather_gn": bins(R[304], "gather_gn"),
       "n_records_per_bin": {n: {k: len([s for s in R[n] if (s - 1) // 1000 == int(k.split("k")[0])]) for k in bins(R[n], "answer_ce")} for n in CELLS}}
json.dump(out, open("scratch/gate_s211/bins_211_10.json", "w", newline="\n"), indent=1); print(json.dumps(out))
