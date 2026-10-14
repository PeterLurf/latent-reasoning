#!/usr/bin/env bash
# §199-A A4 launch check (Audit, BLOCKING): once the cell's host log has step >= 500, stream its log.jsonl down read-only (tar to stdout) and compare
# every record at steps 100..500 (loss, answer_ce, aux_ce) with the control s185_rms_s<seed>'s local log.jsonl, bitwise (float equality of the JSON
# values). Writes boxes/runs_fsq/<cell>/a4_check.json and prints PAIRED / UNPAIRED.
#   bash scratch/a4_check199.sh <host> <port> <seed>
set -u
cd "$(dirname "$0")/.."
H=$1; P=$2; S=$3; C=s199_csr_s$S; SSH="ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p $P root@$H"
for i in $(seq 1 120); do
  st=$(timeout 40 $SSH "tail -n 1 /workspace/Latent/boxes/runs_fsq/$C/log.jsonl | cut -d, -f1 | tr -dc 0-9" 2>/dev/null)
  [ -n "$st" ] && [ "$st" -ge 500 ] && break; sleep 60
done
[ -n "${st:-}" ] && [ "$st" -ge 500 ] || { echo "$(date +%T) $C: step 500 not reached"; exit 1; }
T=$(mktemp -d); timeout 300 $SSH "cd /workspace/Latent/boxes/runs_fsq/$C && tar -cf - log.jsonl" 2>/dev/null | tar -xf - -C "$T" || { echo "fetch failed"; rm -rf "$T"; exit 1; }
python - "$T/log.jsonl" boxes/runs_fsq/s185_rms_s$S/log.jsonl boxes/runs_fsq/$C "$H" <<'EOF'
import json, os, sys
cell, ctl, out, host = sys.argv[1:5]
rd = lambda f: {r["step"]: r for r in (json.loads(l) for l in open(f) if l.strip()) if "loss" in r and 100 <= r["step"] <= 500}
A, B = rd(cell), rd(ctl); rows = []
for s in sorted(B):
    a, b = A.get(s, {}), B[s]
    rows.append({"step": s, **{k: {"cell": a.get(k), "control": b.get(k), "equal": a.get(k) == b.get(k)} for k in ("loss", "answer_ce", "aux_ce")}})
ok = bool(rows) and sorted(A) == sorted(B) and all(r[k]["equal"] for r in rows for k in ("loss", "answer_ce", "aux_ce"))
# CSR is active AT its start step (csr_on: step >= 500), and the record's loss is the mean of steps 401..500's totals, so the step-500 loss carries
# + 0.1 * csr_500 / 100 by construction; answer_ce / aux_ce exclude the CSR term. Reported beside the literal verdict for Theory / Audit to read.
ok_x = bool(rows) and sorted(A) == sorted(B) and all(r[k]["equal"] for r in rows for k in ("loss", "answer_ce", "aux_ce") if not (r["step"] == 500 and k == "loss"))
a5 = A.get(500, {}); s500 = {"loss_diff": (a5.get("loss") - B[500]["loss"]) if (a5.get("loss") is not None and 500 in B) else None,
                              "predicted_0.1xcsr_over_100": 0.1 * a5["csr_loss"] / 100 if a5.get("csr_loss") is not None else None, "csr_loss_500": a5.get("csr_loss")}
o = {"verdict": "PAIRED" if ok else "UNPAIRED", "verdict_excl_step500_loss": "PAIRED" if ok_x else "UNPAIRED", "step500_decomposition": s500,
     "steps": sorted(B), "cell_steps": sorted(A), "rows": rows, "host": host,
     "def": "§199-A A4: every record at steps 100-500 (loss, answer_ce, aux_ce) == the control's log.jsonl, float equality"}
os.makedirs(out, exist_ok=True); json.dump(o, open(os.path.join(out, "a4_check.json"), "w", newline="\n"), indent=1)
print("A4", os.path.basename(out), o["verdict"], "excl500loss", o["verdict_excl_step500_loss"], s500, [(r["step"], r["loss"]["cell"], r["loss"]["control"]) for r in rows if not all(r[k]["equal"] for k in ("loss", "answer_ce", "aux_ce"))][:3])
EOF
rm -rf "$T"
