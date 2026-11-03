#!/usr/bin/env bash
# v6 item 3 fixture (CPU): scratch/s271_gates.py with GATE_SCRIPT = a fake gate whose part 'boom' dies (MemoryError) and parts 'ok1' / 'ok2' pass; one subprocess per part:
# the wrapper must survive 'boom', still run 'ok2' after it, record part_rc, write NO gates_pass.json, exit 1 - and with only ok1 / ok2 write gates_pass.json, exit 0.
set -u
cd "$(dirname "$0")/../.." || exit 2
R=scratch/_tmp/gisofx; rm -rf "$R"; mkdir -p "$R/out1" "$R/out2" "$R/gd"
cat > "$R/fake_gate.py" <<'EOF'
import json, os, sys
part = sys.argv[1]; d = os.environ["GATE_OUTDIR"]
if part == "boom":
    x = [bytearray(10 ** 9) for _ in range(10 ** 6)]                       # dies here (MemoryError / killed)
json.dump({"pass": True}, open(os.path.join(d, part + ".json"), "w")); print(part, "pass True")
EOF
echo blob > "$R/b.txt"; sha=$(git hash-object "$R/b.txt")
GATE_SCRIPT="$R/fake_gate.py" GATE_OUTDIR="$R/gd" python -B -m scratch.s271_gates --blob "$R/b.txt=$sha" --gates ok1,boom,ok2 --device cpu --out "$R/out1" > "$R/w1.log" 2>&1; rc1=$?
rm -f "$R"/gd/*.json
GATE_SCRIPT="$R/fake_gate.py" GATE_OUTDIR="$R/gd" python -B -m scratch.s271_gates --blob "$R/b.txt=$sha" --gates ok1,ok2 --device cpu --out "$R/out2" > "$R/w2.log" 2>&1; rc2=$?
ok=1
[ $rc1 = 1 ] && [ ! -e "$R/out1/gates_pass.json" ] || { echo "FAIL run 1: rc $rc1 or gates_pass written"; ok=0; }
python -c "import json,sys; e=json.load(open('$R/out1/eval.json')); sys.exit(0 if e['part_rc']['boom'] != 0 and e['part_rc']['ok1'] == 0 and e['part_rc']['ok2'] == 0 and e['gates'].get('ok2') is True else 1)" || { echo "FAIL run 1: part_rc / ok2 after boom"; ok=0; }
grep -q "=== part ok2 rc 0" "$R/out1/gates.log" || { echo "FAIL run 1: ok2 did not run after boom"; ok=0; }
[ $rc2 = 0 ] && [ -e "$R/out2/gates_pass.json" ] || { echo "FAIL run 2: rc $rc2 or no gates_pass"; ok=0; }
[ $ok = 1 ] && echo "FIXTURE gates_isolation PASS" || { echo "FIXTURE gates_isolation FAIL"; tail -n 5 "$R/w1.log"; }
