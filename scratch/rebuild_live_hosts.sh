#!/usr/bin/env bash
# Rebuild scratch/live_hosts.txt from `vast show instances` (Audit: stale labels after destroys/redeploys).
# Line format as the deploy scripts write it: "<ip> <port> <label>"; one line per live labelled instance; LF endings
# (Windows python prints CRLF: stripped with tr). Run after every destroy / redeploy.
cd "$(dirname "$0")/.."
V=C:/Users/peter/.local/vastai/vast.sh; export PYTHONIOENCODING=utf-8
bash $V show instances --raw 2>/dev/null | python -c "
import json,sys
rows=[]
for i in json.load(sys.stdin):
    lab=i.get('label'); ip=i.get('public_ipaddr'); port=(i.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort') or i.get('ssh_port')
    if lab and ip and port: rows.append(f'{ip} {port} {lab}')
print(chr(10).join(sorted(rows, key=lambda r: r.split()[-1])))" | tr -d '\r' > scratch/live_hosts.new && [ -s scratch/live_hosts.new ] && mv scratch/live_hosts.new scratch/live_hosts.txt
wc -l < scratch/live_hosts.txt
