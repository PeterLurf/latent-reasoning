#!/usr/bin/env bash
# One-off: when s71-dpvar's first queue ends, rerun its OOM'd seed(s) one at a time with the fixed job (finished runs are
# skipped), void the failed attempt, and re-arm the watcher.
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=15 -i $HOME/.ssh/runpod_ed25519 -p 26168 root@38.64.63.175"
until timeout 30 $S "grep -q QUEUE_DONE /workspace/Latent/logs/queue_s71_dpvar.out" 2>/dev/null; do sleep 60; done
timeout 60 $S "cd /workspace/Latent; mv scratch/s71_dp_job.sh.new scratch/s71_dp_job.sh; for r in 1 2 3; do [ -s boxes/runs_s5/dpvar_s\$r/preds_test.npy ] || { mv boxes/runs_s5/dpvar_s\$r boxes/runs_s5/dpvar_s\$r.void_oom 2>/dev/null; mv logs/s71_dpvar_s\$r.log logs/s71_dpvar_s\$r.log.void_oom 2>/dev/null; }; done; mv logs/queue_s71_dpvar.out logs/queue_s71_dpvar.out.void_first; (setsid nohup bash scratch/s71_dp_job.sh s71_dpvar cuda:0 scratch/s71_q_dpvar.txt > /dev/null 2>&1 < /dev/null &)"
echo "$(date +%T) dpvar rerun started" >> logs/s71_deploy.log
pkill -f "auto_drain_s71.sh \"38.64.63.175" 2>/dev/null
nohup bash scratch/auto_drain_s71.sh "38.64.63.175 26168 s71_dpvar 53805951 s71" > logs/auto_drain_s71_dpvar.out 2>&1 &
