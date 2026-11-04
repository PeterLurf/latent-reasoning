# Host-side idle stop (Audit MED, 11-04 after the 2nd reboot in 11 h)

Problem: the cost guard, wall guard and drain all run on this box. A box outage leaves finished hosts idling
(b415/b416 ~$1.0 on 11-03, b417 ~$1.7 on 11-04).

## Mechanism
`scratch/host_idle_stop.sh`, started by `gen_vast_job.sh` right after the queue line loop starts (setsid nohup,
so it outlives the queue runner). It polls every 5 min and uses only host state:

1. **Idle after QUEUE_DONE:** `logs/queue_<tag>.out` ends in QUEUE_DONE, AND no `logs/FETCH_ACTIVE` touched in the
   last 45 min, AND QUEUE_DONE is older than `IDLE_H` (default 1.5 h, from the queue header `# IDLE_H`).
2. **Hard wall:** host uptime of the job > `HARD_WALL_H` (queue header; = the cell's guard cap / dph x 1.15),
   whatever the queue state. This is a backstop for the local cost guard, not a replacement.

On either condition it writes `logs/IDLE_STOP <reason>` and calls the Vast API to STOP (not destroy) its own
instance: `PUT /api/v0/instances/$CONTAINER_ID/ {"state":"stopped"}` with the per-instance `CONTAINER_API_KEY`.
Both are read from `/proc/1/environ` (the docker env; ssh shells don't inherit it). The key is never echoed or logged.
Stopping ends GPU billing. Disk persists at ~$0.005-0.02/h, so no result is lost.

## Drain side (gen9_s224 -> gen10 copy, default-preserving)
- `drain_gen_partial_s224.sh` touches `logs/FETCH_ACTIVE` before each tar and every scp retry.
- The watcher reads `actual_status` from `show instance`. If it is `exited` / `stopped` with `logs/IDLE_STOP`
  unknown, it logs `HOST_SELF_STOPPED`, runs `vast start instance <id>`, waits up to 30 min for ssh, and drains as
  usual. If the start fails because the GPU was re-rented, it logs `START_BLOCKED` and retries every 30 min. The disk
  is intact either way. A destroy happens only after a verified fetch, as now.
- Post-reboot recovery adds one step: `show instances` lists stopped hosts too, so the re-arm covers them.

## Gates (before any family uses it)
- G0 (~$0.03): rent the cheapest verified offer, run host_idle_stop.sh with IDLE_H=0.02, a fake QUEUE_DONE and no
  FETCH_ACTIVE. It must reach `stopped` within 10 min. Then `start`, check the files survive, and destroy.
  Also check that CONTAINER_API_KEY exists in /proc/1/environ and that the key can only act on its own instance
  (a PUT on another id must be refused).
- G1: FETCH_ACTIVE suppression. Touch the marker, confirm no stop over 2 polls.
- G2: gen10 drain on a self-stopped G0 host: start, drain, destroy, DESTROYED_VERIFIED.
- `bash -n` + `set -u` run of both scripts (deploy-script-edit rule).

## If CONTAINER_API_KEY is absent or can't stop
Fallback: a cloud scheduled task (claude.ai routine, hourly) that runs `show instances` with the account key and
stops any instance whose queue log says QUEUE_DONE > 1.5 h. That needs the account key in a cloud environment,
which is a user decision, so it goes to the user.

Cost: G0-G2 ~$0.05. Build ~1 h. Applies to every queue launched after G-D (s272 b423/b424, §265 pilot, §271 S1).
