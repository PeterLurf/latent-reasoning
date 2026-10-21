#!/usr/bin/env bash
# unit test (Audit G-T13 condition 2, 10-20): s219_resume_ckpt picks ckpt_last.pt, else ckpt_prev.pt, else nothing; and the partial-run drain's
# host command (built exactly as drain_gen_partial_s219.sh line 78 builds it: declare -f + \$(s219_resume_ckpt $j)) tars the right checkpoint.
#   bash scratch/gate_s219/test_resume_pick.sh
set -u
cd /d/Latent; . scratch/s219_done_marker.sh; F=$(pwd)/scratch/_tmp/s219_resume_pick; rm -rf "$F"; mkdir -p "$F/workspace/Latent"; fails=0
ok() { if [ "$2" = "$3" ]; then echo "ok   $1: $2"; else echo "FAIL $1: got '$2' want '$3'"; fails=$((fails + 1)); fi; }
cd "$F/workspace/Latent"; R=boxes/runs_fsq
mkdir -p $R/both $R/prevonly $R/none $R/emptylast
echo a > $R/both/ckpt_last.pt; echo b > $R/both/ckpt_prev.pt; echo b > $R/prevonly/ckpt_prev.pt; : > $R/emptylast/ckpt_last.pt; echo b > $R/emptylast/ckpt_prev.pt
for d in both prevonly none emptylast; do echo '{}' > $R/$d/args.json; echo '{}' > $R/$d/log.jsonl; done
ok pick_both "$(s219_resume_ckpt $R/both)" "$R/both/ckpt_last.pt"
ok pick_prevonly "$(s219_resume_ckpt $R/prevonly)" "$R/prevonly/ckpt_prev.pt"
ok pick_none "$(s219_resume_ckpt $R/none)" ""
ok pick_emptylast "$(s219_resume_ckpt $R/emptylast)" "$R/emptylast/ckpt_prev.pt"
for d in both prevonly none emptylast; do
  j=$R/$d; n=$(echo $j | tr '/' '_')_vp
  cmd="$(declare -f s219_resume_ckpt); cd /workspace/Latent && tar -czf /tmp/f_$n.tgz --ignore-failed-read $j/args*.json $j/log.jsonl* $j/draws_log.json $j/init_from.json \$(s219_resume_ckpt $j) 2>/dev/null; sha256sum /tmp/f_$n.tgz"
  cmd=${cmd//\/workspace/$F/workspace}; cmd=${cmd//\/tmp\//$F/}
  ( unset -f s219_resume_ckpt; bash -c "$cmd" > /dev/null )
  got=$(tar -tzf "$F/f_$n.tgz" | grep ckpt | tr '\n' ' ' | sed 's/ $//')
  case $d in both) want="$j/ckpt_last.pt";; prevonly|emptylast) want="$j/ckpt_prev.pt";; none) want="";; esac
  ok tar_$d "$got" "$want"
done
echo "FAILS $fails"
