#!/usr/bin/env bash
# Chunked, sha-verified fetch of one file from a Vast host whose link resets long transfers (b102, 2026-10-07: ~53 KB/s, resets every
# ~70 s, so a 463 MB scp never completes). The host splits the file into 8 MB pieces with per-piece sha256; each piece is fetched with
# up to 20 tries until its sha matches; the pieces are joined and the whole file checked against the host's sha256.
#   bash scratch/chunk_fetch.sh <host> <port> <remote file> <local file>      -> prints CHUNK_FETCH_OK <sha> or CHUNK_FETCH_FAILED
H=$1; P=$2; RF=$3; LF=$4; O="-o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=6 -i $HOME/.ssh/runpod_ed25519"
S="ssh $O -p $P root@$H"; W=$(mktemp -d /d/drain_tmp/chunks_XXXX)
for t in $(seq 1 10); do
  man=$(timeout 300 $S "cd /tmp && rm -rf cf_parts && mkdir cf_parts && split -b 8M -d -a 4 $RF cf_parts/p_ && sha256sum $RF | cut -c1-64 && (cd cf_parts && sha256sum p_*)" 2>/dev/null)
  [ -n "$man" ] && break; sleep 20
done
[ -n "$man" ] || { echo "CHUNK_FETCH_FAILED (no manifest)"; exit 1; }
whole=$(echo "$man" | head -n 1); echo "$man" | tail -n +2 > $W/manifest; n=$(wc -l < $W/manifest); echo "$(date +%T) $n pieces, whole sha $whole"
while read -r sh name; do
  ok=0
  for t in $(seq 1 20); do
    timeout 600 scp -q $O -P $P root@$H:/tmp/cf_parts/$name $W/$name 2>/dev/null
    [ "$(sha256sum $W/$name 2>/dev/null | cut -c1-64)" = "$sh" ] && { ok=1; break; }; sleep 5
  done
  [ $ok = 1 ] || { echo "CHUNK_FETCH_FAILED ($name after 20 tries)"; exit 1; }
done < $W/manifest
cat $(awk '{print "'$W'/"$2}' $W/manifest) > $LF
[ "$(sha256sum $LF | cut -c1-64)" = "$whole" ] || { echo "CHUNK_FETCH_FAILED (whole sha mismatch)"; exit 1; }
rm -rf $W; timeout 120 $S "rm -rf /tmp/cf_parts" 2>/dev/null; echo "$(date +%T) CHUNK_FETCH_OK $whole"
