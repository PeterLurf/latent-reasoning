#!/bin/bash
# start the §223 P2 pilot on the 4060 after the queued K9b -> T2 textself GPU parts (scratch/s224_k9_t2_4060.sh) finish (Note 223-8 amendment 2 eebdc988 in).
cd /d/Latent
until grep -q "s219_post_t2 textself_gpu rc" logs/s219_post_landing.out; do sleep 120; done; sleep 30
bash scratch/s223_p2_pilot.sh >> logs/s222_chain.out 2>&1
