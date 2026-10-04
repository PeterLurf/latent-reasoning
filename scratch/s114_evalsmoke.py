"""§114 early proof (Audit): eval-path smoke at r = 8 with the stop-eval batch size; peak memory. Weights do not change
memory, so a fresh Loop114 is used. Run on the proof host beside the training job.
    python scratch/s114_evalsmoke.py [cuda:0]"""
import os
import sys
import time

os.environ["BOXES_NB"] = "5"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from boxes import s5  # noqa: E402
from boxes.accel99 import rollout_safe  # noqa: E402
from boxes.loop114 import Loop114, donor  # noqa: E402
from boxes.mult98 import MeterR, gen_mean_inputs, load  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; amp = torch.bfloat16; t0 = time.time()
Dte = load("bench/data/mult_4x4_test.txt", limit=200); Dtr = load("bench/data/mult_train/4x4_train.txt", limit=2000)
m = Loop114(s5.load_gpt2(dev)).to(dev).eval(); m.r = 8; torch.cuda.reset_peak_memory_stats(dev)
per, ok, _ = rollout_safe(m, Dte, dev, amp, 100, meter=MeterR(m.g)); p1 = torch.cuda.max_memory_allocated(dev) / 2**30
mu = gen_mean_inputs(m, Dtr, dev, amp); _, o_, _ = rollout_safe(m, Dte, dev, amp, 100, ablate="mean", mean_in=mu)
d = donor(m, dev, amp, n=200, bs=100); p2 = torch.cuda.max_memory_allocated(dev) / 2**30
m.r = 16; rollout_safe(m, Dte, dev, amp, 100); p3 = torch.cuda.max_memory_allocated(dev) / 2**30
print(f"EVALSMOKE ok: rollout r=8 peak {p1:.2f} GiB, +removal+donor {p2:.2f} GiB, r=16 rollout {p3:.2f} GiB, {time.time() - t0:.0f}s", flush=True)
