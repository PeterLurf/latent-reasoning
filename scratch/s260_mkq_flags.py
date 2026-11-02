"""§258 B bridge_s1's training flags (= scratch/s260_mkq.py's flags(), which builds and writes the b396 queue at import): importable for the S0 same-class anchor.
s258m_mkq asserts the anchor's --args_only == scratch/s260_args_frozen_bridge_s1.json except stop_at / eval_parts / out / device."""
import gate_s251 as G1

INIT = ("boxes/runs_fsq/aw_plaw2_s1_init/model.pt", "484dab1eab9e2c12")


def flags():
    fl = list(G1.RCB); i = fl.index("--init_model"); fl[i + 1] = INIT[0]; j = fl.index("--init_model_sha"); fl[j + 1] = INIT[1]
    return fl + ["--distract_p", "1", "--micro", "2", "--eval_bs", "250"]
