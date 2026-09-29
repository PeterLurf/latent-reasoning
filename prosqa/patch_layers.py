"""PROTOCOL6 addendum N: optionally truncate GPT-2 to its first n_layer pretrained blocks (config key `n_layer`).
    python patch_layers.py /workspace/coconut      (idempotent; without the key run.py behaves as before)"""
import os
import sys

p = os.path.join(sys.argv[1], "run.py")
s = open(p).read()
MARK = "# PROTOCOL6-N layers"
if MARK not in s:
    old = "    model = AutoModelForCausalLM.from_pretrained(configs.model_id)\n"
    assert old in s
    s = s.replace(old, f"""    {MARK}: first n_layer pretrained blocks only (unused blocks' weights are dropped)
    _nl = getattr(configs, "n_layer", None)
    model = AutoModelForCausalLM.from_pretrained(configs.model_id, **({{"n_layer": int(_nl)}} if _nl else {{}}))
    print("N_LAYER", model.config.n_layer, flush=True)
""", 1)
    open(p, "w").write(s)
    print("N: run.py patched")
else:
    print("N: run.py already patched")
