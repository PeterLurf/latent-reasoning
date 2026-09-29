"""PROTOCOL7 T11: train a GPT-2-architecture model from scratch (random init) with config keys `from_scratch: True`,
`n_layer`, optional `n_embd`/`n_head`. Idempotent; without `from_scratch` run.py behaves as before.
    python patch_scratch.py /workspace/coconut"""
import os
import sys

p = os.path.join(sys.argv[1], "run.py")
s = open(p).read()
MARK = "# PROTOCOL7-T11 scratch"
if MARK not in s:
    old = "    model = AutoModelForCausalLM.from_pretrained(configs.model_id)\n"
    assert old in s, "run.py model line not found (patch_layers applied?)"
    s = s.replace(old, f"""    {MARK}: random init of the GPT-2 architecture when configs.from_scratch
    if getattr(configs, "from_scratch", False):
        from transformers import AutoConfig
        _kw = {{k: int(getattr(configs, k)) for k in ("n_layer", "n_embd", "n_head") if getattr(configs, k, None)}}
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(configs.model_id, **_kw))
        print("FROM_SCRATCH", model.config.n_layer, model.config.n_embd, model.config.n_head, flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(configs.model_id)
""", 1)
    open(p, "w").write(s)
    print("T11: run.py patched")
else:
    print("T11: run.py already patched")
