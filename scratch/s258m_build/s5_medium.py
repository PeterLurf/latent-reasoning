MEDIUM = {"file": "prosqa/ckpt/gpt2-medium.safetensors", "url": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors",
          "sha256": "fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd", "H": 16, "L": 24, "D": 1024}   # §258 S0 (LB2_v1): the HF LFS sha256


def medium_sd():
    """§258 S0: HF openai-community/gpt2-medium model.safetensors (fetched once to prosqa/ckpt/ if absent; its sha256 asserted on EVERY load) -> a GPT2KV state dict:
    'transformer.' stripped if present, lm_head.weight dropped; the F32 tensors parsed from the safetensors header (no extra dependency)."""
    import hashlib
    import json as _json
    import struct
    import urllib.request
    p = os.path.join(ROOT, MEDIUM["file"])
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".part"; urllib.request.urlretrieve(MEDIUM["url"], tmp); os.replace(tmp, p)
    b = open(p, "rb").read(); sha = hashlib.sha256(b).hexdigest(); assert sha == MEDIUM["sha256"], ("gpt2-medium safetensors sha256", sha)
    n = struct.unpack("<Q", b[:8])[0]; hdr = _json.loads(b[8:8 + n]); off = 8 + n; sd = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", (k, v["dtype"]); s, e = v["data_offsets"]
        t = torch.frombuffer(bytearray(b[off + s:off + e]), dtype=torch.float32).reshape(v["shape"])
        k = k[len("transformer."):] if k.startswith("transformer.") else k
        if k != "lm_head.weight":
            sd[k] = t
    return sd


def load_gpt2(dev, base="gpt2"):
    try:                                                                    # local hard memory cap (LATENT_MEM_CAP_GB; no-op on the cloud):
        from boxes import memcap                                            # every model loads GPT-2 here, so every local job is capped
        memcap.apply(dev)
    except ImportError:
        pass
    if base == "gpt2":                                                      # GPT-2 small: unchanged
        sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
        return bc.GPT2KV(sd).to(dev)
    assert base == "gpt2-medium", ("unknown --base", base)                  # §258 S0: H from the config; L / D from the weights, asserted
    g = bc.GPT2KV(medium_sd(), MEDIUM["H"]); assert (g.L, g.D) == (MEDIUM["L"], MEDIUM["D"]), ("gpt2-medium shape", g.L, g.D); return g.to(dev)
