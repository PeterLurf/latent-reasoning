"""§221 GSM-CACHE-SWAP-2, G-FID(a) for zen-gpt2: run the OFFICIAL zhenyi4/codi test.py at batch size 1 on all 1,319 GSM8K test items
-> <run dir>/fid_official_bs1.json (the "ref" block of gate_fid.json, kind "official_bs1"; contract = theory/score_s221.py's docstring).
scratch/s221_gfid.py --model zen later merges it with the pairs[0:50] re-run into gate_fid.json. Registered text: GCS2_v1.md at b7c66d8f (§2 G-FID(a)).
A NEW scratch file: it edits nothing in the official repo, under boxes/ or theory/ (smokes only import scratch/s220_gcs.py's synthetic()).

WHAT RUNS: test.py itself as __main__ (= runpy.run_path; compiled from its own path, with exactly ONE line changed in memory, D9), from its own
directory (cwd = repo root, repo root first on sys.path, as `cd codi && python test.py ...`), with test.py's own CLI (scripts/test_gpt2.sh at the
pinned commit) except:
  C1  --batch_size 1                       (test_gpt2.sh: 128). batch_size is a CLI field (src/model.py DataArguments), so no batch-size patch;
                                           at bs 1 the tokenizer's padding="longest" adds no pads (asserted per batch: 0 pad tokens, 1,319 batches).
  C2  --ckpt_dir prosqa/ckpt_ext/hf/codi   (= zen-E/CODI-gpt2 pytorch_model.bin, sha256 asserted == boxes/codi_g1.SHA == the pinned eval_gcs meta).
  C3  --output_dir <scratch/_tmp/...>      (TrainingArguments needs one; test.py never writes to it).
Decode settings are therefore test.py's own (printed and source-asserted by --settings_only, see SRC_PINS): greedy (--greedy True ->
torch.argmax over logits[:, -1, :vocab_size-1], i.e. <eot> never generated), max_new_tokens 256 (gen_kwargs), stop = per sequence at
tokenizer.eos_token_id (50256, the EOS id is appended before the stop), 6 latent iterations with prj (--inf_latent_iterations 6, --use_prj True),
remove_eos True (q + <bot>; <eot> only), tokenizer.decode(pred_token, skip_special_tokens=True), parser = test.py's extract_answer_number
(commas removed, last -?\\d+\\.?\\d* number, none -> inf), accuracy = test.py's compute_accuracy (p == g), inf_num_iterations 1. No position_ids
anywhere (HF default = arange from the cache length; at bs 1 with no pads that equals the port's explicit 0..T-1), attention_mask on the question pass
only (all ones at bs 1), sdpa attention (HF default; the port uses eager), LoRA unmerged (peft; the port merges into fp32).

DEVIATIONS (each listed, for Audit's G-D; none touches the decode loop, the stop rule, the parser or the argmax):
  D1  dtype (--dtype, REQUIRED, no default). test.py hard-codes model.to(torch.bfloat16) (line 100) and loads the base in bf16 (--bf16) or fp16.
        --dtype bf16 : no patch (test.py's own numerics).
        --dtype fp32 : two patches, recorded with counts: (P1) AutoModelForCausalLM.from_pretrained's torch_dtype/dtype -> float32 (the base is then
                       overwritten by the checkpoint's bf16 base weights, upcast exactly, and its fp32 LoRA / prj, as our port's merged fp32 path);
                       (P2) CODI.to(torch.bfloat16) is a no-op (CODI.to(...) with any other argument is untouched). §221's summary says "3070, fp32".
        CPU smoke on 30 s220-synthetic items (not GSM8K) vs the s220 port's fp32 INTACT: fp32 30/30 preds and 30/30 token shas equal; bf16 26/30.
  D2  per-item outputs: test.py keeps predictions in a local list and returns only the accuracy. Observation-only hooks record them: the tokenizer
      (class swapped to a subclass whose decode / __call__ call the original and record when called from test.py's evaluation()) -> the generated
      ids of each item (pred_token, incl. the stopping EOS) and the decoded string; the pred is test.py's own extract_answer_number on that string
      (taken from the run's own globals). inf (no number) is stored as None (contract num|None; the port's parser returns None there). Cross-checks:
      the count and values equal test.py's printed "Prediction=" lines, and the recomputed accuracy equals test.py's printed average accuracy.
  D3  data: test.py's own load_dataset("gsm8k", "main") (HF hub; needs network or the HF cache) is observed, not replaced: its test split must equal
      prosqa/ckpt_ext/hf/data/gsm8k_test.json (sha 3d497ddb..., the port's file) item for item (question string and gold number) or the run aborts.
      --local_test serves that json instead (a listed deviation; use only if the hub is unreachable).
  D4  CODI.load_state_dict is observed (missing / unexpected keys recorded; abort if any unexpected key or any missing PARAMETER: test.py's
      strict=False would hide a key mismatch).
  D5  process settings as the port (s220_gcs.py main): CUBLAS_WORKSPACE_CONFIG :4096:8, TF32 off (matmul + cudnn), use_deterministic_algorithms(True,
      warn_only=True), torch threads --threads. --device cuda:N sets CUDA_VISIBLE_DEVICES=N before torch is imported (test.py hard-codes "cuda"), and
      the visible card's name must equal the pinned eval_gcs meta card (NVIDIA GeForce RTX 3070).
  D6  test.py's stdout (its per-item Q / decoded text / Prediction prints, unchanged) goes to <out>.stdout.log, not the console; progress lines only.
  D7  sys.dont_write_bytecode (keeps the pinned checkout clean; asserted clean before and after).
  D8  API compat: src/model.py passes resume_download=True to from_pretrained; transformers 5 forwards it to GPT2LMHeadModel.__init__ (TypeError),
      so it is dropped (a download-resume flag; no effect on weights or numerics). Recorded in meta.compat.
  D3b API compat: this hub client rejects the bare legacy dataset id "gsm8k" (HfUriError "Repository id must be 'namespace/name'"); the hook
      resolves it to the canonical "openai/gsm8k" (rev 740312ad); the split is then asserted equal to the local json (D3; checked 10-21: 1,319 / 1,319).
  D9  bs-1 crash fix (the ONLY source change, in memory, the checkout untouched): test.py:261 `torch.argmax(logits, dim=-1).squeeze(-1)` is 0-dim at
      B = 1 and test.py:287 `next_token_ids[b].item()` then raises IndexError (any torch version: test.py cannot run at bs 1 as written).
      `.squeeze(-1)` -> `.reshape(-1)`: the same argmax values, shape (B,) for every B (for B >= 2 identical to squeeze). Executed source sha in meta.
Environment: third_party/venv_codi (python -m venv --system-site-packages; + peft 0.15.2, accelerate 1.7.0 = the repo's requirements.txt pins,
installed --no-deps), so torch / transformers / datasets are the port's (2.14.0+cu130 / 5.18.0 / 5.0.1; the repo pins 2.7.1 / 4.52.4 / 3.6.0, a
recorded difference). See scratch/s221_official_setup.sh (clone, venv, HF prefetch of gpt2 + openai/gsm8k, data check). The repo:
github.com/zhenyi4/codi at REPO_COMMIT, cloned into third_party/codi with core.autocrlf false (LF bytes = git blobs; shas pinned below).

SMOKES (CPU only; never GSM8K items): --smoke_cpu --synthetic N --out_dir scratch/_tmp/... maps test.py's hard-coded "cuda" to cpu (torch.tensor /
torch.zeros / Tensor.to / Module.to device args; smoke-only) and serves N template arithmetic questions (NOT GSM8K) through load_dataset; with
--hf_model openai-community/gpt2 (smoke only: the same HF snapshot 607a30d7 as "gpt2", whose weights are in the local cache) and HF_HUB_OFFLINE=1.
    third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --settings_only --dtype fp32 --device cpu --out_dir scratch/_tmp/s221/off_settings
    third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --check_data --dtype fp32 --device cpu --out_dir scratch/_tmp/s221/off_settings
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --device cuda:1 --dtype fp32
                                                                    # -> scratch/s220/zen_codi_gpt2/fid_official_bs1.json (+ .stdout.log)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "third_party", "codi")
REPO_URL = "https://github.com/zhenyi4/codi"
REPO_COMMIT = "2c2314662c63e9f482ebc46614ffe9af17a241e5"          # main @ 2025-12-15 "Add acceptance note for EMNLP 2025 in README" (ls-remote 2026-10-21)
TESTPY_SHA = "93a8bdba36e0b116e3be3e9605fa2722c200691662e88fef8bc4ea3737b7bd51"   # sha256 of test.py at REPO_COMMIT (LF, git blob bytes)
MODELPY_SHA = "818fa2f88d2ff348a6242fc62ce244e4d94e220de727a1688058348247440d00"  # sha256 of src/model.py at REPO_COMMIT
CKPT_DIR = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "codi")
CKPT_SHA = "fd223b14932b8b66605c2a688e7d7193058ff2b40b2c6f6c1547c227d06d8417"     # = boxes/codi_g1.SHA
TEST = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "data", "gsm8k_test.json")
TEST_SHA = "3d497ddbd728358309e25046b47a383adf582428cd3b5574767a55376b291d8d"
RUN_DIR = os.path.join(ROOT, "scratch", "s220", "zen_codi_gpt2")
PINNED = os.path.join(RUN_DIR, "eval_gcs.json")
PIN_SHA = "fec1f9ec1bda121d33485b8a325fb92e963756b3b0ef1f1fd195ba6b512de769"      # GCS2_v1 §3
OUT_NAME = "fid_official_bs1.json"
# scripts/test_gpt2.sh at REPO_COMMIT, verbatim order; only batch_size / ckpt_dir / output_dir differ (C1-C3)
CLI = ["--data_name", "gsm8k", "--output_dir", None, "--model_name_or_path", "gpt2", "--seed", "11", "--model_max_length", "512", "--bf16",
       "--lora_r", "128", "--lora_alpha", "32", "--lora_init", "--batch_size", "1", "--greedy", "True", "--num_latent", "6", "--use_prj", "True",
       "--prj_dim", "768", "--prj_no_ln", "False", "--prj_dropout", "0.0", "--inf_latent_iterations", "6", "--inf_num_iterations", "1",
       "--remove_eos", "True", "--use_lora", "True", "--ckpt_dir", CKPT_DIR]
TEST_GPT2_SH_BS = "128"
# test.py source lines that fix the decode semantics (each must occur exactly once; printed with line numbers by settings())
SRC_PINS = {
    "bs from CLI": "eval_step = math.ceil(len(question)/data_args.batch_size)",
    "pad side": 'padding_side="left",',
    "question format": "question = [f\"{example[question_name].strip().replace('  ', ' ')}\" for example in test_set]",
    "remove_eos -> q + <bot>": "bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch[\"input_ids\"].size(0), 1)",
    "dtype cast (D1)": "model.to(torch.bfloat16)",
    "max new tokens": '"max_new_tokens": 256,',
    "latent loop": "for i in range(inf_latent_iterations):",
    "logits slice (no <eot>)": "logits = out.logits[:, -1, :model.codi.config.vocab_size-1]",
    "greedy branch": "if training_args.greedy:",
    "greedy argmax": "next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)",
    "token appended before stop": "pred_tokens[b].append(next_token_ids[b].item())",
    "stop at EOS": "if next_token_ids[b] == tokenizer.eos_token_id:",
    "all finished": "if finished.all():",
    "decode": "decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)",
    "parser commas": "sentence = sentence.replace(',', '')",
    "parser regex": "pred = [s for s in re.findall(r'-?\\d+\\.?\\d*', sentence)]",
    "parser last number": "pred_answer = float(pred[-1])",
    "parser none -> inf": "return float('inf')",
    "accuracy": "if p == g:",
}
# D9: test.py cannot run at batch size 1 as written. logits is (B, V-1); argmax(dim=-1) is (B,); .squeeze(-1) makes it 0-dim when B == 1, and line 287
# `next_token_ids[b].item()` raises IndexError (CPU smoke 2026-10-21; any torch version). The one-line shape fix below keeps (B,) for every B
# (values identical: the same argmax; for B >= 2 squeeze(-1) is already a no-op). Applied to an in-memory copy compiled under test.py's own path;
# the checkout is never modified. The sampling branch (line 282, never reached with --greedy True) is left as is.
PATCH_OLD = "                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)"
PATCH_NEW = "                    next_token_ids = torch.argmax(logits, dim=-1).reshape(-1)"
PATCH_LINE = 261


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gen_sha(g):
    """= scratch/s220_gcs.py gen_sha (U10): sha256 of ",".join(generated ids incl. the stopping EOS)."""
    return hashlib.sha256(",".join(map(str, g)).encode()).hexdigest()


def git(cwd, *a):
    try:
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "error: %s" % e


def repo_state():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        raise SystemExit("official repo missing: run scratch/s221_official_setup.sh (git clone %s %s; checkout %s)" % (REPO_URL, REPO, REPO_COMMIT))
    st = {"path": os.path.relpath(REPO, ROOT).replace(os.sep, "/"), "url": REPO_URL, "head": git(REPO, "rev-parse", "HEAD"),
          "status": git(REPO, "status", "--porcelain"), "test_py_sha256": sha256_file(os.path.join(REPO, "test.py")),
          "model_py_sha256": sha256_file(os.path.join(REPO, "src", "model.py"))}
    assert st["head"] == REPO_COMMIT, ("official repo not at the pinned commit", st["head"])
    assert st["status"] == "", ("official repo working tree not clean", st["status"])
    assert st["test_py_sha256"] == TESTPY_SHA and st["model_py_sha256"] == MODELPY_SHA, ("test.py / src/model.py bytes differ from the pin (CRLF?)", st)
    return st


def pinned_meta():
    """the pinned eval_gcs.json: its sha256 and its meta ONLY (no other key is read)."""
    sha = sha256_file(PINNED); assert sha == PIN_SHA, ("pinned eval_gcs.json sha mismatch", sha)
    with open(PINNED) as f:
        meta = json.load(f)["meta"]
    return sha, meta


def settings(cli):
    """print test.py's decode-fixing source lines (line numbers) and the parsed CLI dataclass values; returns both for the meta."""
    src = open(os.path.join(REPO, "test.py"), encoding="utf-8").read().split("\n"); lines = {}
    for name, s in SRC_PINS.items():
        hits = [i + 1 for i, ln in enumerate(src) if ln.strip() == s]
        assert len(hits) == 1, ("test.py source pin not found exactly once", name, s, hits)
        lines[name] = {"line": hits[0], "src": s}
        print("SETTING test.py:%-4d %-26s %s" % (hits[0], name, s), flush=True)
    import transformers
    from src.model import DataArguments, ModelArguments, TrainingArguments
    ma, da, ta = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=cli)
    vals = {"data_name": da.data_name, "batch_size": da.batch_size, "model_name_or_path": ma.model_name_or_path, "ckpt_dir": os.path.relpath(ma.ckpt_dir, ROOT).replace(os.sep, "/"),
            "lora_init": ma.lora_init, "lora_r": ma.lora_r, "lora_alpha": ma.lora_alpha, "full_precision": ma.full_precision, "greedy": ta.greedy,
            "num_latent": ta.num_latent, "use_prj": ta.use_prj, "prj_dim": ta.prj_dim, "prj_no_ln": ta.prj_no_ln, "prj_dropout": ta.prj_dropout,
            "inf_latent_iterations": ta.inf_latent_iterations, "inf_num_iterations": ta.inf_num_iterations, "remove_eos": ta.remove_eos,
            "use_lora": ta.use_lora, "bf16": ta.bf16, "model_max_length": ta.model_max_length, "seed": ta.seed}
    print("SETTING parsed CLI", json.dumps(vals), flush=True)
    exp = {"batch_size": 1, "greedy": True, "inf_latent_iterations": 6, "inf_num_iterations": 1, "remove_eos": True, "use_prj": True, "prj_no_ln": False,
           "prj_dropout": 0.0, "data_name": "gsm8k", "use_lora": True, "lora_r": 128, "lora_alpha": 32}
    bad = {k: (vals[k], v) for k, v in exp.items() if vals[k] != v}; assert not bad, ("parsed CLI differs from the expected settings", bad)
    print("SETTING deviations from scripts/test_gpt2.sh: batch_size %s -> 1; ckpt_dir / output_dir paths" % TEST_GPT2_SH_BS, flush=True)
    assert src[PATCH_LINE - 1] == PATCH_OLD and sum(ln == PATCH_OLD for ln in src) == 1, "D9 patch target moved"
    print("SETTING D9 bs-1 shape fix, test.py:%d\n  - %s\n  + %s" % (PATCH_LINE, PATCH_OLD.strip(), PATCH_NEW.strip()), flush=True)
    return {"source_pins": lines, "parsed_cli": vals, "D9_patch": {"line": PATCH_LINE, "old": PATCH_OLD.strip(), "new": PATCH_NEW.strip()}}


def patched_test_source():
    """test.py's bytes with exactly the D9 line replaced (asserted: one line differs, at PATCH_LINE)."""
    p = os.path.join(REPO, "test.py"); src = open(p, encoding="utf-8").read(); a_ = src.split("\n")
    assert a_[PATCH_LINE - 1] == PATCH_OLD and src.count(PATCH_OLD) == 1
    b_ = list(a_); b_[PATCH_LINE - 1] = PATCH_NEW
    assert [i for i, (x, y) in enumerate(zip(a_, b_)) if x != y] == [PATCH_LINE - 1] and len(a_) == len(b_)
    s2 = "\n".join(b_)
    return p, s2, hashlib.sha256(s2.encode("utf-8")).hexdigest()


def synthetic(n, seed=0):
    """template arithmetic questions (NOT GSM8K; smokes only), answers in GSM8K's '... #### N' form. = scratch/s220_gcs.py synthetic(n) (its
    questions and golds, imported read-only; smoke only), so a smoke can compare the official path with an s220 --synthetic smoke item for item."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s220_gcs_smoke", os.path.join(ROOT, "scratch", "s220_gcs.py")); S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S); it = S.synthetic(n, seed)
    return [x["q"] for x in it], ["%s #### %s" % (x["cot"], x["gold_s"]) for x in it]


def synthetic_template(n, seed=0):
    import random
    rng = random.Random(seed); qs, ans = [], []
    for i in range(n):
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        qs.append("Ann has %d apples and buys %d more. How many apples does Ann have?" % (a, b)); ans.append("<<%d+%d=%d>> #### %d" % (a, b, a + b, a + b))
    return qs, ans


def local_test():
    loc = json.load(open(TEST, encoding="utf-8"))
    return loc, list(loc["question"]), [float(str(x).replace(",", "").split("####")[-1].strip()) for x in loc["answer"]]


def hub_test(orig_ld, x, k, loc_q, loc_g):
    """test.py's load_dataset("gsm8k", "main"). D3b: this hub client rejects the bare legacy id ("Repository id must be 'namespace/name'"),
    so "gsm8k" is resolved to its canonical "openai/gsm8k" (recorded); the split must equal the port's json item for item, else abort."""
    x = list(x); comp = None
    if x and x[0] == "gsm8k":
        x[0] = "openai/gsm8k"; comp = "load_dataset id 'gsm8k' -> 'openai/gsm8k' (D3b)"
    d = orig_ld(*x, **k); t = d["test"]; hq = list(t["question"]); hg = [float(s.split("####")[-1].replace(",", "").strip()) for s in t["answer"]]
    info = {"source": "HF hub %s" % (tuple(x),), "compat": comp, "n": len(hq), "fingerprint": getattr(t, "_fingerprint", None),
            "cache_files": [os.path.basename(c["filename"]) for c in t.cache_files], "cache_dir_rev": [os.path.basename(os.path.dirname(c["filename"])) for c in t.cache_files],
            "questions_equal_local": hq == loc_q, "golds_equal_local": hg == loc_g}
    assert info["questions_equal_local"] and info["golds_equal_local"], ("HF gsm8k test split differs from the port's gsm8k_test.json", info)
    return d, info


class Tee:
    """test.py's stdout -> the log file only; counts its Prediction= lines."""
    def __init__(self, f):
        self.f = f; self.pred_lines = []; self.avg = None; self.buf = ""

    def write(self, s):
        self.f.write(s); self.buf += s
        while "\n" in self.buf:
            ln, self.buf = self.buf.split("\n", 1)
            m = re.match(r"Prediction=(.*); Groundtruth=", ln)
            if m:
                self.pred_lines.append(m.group(1))
            m = re.match(r"Average accuracy over \d+ sampling: (.*)$", ln)
            if m:
                self.avg = float(m.group(1))
        return len(s)

    def flush(self):
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, help="cuda:N (sets CUDA_VISIBLE_DEVICES=N; test.py uses 'cuda') or cpu (smokes / --settings_only)")
    ap.add_argument("--dtype", required=True, choices=["fp32", "bf16"], help="D1: fp32 = patches P1+P2; bf16 = test.py's own cast, no patch")
    ap.add_argument("--settings_only", action="store_true"); ap.add_argument("--smoke_cpu", action="store_true"); ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--hf_model", default=None, help="smoke only: replaces --model_name_or_path gpt2 (e.g. openai-community/gpt2, same snapshot)")
    ap.add_argument("--local_test", action="store_true", help="D3 fallback: serve the local gsm8k_test.json instead of the HF hub split (listed deviation)")
    ap.add_argument("--out_dir", default=None); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--check_data", action="store_true", help="only load test.py's dataset (D3 / D3b) and verify it against the local json; no model")
    a = ap.parse_args()
    smoke = bool(a.smoke_cpu or a.synthetic or a.hf_model)
    tmp_root = os.path.abspath(os.path.join(ROOT, "scratch", "_tmp"))
    out_dir = os.path.abspath(a.out_dir or RUN_DIR)
    a.settings_only = a.settings_only or a.check_data
    if smoke or a.settings_only:
        assert a.out_dir and out_dir.startswith(tmp_root + os.sep), "smokes / --settings_only write under scratch/_tmp only"
    if smoke:
        assert a.smoke_cpu and a.synthetic and a.device == "cpu", "smokes are CPU + synthetic (never GSM8K items)"
    else:
        assert a.settings_only or a.device.startswith("cuda:"), "the real run needs --device cuda:N"
    if a.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = a.device.split(":", 1)[1]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); sys.dont_write_bytecode = True                       # D5, D7
    t0 = time.time()
    st = repo_state(); ck = sha256_file(os.path.join(CKPT_DIR, "pytorch_model.bin")); assert ck == CKPT_SHA, ("checkpoint sha256 mismatch", ck)
    pin_sha, pmeta = pinned_meta(); assert pmeta["model_sha256"] == CKPT_SHA, ("pinned meta model sha != checkpoint", pmeta["model_sha256"])
    tsha = sha256_file(TEST); assert tsha == TEST_SHA == pmeta["test_sha256"], ("test json sha", tsha)
    os.makedirs(out_dir, exist_ok=True); fo = os.path.join(out_dir, OUT_NAME)
    assert a.settings_only or not os.path.exists(fo), "refusing to overwrite %s" % fo
    cli = list(CLI); cli[cli.index("--output_dir") + 1] = os.path.join(tmp_root, "s221_official_outdir")
    if a.hf_model:
        cli[cli.index("--model_name_or_path") + 1] = a.hf_model
    os.chdir(REPO); sys.path.insert(0, REPO)                                                                              # run from its own directory
    import torch
    torch.set_num_threads(a.threads); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    card = "cpu"
    if a.device.startswith("cuda:"):
        assert torch.cuda.device_count() == 1, torch.cuda.device_count()
        card = torch.cuda.get_device_name(0); assert card == pmeta["card"], ("visible card %s != pinned card %s" % (card, pmeta["card"]))
    import datasets
    import transformers
    import src.model as SM
    S = settings(cli)
    if a.check_data:                                                                                  # D3 / D3b, no model
        _, lq, lg = local_test(); _, info = hub_test(datasets.load_dataset, ("gsm8k", "main"), {}, lq, lg); print("CHECK_DATA", json.dumps(info), flush=True); return
    if a.settings_only:
        print("SETTINGS_ONLY ok (repo %s clean, test.py %s, ckpt %s, pinned %s)" % (REPO_COMMIT[:8], TESTPY_SHA[:8], CKPT_SHA[:8], pin_sha[:8]), flush=True); return

    rec = {"decode": [], "batches": [], "from_pretrained": [], "dtype_overrides": 0, "bf16_cast_skips": 0, "load_state_dict": None, "dataset": None}

    def from_test_eval(depth=2):
        f = sys._getframe(depth)
        return f.f_code.co_name == "evaluation" and os.path.basename(f.f_code.co_filename) == "test.py"

    # D2 tokenizer hooks (observation only)
    def wrap_tok(tok):
        base = type(tok)

        class Rec(base):
            def __call__(self, *x, **k):
                out = base.__call__(self, *x, **k)
                if from_test_eval():
                    am = out["attention_mask"]; rec["batches"].append({"n": int(am.shape[0]), "len": int(am.shape[1]), "pads": int((am == 0).sum())})
                return out

            def decode(self, ids, *x, **k):
                s = base.decode(self, ids, *x, **k)
                if from_test_eval():
                    rec["decode"].append({"ids": [int(t) for t in ids], "txt": s, "kw": sorted(k.items())})
                    if len(rec["decode"]) % 100 == 0:
                        print("[%6.0fs] official items %d" % (time.time() - t0, len(rec["decode"])), file=sys.__stdout__, flush=True)
                return s
        Rec.__name__ = base.__name__; Rec.__qualname__ = base.__qualname__; tok.__class__ = Rec
        rec.setdefault("tok_class", []).append("%s.%s" % (base.__module__, base.__name__))
        return tok
    orig_tok = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = lambda *x, **k: wrap_tok(orig_tok(*x, **k))
    # D1 P1 (fp32 only) + model observation
    orig_fp = transformers.AutoModelForCausalLM.from_pretrained

    def fp(*x, **k):
        if "resume_download" in k and int(transformers.__version__.split(".")[0]) >= 5:                              # D8 (API compat only)
            rec.setdefault("compat", []).append("from_pretrained: dropped resume_download=%s (removed in transformers 5)" % k.pop("resume_download"))
        if a.dtype == "fp32":
            for key in ("torch_dtype", "dtype"):
                if key in k:
                    k[key] = torch.float32; rec["dtype_overrides"] += 1
        m = orig_fp(*x, **k)
        rec["from_pretrained"].append({"args": [str(v) for v in x], "kwargs": {kk: str(v) for kk, v in k.items()}, "param_dtype": str(next(m.parameters()).dtype),
                                       "attn_implementation": getattr(m.config, "_attn_implementation", None), "commit_hash": getattr(m.config, "_commit_hash", None)})
        return m
    transformers.AutoModelForCausalLM.from_pretrained = fp
    # D1 P2 (fp32 only)
    orig_to = torch.nn.Module.to

    def codi_to(self, *x, **k):
        if a.dtype == "fp32" and (any(v is torch.bfloat16 for v in x) or k.get("dtype") is torch.bfloat16):
            rec["bf16_cast_skips"] += 1; return self
        return orig_to(self, *x, **k)
    SM.CODI.to = codi_to
    # D4 load_state_dict observation
    orig_lsd = torch.nn.Module.load_state_dict

    def codi_lsd(self, sd, *x, **k):
        r = orig_lsd(self, sd, *x, **k); params = {n for n, _ in self.named_parameters()}
        rec["load_state_dict"] = {"n_keys": len(sd), "missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys),
                                  "missing_params": [q for q in r.missing_keys if q in params]}
        assert not r.unexpected_keys and not rec["load_state_dict"]["missing_params"], ("checkpoint keys do not match the CODI module", rec["load_state_dict"])
        return r
    SM.CODI.load_state_dict = codi_lsd
    # D3 dataset observation (or --local_test / smoke synthetic)
    loc, loc_q, loc_g = local_test()
    orig_ld = datasets.load_dataset

    def ld(*x, **k):
        if smoke:
            qs, an = synthetic(a.synthetic); d = datasets.DatasetDict({"test": datasets.Dataset.from_dict({"question": qs, "answer": an})})
            rec["dataset"] = {"source": "synthetic (smoke)", "n": len(qs)}; return d
        if a.local_test:
            d = datasets.DatasetDict({"test": datasets.Dataset.from_dict({"question": loc_q, "answer": ["#### " + str(x) for x in loc["answer"]]})})
            rec["dataset"] = {"source": "local json (D3 fallback)", "n": len(loc_q)}; return d
        d, rec["dataset"] = hub_test(orig_ld, x, k, loc_q, loc_g)
        return d
    datasets.load_dataset = ld
    # smoke only: test.py's hard-coded "cuda" -> cpu
    if a.smoke_cpu:
        def dv(k):
            if str(k.get("device", "")).startswith("cuda"):
                k["device"] = "cpu"
            return k
        ot, oz, oT = torch.tensor, torch.zeros, torch.Tensor.to
        torch.tensor = lambda *x, **k: ot(*x, **dv(k)); torch.zeros = lambda *x, **k: oz(*x, **dv(k))
        torch.Tensor.to = lambda self, *x, **k: oT(self, *[("cpu" if str(v).startswith("cuda") else v) for v in x], **dv(k))
        ocodi_to = SM.CODI.to
        SM.CODI.to = lambda self, *x, **k: ocodi_to(self, *[("cpu" if str(v).startswith("cuda") else v) for v in x], **dv(k))

    log_path = fo[:-5] + ".stdout.log"; assert not os.path.exists(log_path), "refusing to overwrite %s" % log_path
    sys.argv = ["test.py"] + cli
    tpath, tsrc, tsrc_sha = patched_test_source()
    code = compile(tsrc, tpath, "exec"); g = {"__name__": "__main__", "__file__": tpath, "__builtins__": __builtins__}   # = runpy.run_path(test.py) + D9
    with open(log_path, "w", newline="\n", encoding="utf-8") as lf:
        tee = Tee(lf); so = sys.stdout; sys.stdout = tee
        try:
            exec(code, g)
        finally:
            sys.stdout = so
    print("[%6.0fs] official test.py finished; %d decoded items" % (time.time() - t0, len(rec["decode"])), flush=True)
    ex = g["extract_answer_number"]; N = len(rec["decode"])
    golds = [float(s.split("####")[-1].replace(",", "").strip()) for s in synthetic(a.synthetic)[1]] if smoke else loc_g
    assert N == len(golds), ("decoded items != test items", N, len(golds))
    assert len(rec["batches"]) == N and all(b["n"] == 1 and b["pads"] == 0 for b in rec["batches"]), "bs-1 / no-pad check failed"
    preds_raw = [ex(d["txt"]) for d in rec["decode"]]
    assert len(tee.pred_lines) == N and all(str(p) == s for p, s in zip(preds_raw, tee.pred_lines)), "recorded preds != test.py's printed Prediction= lines"
    acc = sum(1.0 for p, gd in zip(preds_raw, golds) if p == gd) / N
    assert tee.avg is not None and abs(100 * acc - tee.avg) < 1e-9, ("recomputed accuracy != test.py's printed average", 100 * acc, tee.avg)
    pred = {str(i): (None if (isinstance(p, float) and p == float("inf")) else p) for i, p in enumerate(preds_raw)}
    assert repo_state()["status"] == "", "official repo dirty after the run"
    import peft
    import accelerate
    meta = {"model": "zen", "kind": "official_bs1", "card": card, "device": a.device, "dtype": a.dtype,
            "dtype_patches": {"P1_from_pretrained_fp32": rec["dtype_overrides"], "P2_bf16_cast_skipped": rec["bf16_cast_skips"]} if a.dtype == "fp32" else "none (test.py's own bf16)",
            "test_py_executed_sha256": tsrc_sha, "D9_patch": S["D9_patch"], "compat": rec.get("compat", []),
            "bs": 1, "n_items": N, "batches": len(rec["batches"]), "pads_total": sum(b["pads"] for b in rec["batches"]), "repo": st, "cli": cli,
            "cli_deviations_vs_test_gpt2_sh": ["--batch_size 128 -> 1", "--ckpt_dir", "--output_dir"] + (["--model_name_or_path %s (smoke)" % a.hf_model] if a.hf_model else []),
            "settings": S, "from_pretrained": rec["from_pretrained"], "load_state_dict": {k: (v if k != "missing" else v[:20]) for k, v in rec["load_state_dict"].items()},
            "dataset": rec["dataset"], "decode_kwargs": rec["decode"][0]["kw"] if rec["decode"] else None, "tokenizer_class": rec.get("tok_class"),
            "ckpt_sha256": ck, "test_sha256": tsha, "pinned_eval_gcs_sha256": pin_sha, "pinned_card": pmeta["card"],
            "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "accelerate": accelerate.__version__, "datasets": datasets.__version__, "tf32": False, "deterministic": "use_deterministic_algorithms(True, warn_only=True)",
            "threads": a.threads, "commit": git(ROOT, "rev-parse", "HEAD"), "script_sha256": sha256_file(os.path.abspath(__file__)),
            "script_git_status": git(ROOT, "status", "--porcelain", "--", "scratch/s221_official_zen.py"), "stdout_log": os.path.basename(log_path),
            "smoke": smoke, "synthetic": a.synthetic or None, "local_test": a.local_test, "sec": round(time.time() - t0),
            "registered": "docs/research_scaleprobe_2026-10-20/GCS2_v1.md at b7c66d8f (G-FID(a))"}
    out = {"kind": "official_bs1", "ref": {"pred": pred, "acc": acc, "sha": {str(i): gen_sha(d["ids"]) for i, d in enumerate(rec["decode"])},
                                           "ntok": {str(i): len(d["ids"]) for i, d in enumerate(rec["decode"])}, "repo_commit": REPO_COMMIT, "ckpt_sha256": ck,
                                           "sha_def": "U10 over test.py's own pred_tokens (incl. the stopping EOS); pred: test.py's extract_answer_number, inf -> None"},
           "meta": meta}
    tmp = fo + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        json.dump(out, f)
    os.replace(tmp, fo)
    print("WROTE", os.path.relpath(fo, ROOT), "items", N, "sha256", sha256_file(fo), flush=True)


if __name__ == "__main__":
    main()
