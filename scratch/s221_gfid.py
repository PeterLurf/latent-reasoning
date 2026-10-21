"""§221 GSM-CACHE-SWAP-2, G-FID port side -> <run dir>/fid_port.json, then <run dir>/gate_fid.json (contract = theory/score_s221.py's docstring).
Registered text: docs/research_scaleprobe_2026-10-20/GCS2_v1.md at b7c66d8f (§2: G-FID(a) CODI20 determinism, G-FID(a') pairs[0:50] bitwise).
A NEW scratch file: edits nothing under boxes/ or theory/ and does not edit scratch/s220_gcs.py.

PROVENANCE: the §220 code path is NOT imported from the working tree. s220_gcs.py is read from git at the pinned file's own run commit
(meta.commit: CODI20 59a90e07, zen b1a6ae32), its bytes' sha256 asserted == meta.script_sha256 (575d4cfb..., = e5ea9325's G-D-reviewed version),
and executed as a module whose __file__ is scratch/s220_gcs.py (so its ROOT is the repo). boxes/codi_port.py's sha256 must equal meta.codi_port_sha256;
the model sha256 (load_model's own check) must equal meta.model_sha256; the test json sha256 meta.test_sha256; the card meta.card (same card, GCS2_v1 §2);
torch / transformers must equal meta's (else abort; --allow_env_drift records the drift instead). Process settings are s220 main()'s: TF32 off,
use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG :4096:8 (set by the module at import), torch threads --threads (2).

WHAT RUNS (the pinned module's own Runner / load_model / load_test; the arm code below is s220 main()'s, copied line for line):
  INTACT       R.decode(R.prefix(ids[k])) -> U10 token sha (+ parsed pred, stored for kind "determinism" only).
               CODI20 (kind "determinism"): all items in index order -> ref.sha {idx: sha}, ref.pred, ref.acc (G-FID(a)).
               zen (kind "official_bs1"): only the pairs[0:50] recipients (--full_intact: all items, stored under port_rerun, descriptive).
  pairs[0:50]  of the pinned file, (r, d) taken from it (never re-drawn), each in s220's op order: pr = prefix(r), pdn = prefix(d), ps = prefix(r)
               (unused here: s220's SELF source); cands with s220's rand_cand on np rng 22023 (consumed once per pair, as s220); CACHE-SWAP = copy(pr),
               write_rows(latent slots, all layers, k+v), SCIT on a clone (as s220's cache_arm with cands; output discarded), decode;
               LEAK-SEG = copy(pr), write_rows(leak slots Lq-6..Lq-1), decode. -> subset.cache_sha / leak_sha {"0".."49"}, subset.intact_sha {r}.
  Nothing decoded or parsed from a swap arm is stored or printed: only token shas (and the write_rows assertion result).
BITWISE ASSERTS (stored under "asserts", printed as counts only): ref.sha vs the pinned intact_sha (determinism), subset shas vs the pinned
intact_sha / pairs[i].sha.cache / pairs[i].sha.leak. Of the pinned file only meta, intact_sha and pairs[0:50]'s r / d / Lq / sha.cache / sha.leak
are read (pinned_view); no arm value (out / txt / acc / k_sweep / variants / ...) is touched.
ASSEMBLY: kind determinism -> gate_fid.json at once. kind official_bs1 -> gate_fid.json when <run dir>/fid_official_bs1.json (scratch/s221_official_zen.py)
exists, else later with --assemble_only. Every output refuses to overwrite.
SMOKES (CPU, CUDA_VISIBLE_DEVICES=""): --model tiny --pinned scratch/_tmp/.../eval_gcs.json (an s220_gcs.py tiny --synthetic --smoke_all_correct run)
--out_dir scratch/_tmp/...; --smoke_kind official_bs1 exercises the zen assembly with a smoke fid_official_bs1.json.
    python scratch/s221_gfid.py --model codi20 --device cuda:2      # RTX 2070 -> boxes/runs_fsq/gsm_codi_20ep_s1/{fid_port,gate_fid}.json
    python scratch/s221_gfid.py --model zen --device cuda:1         # RTX 3070 -> scratch/s220/zen_codi_gpt2/fid_port.json (+ gate_fid.json if the official ref exists)
    python scratch/s221_gfid.py --model zen --assemble_only --device cpu
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = {"codi20": os.path.join(ROOT, "boxes", "runs_fsq", "gsm_codi_20ep_s1", "eval_gcs.json"),
          "zen": os.path.join(ROOT, "scratch", "s220", "zen_codi_gpt2", "eval_gcs.json")}
PIN = {"codi20": "82aa84138eea9a390d713cdb0f09c4f7e78deac295a09519ce2f8d716b6bef36",
       "zen": "fec1f9ec1bda121d33485b8a325fb92e963756b3b0ef1f1fd195ba6b512de769"}       # GCS2_v1 §3 (= score_s221.PIN)
KIND = {"codi20": "determinism", "zen": "official_bs1", "tiny": "determinism"}            # = score_s221.FID_KIND
NSUB = 50                                                                                 # GCS2_v1 §2: pairs[0:50]
S220_REL = "scratch/s220_gcs.py"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "error: %s" % e


def dump(path, obj, overwrite=False):
    assert overwrite or not os.path.exists(path), "refusing to overwrite %s" % path
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def pinned_view(path, nsub=NSUB):
    """sha256 + meta + intact_sha + pairs[0:nsub] (r, d, Lq, sha.cache, sha.leak) of a pinned eval_gcs.json; every other key is dropped unread."""
    sha = sha256_file(path)
    with open(path) as f:
        full = json.load(f)
    v = {"sha256": sha, "meta": full["meta"], "intact_sha": dict(full["intact_sha"]), "n_pairs": len(full["pairs"]),
         "pairs": [{"r": int(p["r"]), "d": int(p["d"]), "Lq": int(p["Lq"]), "sha_cache": p["sha"]["cache"], "sha_leak": p["sha"]["leak"]} for p in full["pairs"][:nsub]]}
    del full
    return v


def load_pinned_s220(commit, want_sha):
    """scratch/s220_gcs.py exactly as at the pinned run commit (bytes from git, sha256-asserted), as a module with the repo-path __file__."""
    src = subprocess.run(["git", "show", "%s:%s" % (commit, S220_REL)], cwd=ROOT, capture_output=True, timeout=60).stdout
    sha = hashlib.sha256(src).hexdigest(); assert sha == want_sha, ("s220_gcs.py at %s has sha %s != pinned meta %s" % (commit, sha, want_sha))
    mod = types.ModuleType("s220_gcs_pinned"); mod.__file__ = os.path.join(ROOT, *S220_REL.split("/")); sys.modules["s220_gcs_pinned"] = mod
    exec(compile(src, "%s@%s" % (mod.__file__, commit[:8]), "exec"), mod.__dict__)
    return mod, sha


def assemble(out_dir, kind, smoke=False):
    """gate_fid.json = {"kind", "ref", "subset", "asserts", "meta"} from fid_port.json (+ fid_official_bs1.json for kind official_bs1)."""
    fo = os.path.join(out_dir, "gate_fid.json"); port = json.load(open(os.path.join(out_dir, "fid_port.json")))
    assert port["kind"] == kind, (port["kind"], kind)
    if kind == "determinism":
        g = {"kind": kind, "ref": port["ref"], "subset": port["subset"], "asserts": port["asserts"], "meta": port["meta"]}
    else:
        po = os.path.join(out_dir, "fid_official_bs1.json")
        if not os.path.exists(po):
            print("ASSEMBLE pending: %s missing (run scratch/s221_official_zen.py, then --assemble_only)" % os.path.relpath(po, ROOT), flush=True); return None
        off = json.load(open(po)); checks = {"ckpt_sha_equal": off["ref"]["ckpt_sha256"] == port["meta"]["model_sha256"],
                                             "pinned_sha_equal": off["meta"].get("pinned_eval_gcs_sha256") == port["meta"]["pinned_eval_gcs"]["sha256"],
                                             "n_items_equal": len(off["ref"]["pred"]) == port["meta"]["n_items"], "card_equal": off["meta"]["card"] == port["meta"]["card"]}
        if not smoke:
            assert all(checks.values()), ("official ref and port re-run do not describe the same model / file / card", checks)
        g = {"kind": kind, "ref": off["ref"], "subset": port["subset"], "asserts": {**port["asserts"], "assemble_checks": checks},
             "meta": {**port["meta"], "ref_meta": off["meta"], "fid_official_bs1_sha256": sha256_file(po)}}
    g["meta"] = {**g["meta"], "fid_port_sha256": sha256_file(os.path.join(out_dir, "fid_port.json")), "assembled": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    dump(fo, g); print("WROTE", os.path.relpath(fo, ROOT), "sha256", sha256_file(fo), flush=True)
    return fo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["codi20", "zen", "tiny"]); ap.add_argument("--device", required=True)
    ap.add_argument("--pinned", default=None, help="smoke only (tiny): an s220_gcs.py smoke eval_gcs.json under scratch/_tmp")
    ap.add_argument("--out_dir", default=None, help="default: the pinned file's directory; smokes: under scratch/_tmp")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("S220_THREADS", "2"))); ap.add_argument("--full_intact", action="store_true")
    ap.add_argument("--allow_env_drift", action="store_true"); ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--smoke_kind", default=None, choices=["official_bs1"], help="tiny only: exercise the official_bs1 assembly")
    a = ap.parse_args(); t0 = time.time()
    tmp_root = os.path.abspath(os.path.join(ROOT, "scratch", "_tmp"))
    smoke = a.model == "tiny" or a.pinned is not None                                  # a smoke = a --pinned s220 SMOKE file (tiny, or a real model on --synthetic)
    pinned = os.path.abspath(a.pinned) if a.pinned else PINNED.get(a.model)
    if smoke:
        assert pinned and pinned.startswith(tmp_root + os.sep) and a.out_dir, "smokes: --pinned and --out_dir under scratch/_tmp"
    else:
        assert a.smoke_kind is None, "real runs use the pinned file (GCS2_v1 §3)"
    out_dir = os.path.abspath(a.out_dir or os.path.dirname(pinned))
    if smoke or a.out_dir:
        assert out_dir.startswith(tmp_root + os.sep), "non-default outputs go under scratch/_tmp only"
    kind = a.smoke_kind or KIND[a.model]
    if a.assemble_only:
        assemble(out_dir, kind, smoke); return
    fport = os.path.join(out_dir, "fid_port.json"); fpart = os.path.join(out_dir, "fid_port.partial.json")
    assert not os.path.exists(fport), "refusing to overwrite %s" % fport
    assert not os.path.exists(os.path.join(out_dir, "gate_fid.json")), "refusing: gate_fid.json exists in %s" % out_dir
    pv = pinned_view(pinned); pm = pv["meta"]
    if smoke:
        assert pm.get("smoke") and pm["model"] == a.model and (pm.get("synthetic") or a.model == "tiny"), "smoke pinned files: s220 smokes of this model on tiny / synthetic items"
    else:
        assert pv["sha256"] == PIN[a.model], ("pinned eval_gcs.json sha mismatch", pv["sha256"])
        assert pm["model"] == a.model and not pm.get("smoke"), pm.get("model")
    assert len(pv["pairs"]) == min(NSUB, pv["n_pairs"]) and (smoke or len(pv["pairs"]) == NSUB)
    S, s220_sha = load_pinned_s220(pm["commit"], pm["script_sha256"])                    # sets CUBLAS_WORKSPACE_CONFIG before any CUDA use
    import numpy as np
    import torch
    import transformers
    cp_sha = S.sha256_file(os.path.join(ROOT, "boxes", "codi_port.py")); assert cp_sha == pm["codi_port_sha256"], ("boxes/codi_port.py changed", cp_sha)
    drift = {k: (pm.get(k), v) for k, v in (("torch", torch.__version__), ("transformers", transformers.__version__)) if pm.get(k) != v}
    assert not drift or a.allow_env_drift, ("library versions differ from the pinned run (--allow_env_drift records instead)", drift)
    dev = a.device; torch.set_num_threads(a.threads)
    card = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"
    if not smoke:
        assert dev.startswith("cuda") and card == pm["card"], ("same card required (GCS2_v1 §2): %s != %s" % (card, pm["card"]))
    assert card == pm["card"] or smoke
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True, warn_only=True)   # = s220 main()
    test_sha = S.sha256_file(S.TEST)
    if pm.get("synthetic"):
        items = S.synthetic(pm["synthetic"])
    else:
        assert test_sha == S.TEST_SHA == pm["test_sha256"], ("test file sha256 mismatch", test_sha)
        items = S.load_test()
    items = items[:pm["limit"]] if pm.get("limit") else items; N = len(items); assert N == pm["n_items"], (N, pm["n_items"])
    assert all(it["i"] == k for k, it in enumerate(items))
    m, prj, tok, info = S.load_model(a.model, dev); assert info["model_sha256"] == pm["model_sha256"], ("model sha256", info["model_sha256"])
    R = S.Runner(m, prj, tok, dev, pm["max_new"])
    os.makedirs(out_dir, exist_ok=True)
    meta = {"model": a.model, "kind": kind, "card": card, "device": dev, "dtype": pm["dtype"], "bs": 1, "attn_implementation": m.config._attn_implementation,
            "commit": git("rev-parse", "HEAD"), "script_sha256": sha256_file(os.path.abspath(__file__)), "script_git_status": git("status", "--porcelain", "--", "scratch/s221_gfid.py"),
            "s220_source": {"commit": pm["commit"], "sha256": s220_sha, "equals_pinned_meta_script_sha256": s220_sha == pm["script_sha256"], "from": "git show <commit>:%s" % S220_REL},
            "codi_port_sha256": cp_sha, "codi_g1_sha256": S.sha256_file(os.path.join(ROOT, "boxes", "codi_g1.py")),
            "torch": torch.__version__, "transformers": transformers.__version__, "env_drift_vs_pinned": drift, "model_sha256": info["model_sha256"],
            "model_path": info.get("path"), "test_sha256": test_sha if not pm.get("synthetic") else "synthetic", "n_items": N, "max_new": pm["max_new"],
            "pinned_eval_gcs": {"path": os.path.relpath(pinned, ROOT).replace(os.sep, "/"), "sha256": pv["sha256"], "card": pm["card"], "commit": pm["commit"]},
            "n_pairs_subset": len(pv["pairs"]), "threads": a.threads, "tf32": False, "deterministic": "use_deterministic_algorithms(True, warn_only=True)",
            "smoke": smoke, "registered": "docs/research_scaleprobe_2026-10-20/GCS2_v1.md at b7c66d8f (G-FID(a) / (a'))"}

    def log(*x):
        print("[%6.0fs]" % (time.time() - t0), *x, flush=True)
    log("START", a.model, kind, card, "s220@%s" % pm["commit"][:8], "items", N, "subset pairs", len(pv["pairs"]))
    ids = [R.ids(it["q"]) for it in items]; L = [len(x) for x in ids]; gold = [it["gold"] for it in items]
    rec_r = sorted({p["r"] for p in pv["pairs"]})
    for p in pv["pairs"]:                                                                                       # structure only (exact-length pairs)
        assert L[p["r"]] == L[p["d"]] == p["Lq"] and p["r"] != p["d"], ("pinned pair structure", p["r"], p["d"])
    full = kind == "determinism" or a.full_intact
    todo = list(range(N)) if full else rec_r
    isha, ipred, intk = {}, {}, {}
    for n_, k in enumerate(todo):                                                                               # 1. INTACT (s220 main() step 1)
        o = R.decode(R.prefix(ids[k])); isha[str(k)] = o["sha"]; ipred[str(k)] = o["pred"]; intk[str(k)] = o["ntok"]
        if n_ % 100 == 99:
            log("INTACT", n_ + 1, "/", len(todo)); dump(fpart, {"intact_sha": isha, "n_done": n_ + 1, "meta": meta}, overwrite=True)
    log("INTACT done", len(todo))
    # 2. pairs[0:50]: s220 main() step 5's per-pair code for CACHE-SWAP (with its SCIT) and LEAK-SEG, in its op order
    allsl = list(range(R.nl)); rng_rand = np.random.default_rng(S.SEED_RAND)

    def rand_cand(rng, excl):                                                                                   # = s220 main()'s rand_cand
        pool = [k for k in range(N) if all(gold[k] != e for e in excl if e is not None)]
        return items[pool[int(rng.integers(0, len(pool)))]]["gold_s"] if pool else None

    def cache_arm(base, src, dst_sl, src_sl, layers, kinds, tag, cands=None):                                   # = s220 main()'s cache_arm
        c = R.copy(base); R.write_rows(c, src, dst_sl, src_sl, layers, kinds, tag)
        sc = R.scit(c, cands) if cands else None
        return R.decode(c), sc
    csha, lsha = {}, {}
    for q, p in enumerate(pv["pairs"]):
        r, d = p["r"], p["d"]
        pr = R.prefix(ids[r]); pdn = R.prefix(ids[d]); ps = R.prefix(ids[r])                                   # ps: s220's SELF source (op order only)
        cands = {"recipient": items[r]["gold_s"], "donor": items[d]["gold_s"], "random": rand_cand(rng_rand, (gold[r], gold[d]))}
        o_c, _ = cache_arm(pr, pdn, R.latent_slots(pr), R.latent_slots(pdn), allsl, "kv", ("cache", r, d), cands)
        o_l, _ = cache_arm(pr, pdn, R.leak_slots(pr), R.leak_slots(pdn), allsl, "kv", ("leak", r, d))
        csha[str(q)] = o_c["sha"]; lsha[str(q)] = o_l["sha"]; del pr, pdn, ps, o_c, o_l
        if q % 10 == 9:
            log("pairs", q + 1, "/", len(pv["pairs"])); dump(fpart, {"intact_sha": isha, "cache_sha": csha, "leak_sha": lsha, "meta": meta}, overwrite=True)
    sub = {"intact_sha": {str(r): isha[str(r)] for r in [p["r"] for p in pv["pairs"]]}, "cache_sha": csha, "leak_sha": lsha}
    # 3. bitwise asserts vs the pinned shas (counts only)
    P = pv["pairs"]; ps_ = pv["intact_sha"]
    asserts = {"subset_intact_match": sum(sub["intact_sha"][str(p["r"])] == ps_[str(p["r"])] for p in P),
               "subset_cache_match": sum(csha[str(q)] == p["sha_cache"] for q, p in enumerate(P)),
               "subset_leak_match": sum(lsha[str(q)] == p["sha_leak"] for q, p in enumerate(P)), "subset_n": len(P),
               "cache_rows_ok": not R.fails and R.n_checks > 0, "n_row_checks": R.n_checks, "row_fails": [list(map(str, f)) for f in R.fails[:20]]}
    asserts["subset_bitwise"] = asserts["subset_intact_match"] == asserts["subset_cache_match"] == asserts["subset_leak_match"] == len(P)
    out = {"kind": kind, "subset": sub, "asserts": asserts, "meta": meta}
    if full:
        mt = sum(isha[k] == v for k, v in ps_.items()); asserts.update(intact_full_match=mt, intact_full_n=len(ps_), intact_full_bitwise=mt == len(ps_) == N)
        acc = float(np.mean([ipred[str(k)] is not None and ipred[str(k)] == gold[k] for k in range(N)]))
        blk = {"sha": isha, "pred": ipred, "ntok": intk, "acc": acc, "repo_commit": pm["commit"], "ckpt_sha256": info["model_sha256"],
               "def": "bs-1 INTACT re-run with s220_gcs.py at the pinned commit; sha = U10, pred = s220 parse_official"}
        if kind == "determinism":
            out["ref"] = blk
        else:
            out["port_rerun"] = blk                                                                             # zen --full_intact: descriptive only
    meta["sec"] = round(time.time() - t0)
    dump(fport, out); log("WROTE", os.path.relpath(fport, ROOT), "sha256", sha256_file(fport))
    log("BITWISE", json.dumps({k: v for k, v in asserts.items() if k != "row_fails"}))
    if os.path.exists(fpart):
        os.replace(fpart, fpart + ".done")
    assemble(out_dir, kind, smoke)


if __name__ == "__main__":
    main()
