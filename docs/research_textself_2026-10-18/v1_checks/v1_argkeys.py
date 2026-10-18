"""TS_v1 check A (T1 / RT B1): the TEXT_NEUTRAL_NEW map. AST-reads boxes/nested170.py's argparse block (no import, no execution), lists every key
absent from boxes/runs_fsq/s170_cot_s1/args.json with its parser default and action, the runtime / info keys main() writes into args.json, the
textcot-arm overrides (mem_write / write_mix_* / think_k), and which keys theory/score_s213.META already covers. Also: keys of the base args.json
that the current parser no longer writes (a missing-reference-key check the s216 args_diff does not have). Read only. python -B v1_argkeys.py"""
import ast
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "v1_argkeys.json")
SRC = "D:/Latent/boxes/nested170.py"; BASE = "D:/Latent/boxes/runs_fsq/s170_cot_s1/args.json"; S213 = "D:/Latent/theory/score_s213.py"


def lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return ("<expr>", ast.unparse(node))


def main():
    src = open(SRC, encoding="utf-8").read(); tree = ast.parse(src)
    keys = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flag = lit(node.args[0]) if node.args else None
            if not (isinstance(flag, str) and flag.startswith("--")):
                continue
            kw = {k.arg: lit(k.value) for k in node.keywords if k.arg in ("default", "action", "type", "choices", "required")}
            if "type" in [k.arg for k in node.keywords]:
                kw["type"] = ast.unparse([k.value for k in node.keywords if k.arg == "type"][0])
            act = kw.get("action")
            default = kw.get("default", False if act == "store_true" else None)
            keys[flag[2:]] = {"default": default, "action": act, "type": kw.get("type"), "required": kw.get("required", False), "line": node.lineno}
    base = json.load(open(BASE))
    meta_src = open(S213, encoding="utf-8").read()
    mt = ast.parse(meta_src); META = None
    for node in mt.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "META" for t in node.targets):
            META = sorted(ast.literal_eval(node.value))
    absent = {k: v for k, v in keys.items() if k not in base}
    # runtime / info keys main() writes into args.json (vars(a) after parse + assignments a.<x> = ..., plus the info dict extras)
    runtime = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                      and isinstance(n.value, ast.Name) and n.value.id == "a"} - set(keys))
    info_extra = ["train_gpu", "train_driver", "hostprobe_sps", "format", "<env_info() keys>", "aux_decoder (only if aux_step)"]
    base_not_in_parser = sorted(k for k in base if k not in keys)
    neutral = {k: v["default"] for k, v in absent.items()}
    res = {"src": SRC, "src_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(), "base_args": BASE, "n_parser_keys": len(keys),
           "parser_keys": keys, "absent_from_base": sorted(absent), "n_absent": len(absent), "TEXT_NEUTRAL_NEW_raw_defaults": neutral,
           "absent_covered_by_S213_META": sorted(k for k in absent if k in META), "S213_META": META,
           "runtime_assigned_attrs_not_in_parser": runtime, "info_extra_keys": info_extra,
           "base_keys_not_in_parser": base_not_in_parser, "base_values_of_those": {k: base[k] for k in base_not_in_parser},
           "base_textcot_overrides": {k: base.get(k) for k in ("mem_write", "write_mix_start", "write_mix_end", "think_k")},
           "parser_defaults_of_overridden": {k: keys[k]["default"] for k in ("mem_write", "write_mix_start", "write_mix_end", "think_k")},
           "base_values_vs_parser_defaults_present_keys": {k: {"base": base[k], "default": keys[k]["default"]} for k in base if k in keys and base[k] != keys[k]["default"]}}
    json.dump(res, open(OUT, "w", newline="\n"), indent=1, default=str)
    print("parser keys", len(keys), "absent from base", len(absent)); print("absent:", sorted(absent))
    print("absent in S213.META:", res["absent_covered_by_S213_META"]); print("runtime attrs:", runtime)
    print("base keys not in parser:", base_not_in_parser)
    print("defaults:", json.dumps(neutral, default=str))
    print("base != default (present keys):", json.dumps(res["base_values_vs_parser_defaults_present_keys"], default=str))


if __name__ == "__main__":
    main()
