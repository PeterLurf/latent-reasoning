"""Build a searchable index of past experiments from files already in the repo.

Outputs (docs/history/index/):
  runs.csv / runs.jsonl          one row per run dir under boxes/runs*/
  sections.csv / sections.jsonl  one row per section (DIGEST_*.md tables, plus
                                 THEORY.md summary-table rows the digests lack)

Read-only over the repo except for the index dir. Never reads .pt files and
never walks run dirs recursively (globs args.json / eval*.json directly).

Rebuild:  python -B tools/build_exp_index.py
"""
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "history", "index")

ARG_KEYS = ["base", "d_max", "steps", "seed", "micro", "init_model", "init_trunk",
            "perm_p", "distract_p", "distract_mix", "hold_inc", "set_law",
            "mul_s_p", "train_gpu", "arm", "task"]
DIGESTS = ["DIGEST_s000-096.md", "DIGEST_s097-170.md", "DIGEST_s171-228.md"]
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def read_text(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cell(v):
    """Scalar -> str for CSV; lists/dicts -> compact JSON; None -> ''."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def git_add_dates():
    """path -> date of first commit adding it (one git log pass)."""
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "log", "--diff-filter=A", "--name-only",
             "--format=@@%ad", "--date=short", "--", "boxes/runs*/*/args.json"],
            capture_output=True, check=True).stdout.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print("git log failed:", e, file=sys.stderr)
        return {}
    dates, cur = {}, ""
    for line in out.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
        elif line.strip():
            dates[line.strip()] = cur  # log is newest-first: last write = oldest add
    return dates


def ledger_costs():
    """run -> summed trailing $ field of 'cell:run ... $x/h ... cost' ledger lines."""
    costs = {}
    pat = re.compile(r"^\d+:(\S+)\s.*\$[0-9.]+/h.*\s([0-9]+(?:\.[0-9]+)?)\s*$")
    for p in sorted(glob.glob(os.path.join(ROOT, "logs", "s2*_admitted*.log"))):
        for line in read_text(p).splitlines():
            m = pat.match(line.strip())
            if m:
                costs[m.group(1)] = costs.get(m.group(1), 0.0) + float(m.group(2))
    return {k: round(v, 4) for k, v in costs.items()}


def get_path(d, *keys):
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return None


def split_row(line):
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return None
    return [c.strip() for c in s[1:-1].split("|")]


def sec_key(num):
    m = re.match(r"(\d+)", num)
    return (int(m.group(1)) if m else 10**9, num)


def parse_sections():
    rows = {}
    for name in DIGESTS:
        for line in read_text(os.path.join(ROOT, "docs", "history", name)).splitlines():
            c = split_row(line)
            if not c or len(c) < 7 or not re.match(r"^\d", c[0]):
                continue
            if len(c) > 7:  # stray '|' inside a cell: keep the tail in the C column
                c = c[:6] + [" | ".join(c[6:])]
            rows[c[0]] = {"section": c[0], "name": c[1], "question": c[2], "verdict": c[3],
                          "key_numbers": c[4], "reading": c[5], "criteria": c[6],
                          "status": "", "source": "docs/history/" + name}
    # digest keys like "0-1", "3-5", "6,15", "54/54b" cover several table numbers
    covered = set()
    for k in rows:
        for part in re.split(r"[,/]", k):
            m = re.match(r"^(\d+)-(\d+)$", part.strip())
            if m:
                covered.update(str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1))
            else:
                covered.add(part.strip())
    # THEORY.md summary table: status for all, full row for sections not in digests
    theory = read_text(os.path.join(ROOT, "docs", "THEORY.md"))
    m = re.search(r"^## Sections .*?$(.*?)^## ", theory, re.S | re.M)
    for line in (m.group(1) if m else "").splitlines():
        c = split_row(line)
        if not c or len(c) < 4 or not re.match(r"^\d", c[0]):
            continue
        num, status = c[0], c[-1]
        if num in rows:
            rows[num]["status"] = status
            continue
        if num in covered:
            continue
        q = " | ".join(c[1:-2])
        mm = re.match(r"([A-Z0-9][A-Z0-9 /\-+]*[A-Z0-9])\b", q)
        rows[num] = {"section": num, "name": mm.group(1) if mm else "", "question": q,
                     "verdict": c[-2], "key_numbers": "", "reading": "", "criteria": "",
                     "status": status, "source": "docs/THEORY.md (summary table)"}
    for r in rows.values():
        r["link"] = "docs/THEORY.md#" + r["section"]
    return [rows[k] for k in sorted(rows, key=sec_key)]


def section_bodies(sections):
    """section number -> token set of its body text (THEORY.md + archive + digest/table rows)."""
    texts = {}
    for p in [os.path.join(ROOT, "docs", "archive", "THEORY_s0-s96.md"),
              os.path.join(ROOT, "docs", "THEORY.md")]:
        cur = None
        for line in read_text(p).splitlines():
            m = re.match(r"^## (\d+)\.", line)
            if m:
                cur = m.group(1)
            elif line.startswith("## "):
                cur = None
            if cur is not None:
                texts.setdefault(cur, []).append(line)
    for s in sections:
        texts.setdefault(s["section"], []).append(
            " ".join(s[k] for k in ("name", "question", "verdict", "key_numbers", "reading")))
    out = {}
    for k, lines in texts.items():
        toks = set()
        for t in TOKEN_RE.findall("\n".join(lines)):
            toks.add(t)
            toks.update(t.split("."))
        out[k] = toks
    return out


def build_runs(sections):
    dates = git_add_dates()
    costs = ledger_costs()
    bodies = section_bodies(sections)
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "boxes", "runs*", "*"))
                  if os.path.isdir(d))
    rows = []
    for d in dirs:
        run = os.path.basename(d)
        base_name = run.split(".")[0]
        r = {"run": run, "dir": rel(d)}
        args_p = os.path.join(d, "args.json")
        r["args_added"] = dates.get(rel(args_p), "")
        args = load_json(args_p) if os.path.isfile(args_p) else None
        r["has_args"] = "1" if isinstance(args, dict) else "0"
        for k in ARG_KEYS:
            r[k] = cell(args.get(k)) if isinstance(args, dict) else ""
        evals = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "eval*.json")))
        r["eval_files"] = ";".join(evals)
        r["has_train_summary"] = "1" if os.path.isfile(os.path.join(d, "train_summary.json")) else "0"
        acc = load_json(os.path.join(d, "eval_acc.json")) if "eval_acc.json" in evals else None
        for dd in range(1, 9):
            r["acc_d%d" % dd] = cell(get_path(acc, "by_d", str(dd), "acc"))
        r["nodec_k2"] = cell(get_path(acc, "by_d", "4", "acc_kprime", "2"))
        tp = load_json(os.path.join(d, "eval_test_perm.json")) if "eval_test_perm.json" in evals else None
        for dd in range(2, 5):
            r["perm_d%d_k8" % dd] = cell(get_path(tp, "by_d", str(dd), "acc_kprime", "8"))
        r["ledger_usd_est"] = cell(costs.get(base_name))
        secs = []
        if len(base_name) >= 5:
            secs = sorted((k for k, toks in bodies.items() if base_name in toks), key=sec_key)
        r["sections"] = ";".join(secs)
        rows.append(r)
    return rows


def write_all(name, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    with open(os.path.join(OUT, name + ".csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    sections = parse_sections()
    runs = build_runs(sections)
    write_all("sections", sections)
    write_all("runs", runs)
    print("sections %d  runs %d  -> %s" % (len(sections), len(runs), rel(OUT)))


if __name__ == "__main__":
    main()
