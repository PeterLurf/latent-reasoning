"""Build report/latent_branch_report.html from results/summary.json (self-contained page)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "..", "results", "summary.json")))
extra = os.path.join(HERE, "..", "results", "extra.json")
X = json.load(open(extra)) if os.path.exists(extra) else {}
tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
render = open(os.path.join(HERE, "render.js"), encoding="utf-8").read()
for extra_js in ("render2.js", "render3.js"):
    if os.path.exists(os.path.join(HERE, extra_js)):
        render += "\n" + open(os.path.join(HERE, extra_js), encoding="utf-8").read()


# Part III inputs (architecture ablations, held-out seeds, test-time search, development)
def _j(*p):
    f = os.path.join(HERE, "..", *p)
    return json.load(open(f)) if os.path.exists(f) else None


P3 = {"protocol3": _j("results", "protocol3_confirm.json"),
      "budget": _j("results", "budget_sweep_all.json") or _j("results", "budget_sweep_wd.json"),
      "length": _j("results", "length_gen_all.json") or _j("results", "length_gen.json"),
      "development": _j("results", "development_dev_s4.json"),
      "upcache": {s: (_j("runs", r, "followup", "discovery_upcache", "summary.json") or {}).get("results")
                  for s, r in (("s0", "r3_pc_gated_gpt2"), ("s1", "main_s1"), ("s2", "main_s2"))},
      "integrity": _j("results", "integrity.json"),
      "rewind": {t: _j("results", f"rewind_grid_{t}.json") for t in ("s0", "s1", "s2", "s3", "v3")},
      "phase": _j("results", "phase_shift_s0.json"),
      "trace_up": _j("results", "upfront_trace_v3.json"),
      "trace_delayed": _j("results", "delayed_trace.json"),
      "trace_W": {"2": _j("results", "delayed_trace_W2.json"), "4": _j("results", "delayed_trace.json"),
                  "8": _j("results", "delayed_trace_W8.json")},
      "block": _j("results", "upfront_block_v3.json"),
      "delayed_block": _j("results", "delayed_block.json"),
      "delayed_block_new": _j("results", "delayed_block_new.json"),
      "length_up": _j("results", "length_gen_upfront.json"),
      "clock_swap": _j("results", "clock_swap.json"),
      "clock_trace": _j("results", "clock_trace.json"),
      "clock_freeze": {**(_j("results", "clock_freeze.json") or {}), **(_j("results", "clock_freeze_v1.json") or {}),
                       **(_j("results", "clock_freeze_k8.json") or {})}}
html = tpl.replace("/*__DATA__*/null", json.dumps({"S": S, "X": X, "P3": P3}, separators=(",", ":")).replace("NaN", "null"))
html = html.replace("/*__PROSE__*/", render)
open(os.path.join(HERE, "latent_branch_report.html"), "w", encoding="utf-8").write(html)
print("wrote report/latent_branch_report.html", len(html) // 1024, "KB")

# print variant for the PDF: light theme, all table views expanded, every heatmap cell labelled
pre = ('<script>window.PRINT = true; document.documentElement.dataset.theme = "light";</script>\n'
       '<style>@media print{svg.chart text{font-size:13px}}</style>\n')
post = '\n<script>document.querySelectorAll("details").forEach(d => d.open = true);</script>\n'
open(os.path.join(HERE, "latent_branch_report_print.html"), "w", encoding="utf-8").write(pre + html + post)
print("wrote report/latent_branch_report_print.html")
