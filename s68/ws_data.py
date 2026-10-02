"""THEORY §68 WS arm (written-state chunked): data construction.

Each GSM8K-Aug example (question, steps ["<<e1=v1>>", ...], answer) gives n + 1 calls:
    input  = question + "\\nvalues so far:" + "".join(" " + v_i for i <= k) + "\\n"
    target = steps[k] + "\\n"   for k < n,      "### " + answer + "\\n"   for k = n
(no equation history; only the values computed so far). Loss on the target tokens only.
"""
import re

PREFIX = "\nvalues so far:"


def step_value(step):
    """'<<600*30/100=180>>' -> '180' (text after the last '=' before '>>'); None if unparsable."""
    m = re.search(r"=([^=<>]*)>>\s*$", step.strip())
    return m.group(1).strip() if m else None


def state_text(question, values):
    return question + PREFIX + "".join(" " + v for v in values) + "\n"


def calls(question, steps, answer):
    """List of (input_text, target_text) for one example; None if a step value cannot be parsed."""
    out, vals = [], []
    for s in steps:
        v = step_value(s)
        if v is None:
            return None
        out.append((state_text(question, vals), s.strip() + "\n"))
        vals.append(v)
    out.append((state_text(question, vals), "### " + answer.strip() + "\n"))
    return out


def tc_text(question, steps, answer):
    """The Coconut-repo CoT training sequence (question + "\\n" + steps joined by "\\n" + "\\n### " + answer), for token
    matching."""
    return question + "\n" + "\n".join(steps) + "\n### " + answer
