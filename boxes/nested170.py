"""THEORY §170 NESTED (eed3f25a; scorer theory/score_s170.py): Feng et al. 2305.15408's Arithmetic(n, 11), Alg. 1 - data, rendering, evaluation, CoT.
Generator: t ~ U(Z11) is the expression; n times: a uniform random number position, an operator uniformly from + - * /, and (t1, t2) uniformly among the
pairs with o(t1, t2) = that number (t2 != 0 for /; / = multiplication by the modular inverse); the number becomes "t1 o t2". Brackets exactly where
precedence or left associativity needs them (a child of lower precedence, or a right child of equal precedence). Spaced tokens: " ( 3 + 10 ) * 4 =".
TEXT-COT (Feng's format): each step reduces the leftmost operation evaluable under precedence and brackets (a node whose operands are both numbers;
re-rendering drops a bracket around the single number in the same step): "e1 = e2 = ... = t"."""
import os
import re

import numpy as np
import tiktoken

P = 11
OPS = ["+", "-", "*", "/"]
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
ENC = tiktoken.get_encoding("gpt2")
VOCAB = [" %d" % v for v in range(P)] + [" " + o for o in OPS] + [" (", " )", " =", " ."]
TOK = {w: ENC.encode(w) for w in VOCAB}
assert all(len(t) == 1 for t in TOK.values()) and all(ENC.decode(t) == w for w, t in TOK.items()), "§170: every symbol must be one GPT-2 token"
TID = {w: t[0] for w, t in TOK.items()}; NUM_IDS = np.array([TID[" %d" % v] for v in range(P)])
INV = {v: pow(v, P - 2, P) for v in range(1, P)}


def apply(o, a, b):
    return (a + b) % P if o == "+" else (a - b) % P if o == "-" else (a * b) % P if o == "*" else (a * INV[b]) % P


PAIRS = {(o, v): [(a, b) for a in range(P) for b in range(P) if not (o == "/" and b == 0) and apply(o, a, b) == v] for o in OPS for v in range(P)}


class Node:
    __slots__ = ("op", "l", "r", "v")

    def __init__(self, v=None, op=None, l=None, r=None):
        self.v, self.op, self.l, self.r = v, op, l, r


def generate(rng, n):
    """Alg. 1: returns (root, t)."""
    t = int(rng.integers(0, P)); root = Node(v=t); leaves = [root]
    for _ in range(n):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = OPS[int(rng.integers(0, 4))]; prs = PAIRS[(o, leaf.v)]; a, b = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, Node(v=a), Node(v=b), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root, t


def render(nd, parent=None, right=False):
    """minimal-bracket rendering -> list of symbols (" 3", " +", " (" ...)."""
    if nd.op is None:
        return [" %d" % nd.v]
    s = render(nd.l, nd.op, False) + [" " + nd.op] + render(nd.r, nd.op, True)
    if parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent])):
        s = [" ("] + s + [" )"]
    return s


def value(nd):
    return nd.v if nd.op is None else apply(nd.op, value(nd.l), value(nd.r))


def depth(syms):
    d = m = 0
    for s in syms:
        d += s == " ("; d -= s == " )"; m = max(m, d)
    return m


def eval_tokens(syms):
    """an INDEPENDENT Z11 evaluator: shunting-yard over the rendered symbols (precedence, left associativity, brackets)."""
    out, st = [], []
    for s in syms:
        w = s.strip()
        if w.isdigit():
            out.append(int(w))
        elif w in PREC:
            while st and st[-1] in PREC and PREC[st[-1]] >= PREC[w]:
                b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
            st.append(w)
        elif w == "(":
            st.append(w)
        elif w == ")":
            while st[-1] != "(":
                b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
            st.pop()
    while st:
        b, a = out.pop(), out.pop(); out.append(apply(st.pop(), a, b))
    assert len(out) == 1
    return out[0]


def _leftmost_ready(nd, path=()):
    """the leftmost node (in rendering order) whose operands are both numbers."""
    if nd.op is None:
        return None
    r = _leftmost_ready(nd.l, path + ("l",))
    if r is not None:
        return r
    if nd.l.op is None and nd.r.op is None:
        return path
    return _leftmost_ready(nd.r, path + ("r",))


def _copy(nd):
    return Node(v=nd.v) if nd.op is None else Node(op=nd.op, l=_copy(nd.l), r=_copy(nd.r))


def cot_steps(root):
    """Feng's CoT: the list of expressions e1 (= the input) .. t, each the previous with its leftmost evaluable operation reduced."""
    nd = _copy(root); steps = [render(nd)]
    while nd.op is not None:
        path = _leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.v = apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None; steps.append(render(nd))
    return steps


def ids(syms):
    return [TID[s] for s in syms]


def self_check(n_items=100_000, seed=17000):
    """the generator gate: the independent evaluator re-derives t on n_items samples (n ~ U{1..16}); returns (ok share, n)."""
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n_items):
        root, t = generate(rng, int(rng.integers(1, 17))); syms = render(root); ok += (eval_tokens(syms) == t) and (value(root) == t)
        if _ < 2000:                                                        # the CoT ends at t and every step evaluates to t
            st = cot_steps(root); assert st[-1] == [" %d" % t] and all(eval_tokens(s) == t for s in st), "cot"
    return ok / n_items, n_items
