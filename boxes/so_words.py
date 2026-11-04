"""SHUFFLED-OBJECTS (SO) generator, THEORY §271 N1 (N1_v1 §3.1-3.5, §3.7, §10 item 1). Modelled on boxes/ow_words.py (make / Stream / test_set / solve / set_sha).
Item law (§3.1): n in {3, 4, 5} people from NAMES_SO (nested_words.NAMES minus Eve / Bob: both are BBH names), family f in FAMILIES (BBH's five), n distinct objects,
an initial bijection h_0 (person i holds objects[i]), the asked object o*, m swaps (a_t, b_t) uniform over unordered pairs. S_0 = h_0^-1(o*) (o*'s first holder),
S_t = tau_t(S_{t-1}) with tau_t the transposition (a_t b_t); answer = S_m; values = [S_0 .. S_m]; k_touch = #{t: S_t != S_{t-1}}; d = m.
Render (§3.2): intro + setup (frame-dependent wording; the holding clause is fixed per family) + the swaps + the question "At the end, who <verb> <o*>?".
  NUM: "Swap <L>: <a> and <b> <swap verb>." with consecutive labels o+1 .. o+m; o = 0 for test cells, o ~ U{0 .. 12 - m} for training items (offset=True).
  ORD: "First, ... Then, ... Finally, ..." (BBH-shaped, no labels; L(s) = s).
  Frames: 6 per family (setup + swap wording); 0-3 train (FRAME_TRAIN), 4-5 TRANSFER.
Cues (§3.3; the text of a pass input after the item text, before ' The answer is:'): SI " After swap <L(min(t, m))>, who <verb> <o*>?"; EN " After <a_t> and <b_t>
trade, who <verb> <o*>?"; GENERIC " Who <verb> <o*> now?". Aux R_step[t] = "After swap <L(t)>, <S_t> <verb> <o*>." (vidx = S_t's name token, tok_so).
solve(text): an independent regex parser of the rendered text (people, holdings, swaps in text order, the question) -> the final holder; asserted == answer by callers.
SWAP-MID pairs (§3.7): swap_pairs(n, m, t, seed, mode) -> 1,000 (recipient, donor, cf) with the donor = the same people / objects / h_0 / o* / frame / offset and an
independently redrawn swap sequence of length m; cf = the recipient's swaps t+1 .. m applied to the donor's S_t; kept iff cf not in {own final, donor final, donor S_t}.
Seeds (29xxx family; unused elsewhere in boxes / theory / scratch *.py, grep 2026-11-04): SEED_TRAIN 29100, SEED_TEST 29110, SEED_TRANSFER 29120, SEED_OFFSET 29130,
SEED_PAIRS 29140 (+ cell), SEED_PERM 29150, SEED_LOOKUP 29160.
    python -B -m boxes.so_words --selfcheck"""
import hashlib
import re
import sys

import numpy as np

from boxes.nested_words import NAMES

NAMES_SO = [x for x in NAMES if x not in ("Eve", "Bob")]                   # 38 single-token names (BBH's Eve / Bob excluded, F12a)
SEED_TRAIN, SEED_TEST, SEED_TRANSFER, SEED_OFFSET, SEED_PAIRS, SEED_PERM, SEED_LOOKUP = 29100, 29110, 29120, 29130, 29140, 29150, 29160
FRAME_TRAIN, FRAME_TRANSFER = (0, 1, 2, 3), (4, 5)
MAX_LABEL = 12
NS, MS, MODES = (3, 4, 5), range(1, 13), ("num", "ord")
TEST_N = 500; TEST_BIG = {(5, 8, "num"): 1500, (3, 8, "num"): 1500}       # F7: the M5 / X2 strata enlarged

# family -> objects, holding verb, the object phrase (from the object word), the swap noun, the activity (intro)
FAMILIES = {
    "balls": {"objects": ["red", "blue", "green", "yellow", "white", "black", "pink", "brown"], "verb": "has", "phrase": "the {} ball", "noun": "balls",
              "intro": ["{P} are playing a game.", "{P} meet to play catch.", "{P} stand in a circle with some balls.", "{P} start a ball game.",
                        "{P} gather on the lawn for a game.", "{P} begin a round of passing."],
              "setup": ["At the start each holds one ball:", "To begin, every player holds a ball:", "Each of them starts out with one ball:",
                        "First of all, the balls are handed out:", "Every player begins with a single ball:", "The balls are shared out at the start:"],
              "swap": ["trade balls", "swap balls", "exchange balls", "switch balls", "pass each other their balls", "give each other their balls"]},
    "presents": {"objects": ["red", "blue", "green", "yellow", "white", "black", "pink", "brown"], "verb": "has", "phrase": "the {} present", "noun": "presents",
                 "intro": ["{P} are at a party.", "{P} meet for a gift swap.", "{P} celebrate a holiday together.", "{P} hold a small party.",
                           "{P} get together to share gifts.", "{P} come to a birthday dinner."],
                 "setup": ["At the start each holds one present:", "To begin, everyone holds a present:", "Each of them starts out with one present:",
                           "First of all, the presents are handed out:", "Every guest begins with a single present:", "The presents are shared out at the start:"],
                 "swap": ["trade presents", "swap presents", "exchange presents", "switch presents", "pass each other their presents", "give each other their presents"]},
    "books": {"objects": ["Hamlet", "Dracula", "Ulysses", "Walden", "Ivanhoe", "Beloved", "Persuasion", "Frankenstein"], "verb": "has", "phrase": "{}", "noun": "books",
              "intro": ["{P} are in a book club.", "{P} meet to share books.", "{P} sit in the library together.", "{P} start a reading circle.",
                        "{P} gather at the book shop.", "{P} begin a book exchange."],
              "setup": ["At the start each holds one book:", "To begin, every reader holds a book:", "Each of them starts out with one book:",
                        "First of all, the books are handed out:", "Every reader begins with a single book:", "The books are shared out at the start:"],
              "swap": ["trade books", "swap books", "exchange books", "switch books", "pass each other their books", "give each other their books"]},
    "partners": {"objects": ["Lola", "Rosa", "Nina", "Olga", "Tina", "Vera", "Gina", "Lena"], "verb": "is dancing with", "phrase": "{}", "noun": "partners",
                 "intro": ["{P} are at a dance.", "{P} meet at a dance hall.", "{P} go to a square dance.", "{P} join a dance class.",
                           "{P} attend a ball.", "{P} take part in a dance night."],
                 "setup": ["At the start each has one partner:", "To begin, every dancer has a partner:", "Each of them starts out with one partner:",
                           "First of all, the partners are paired up:", "Every dancer begins with a single partner:", "The partners are assigned at the start:"],
                 "swap": ["trade partners", "swap partners", "exchange partners", "switch partners", "hand each other their partners", "give each other their partners"]},
    "positions": {"objects": ["goalkeeper", "striker", "winger", "defender", "midfielder", "sweeper", "fullback", "centre"], "verb": "is playing", "phrase": "{}",
                  "noun": "positions",
                  "intro": ["{P} are on a soccer team.", "{P} meet for a soccer match.", "{P} play on the same team.", "{P} join a football side.",
                            "{P} line up for a match.", "{P} take the field together."],
                  "setup": ["At the start each plays one position:", "To begin, every player has a position:", "Each of them starts out in one position:",
                            "First of all, the positions are assigned:", "Every player begins in a single position:", "The positions are shared out at the start:"],
                  "swap": ["trade positions", "swap positions", "exchange positions", "switch positions", "take each other's positions", "give each other their positions"]},
}
FAMS = tuple(FAMILIES)
ORD_LEAD = ("First,", "Then,", "Finally,")


def people_str(p):
    return ", ".join(p[:-1]) + " and " + p[-1]


def obj(f, o):
    return FAMILIES[f]["phrase"].format(o)


def hold_clause(f, person, o):
    return "%s %s %s" % (person, FAMILIES[f]["verb"], obj(f, o))


def label(it, s):
    """L(s) = o + s (NUM) or s (ORD)."""
    return it["offset"] + s if it["mode"] == "num" else s


def question(it):
    return "At the end, who %s %s?" % (FAMILIES[it["family"]]["verb"], obj(it["family"], it["ostar"]))


def cue_si(it, t):
    return " After swap %d, who %s %s?" % (label(it, min(t, it["m"])), FAMILIES[it["family"]]["verb"], obj(it["family"], it["ostar"]))


def cue_en(it, t):
    a, b = it["swaps"][min(t, it["m"]) - 1]
    return " After %s and %s trade, who %s %s?" % (a, b, FAMILIES[it["family"]]["verb"], obj(it["family"], it["ostar"]))


def cue_generic(it, t):
    return " Who %s %s now?" % (FAMILIES[it["family"]]["verb"], obj(it["family"], it["ostar"]))


def r_step(it, t):
    return "After swap %d, %s %s %s." % (label(it, t), it["values"][t], FAMILIES[it["family"]]["verb"], obj(it["family"], it["ostar"]))


def swap_sentences(it, order=None):
    """the swap sentences in text order (order = a permutation of 0..m-1 for PERM-LABEL; NUM labels stay the true index)."""
    F = FAMILIES[it["family"]]; v = F["swap"][it["frame"]]; m = it["m"]; idx = list(range(m)) if order is None else list(order); out = []
    for k, s in enumerate(idx):
        a, b = it["swaps"][s]
        if it["mode"] == "num":
            out.append("Swap %d: %s and %s %s." % (label(it, s + 1), a, b, v))
        else:
            lead = ORD_LEAD[0] if k == 0 else (ORD_LEAD[2] if k == m - 1 else ORD_LEAD[1])
            out.append("%s %s and %s %s." % (lead, a, b, v))
    return out


def render(it, order=None):
    F = FAMILIES[it["family"]]; fr = it["frame"]; P = it["people"]
    hold = ", ".join(hold_clause(it["family"], p, o) for p, o in zip(P[:-1], it["objects"][:-1])) + " and " + hold_clause(it["family"], P[-1], it["objects"][-1])
    return " ".join([F["intro"][fr].format(P=people_str(P)), F["setup"][fr] + " " + hold + "."] + swap_sentences(it, order) + [question(it)])


def finish(it):
    """values / answer / k_touch / text / R_step / sig from people, objects, ostar, swaps, mode, offset, frame."""
    s = it["people"][it["objects"].index(it["ostar"])]; vals = [s]
    for a, b in it["swaps"]:
        s = b if s == a else (a if s == b else s); vals.append(s)
    it["values"] = vals; it["answer"] = vals[-1]; it["d"] = it["m"]; it["k_touch"] = sum(vals[t] != vals[t - 1] for t in range(1, len(vals)))
    it["moved"] = int(vals[-1] != vals[0] and it["k_touch"] >= 2); it["text"] = render(it); it["question"] = question(it)
    it["R_step"] = [r_step(it, t) for t in range(1, it["m"] + 1)]; it["sig"] = hashlib.sha1(it["text"].encode()).hexdigest(); return it


def draw_swaps(rng, n, m, people):
    out = []
    for _ in range(m):
        i, j = rng.choice(n, size=2, replace=False); out.append((people[int(min(i, j))], people[int(max(i, j))]))
    return out


def make(rng, n, m, mode, frames=FRAME_TRAIN, offset=False, names=NAMES_SO):
    """one item; draw order: family, frame, people, objects (h_0 = person i holds objects[i]), o*, swaps, offset (offset=True: U{0 .. 12 - m}; NUM only)."""
    assert n in NS and 1 <= m <= MAX_LABEL and mode in MODES
    f = FAMS[int(rng.integers(0, len(FAMS)))]; fr = int(frames[int(rng.integers(0, len(frames)))])
    people = [names[i] for i in rng.choice(len(names), size=n, replace=False)]
    objs = [FAMILIES[f]["objects"][i] for i in rng.choice(len(FAMILIES[f]["objects"]), size=n, replace=False)]
    ostar = objs[int(rng.integers(0, n))]; swaps = draw_swaps(rng, n, m, people)
    o = int(rng.integers(0, MAX_LABEL - m + 1)) if (offset and mode == "num") else 0
    return finish({"n": n, "m": m, "mode": mode, "family": f, "frame": fr, "people": people, "objects": objs, "ostar": ostar, "swaps": swaps, "offset": o})


def with_offset(it, o):
    """the item re-rendered with NUM offset o (OFFSET cells)."""
    assert it["mode"] == "num" and 0 <= o <= MAX_LABEL - it["m"]
    return finish({k: it[k] for k in ("n", "m", "mode", "family", "frame", "people", "objects", "ostar", "swaps")} | {"offset": o})


class Stream:
    """training / set stream: per draw(m) n ~ U(ns), mode ~ U(modes), then make(); items whose sig is in excl are skipped."""
    def __init__(self, seed, excl=(), frames=FRAME_TRAIN, ns=NS, modes=MODES, offset=True, names=NAMES_SO):
        self.rng = np.random.default_rng(seed); self.excl = set(excl); self.frames = frames; self.ns = tuple(ns); self.modes = tuple(modes)
        self.offset = offset; self.names = list(names); self.stats = {}

    def draw(self, m, n=None, mode=None):
        while True:
            n_ = n if n is not None else self.ns[int(self.rng.integers(0, len(self.ns)))]
            md = mode if mode is not None else self.modes[int(self.rng.integers(0, len(self.modes)))]
            it = make(self.rng, n_, m, md, self.frames, self.offset, self.names)
            if it["sig"] not in self.excl:
                return it
            self.stats["excl_skips"] = self.stats.get("excl_skips", 0) + 1


def test_sets():
    """-> {"test": {(n, m, mode): [items]}, "transfer": {m: [items]} (n 5 NUM, frames 4-5, m 1-8), "offset": [n 5 m 4 NUM, o = 8]} (labels from 1 unless OFFSET)."""
    s = Stream(SEED_TEST, offset=False); test = {}
    for n in NS:
        for m in MS:
            for md in MODES:
                test[(n, m, md)] = [s.draw(m, n, md) for _ in range(TEST_BIG.get((n, m, md), TEST_N))]
    s = Stream(SEED_TRANSFER, frames=FRAME_TRANSFER, offset=False); tr = {m: [s.draw(m, 5, "num") for _ in range(TEST_N)] for m in range(1, 9)}
    s = Stream(SEED_OFFSET, offset=False); off = [with_offset(s.draw(4, 5, "num"), 8) for _ in range(TEST_N)]
    return {"test": test, "transfer": tr, "offset": off}


def excl_sigs(sets):
    return ({it["sig"] for v in sets["test"].values() for it in v} | {it["sig"] for v in sets["transfer"].values() for it in v} | {it["sig"] for it in sets["offset"]})


def redraw_swaps(rng, it):
    """F2 donor: the same people / objects / h_0 / o* / family / frame / mode / offset, an independently redrawn swap sequence of length m."""
    return finish({k: it[k] for k in ("n", "m", "mode", "family", "frame", "people", "objects", "ostar", "offset")} | {"swaps": draw_swaps(rng, it["n"], it["m"], it["people"])})


def apply_swaps(s, swaps):
    for a, b in swaps:
        s = b if s == a else (a if s == b else s)
    return s


def swap_pairs(n, m, t, seed=SEED_PAIRS, mode="num", cap=1000, tries=4000):
    """-> (pairs [{rec, don, t, cf}], n_tries): recipient from Stream(seed + 1000 n + 10 m + t), donor via redraw_swaps (default_rng(seed + 50000 + ...));
    keep iff cf = rec's swaps t+1 .. m applied to don's S_t is not in {rec's final, don's final, don's S_t}; token lengths equal (asserted by the eval)."""
    key = 1000 * n + 10 * m + t; s = Stream(seed + key, offset=False); r = np.random.default_rng(seed + 50000 + key); out = []; k = 0
    while len(out) < cap and k < tries:
        rec = s.draw(m, n, mode); don = redraw_swaps(r, rec); k += 1
        cf = apply_swaps(don["values"][t], rec["swaps"][t:])
        if cf not in (rec["answer"], don["answer"], don["values"][t]):
            out.append({"rec": rec, "don": don, "t": t, "cf": cf})
    return out, k


_SWAP_NUM = re.compile(r"Swap (\d+): (\w+) and (\w+) ")
_SWAP_ORD = re.compile(r"(?:First|Then|Finally), (\w+) and (\w+) ")


def solve(text):
    """independent regex parser: the holding clauses after the setup ':' (person <verb> <object phrase>), the swap sentences in text order (NUM sorted by label),
    the question's object -> its final holder (None if unparseable)."""
    m_ = re.search(r":\s(.*?)\.\s(?:Swap \d+:|First,)", text)
    q_ = re.search(r"At the end, who (has|is dancing with|is playing) (.+)\?$", text)
    if not m_ or not q_:
        return None
    verb, o = q_.group(1), q_.group(2); hold = {}
    for cl in re.split(r", | and ", m_.group(1)):
        mm = re.match(r"(\w+) " + re.escape(verb) + r" (.+)$", cl.strip())
        if not mm:
            return None
        hold[mm.group(2)] = mm.group(1)
    if o not in hold:
        return None
    num = _SWAP_NUM.findall(text)
    sw = [(a, b) for _, a, b in sorted(num, key=lambda x: int(x[0]))] if num else _SWAP_ORD.findall(text)
    return apply_swaps(hold[o], sw) if sw else None


FIT_SEED, N_FIT = 27110, 200_000                                            # Note 271-2: pinned floor fit (docs/research_n1_2026-11-04/n1_oc.py)
_FIT = {}


def _fit_draw(R, n, m):
    """n1_oc.draw: S_0 uniform, m swaps uniform over unordered pairs (ordered draw a, b != a) -> (s0, swaps, S) in person indices."""
    s0 = R.randrange(n); sw = []
    for _ in range(m):
        a = R.randrange(n); b = R.randrange(n - 1); b += b >= a; sw.append((a, b))
    S = [s0]
    for a, b in sw:
        x = S[-1]; S.append(b if x == a else a if x == b else x)
    return s0, sw, S


def _moved(S):
    k = sum(S[i] != S[i - 1] for i in range(1, len(S))); return S[-1] != S[0] and k >= 2


def _norm(s0, sw):
    mp = {s0: 0}
    for a, b in sw:
        for x in (a, b):
            if x not in mp:
                mp[x] = len(mp)
    return mp, tuple(tuple(sorted((mp[a], mp[b]))) for a, b in sw)


def _map(counters):
    return {k: min(a for a, v in c.items() if v == max(c.values())) for k, c in counters.items()}


def fit_tables(n, m):
    """bag-of-swaps / last-2 MAP tables for cell (n, m): random.Random(27110), 200,000 draws, MOVED subset, modal answer (normalised index), ties -> smallest."""
    import collections
    import random
    if (n, m) not in _FIT:
        R = random.Random(FIT_SEED); bag = collections.defaultdict(collections.Counter); l2 = collections.defaultdict(collections.Counter)
        for _ in range(N_FIT):
            s0, sw, S = _fit_draw(R, n, m)
            if _moved(S):
                mp, key = _norm(s0, sw); ans = mp[S[-1]]; bag[tuple(sorted(key))][ans] += 1; l2[key[-2:]][ans] += 1
        _FIT[(n, m)] = (_map(bag), _map(l2))
    return _FIT[(n, m)]


def floors(items):
    """Note 271-2 / N1_v1 §6.1 M-7: shortcut floors on the cell's own MOVED items (one cell: same n, m). -> {f, arg, by_shortcut, n_moved, floor_hold, unseen}.
    Shortcuts: non-initial uniform 1/(n-1); bag-of-swaps MAP; last-2-swaps MAP (unseen key -> 1/(n-1) credit); last-2 only (S_0 through the last 2 swaps); the partner
    of the last swap involving S_0; the most frequent name (S_0 excluded; ties shared); S_4 truncation (m > 4 only). f = the max; floor_hold = P(S_m = S_0) on all items."""
    import collections
    n, m = items[0]["n"], items[0]["m"]; assert all(it["n"] == n and it["m"] == m for it in items)
    hold = sum(it["values"][-1] == it["values"][0] for it in items) / len(items)
    mv = [it for it in items if it["moved"]]
    if not mv:
        return {"f": None, "arg": None, "by_shortcut": {}, "n_moved": 0, "floor_hold": hold, "unseen": {}}
    bagm, l2m = fit_tables(n, m); u = 1.0 / (n - 1); acc = collections.Counter(); uns = collections.Counter()
    for it in mv:
        P = it["people"]; s0 = P.index(it["values"][0]); sw = [(P.index(a), P.index(b)) for a, b in it["swaps"]]; S = [P.index(v) for v in it["values"]]; fin = S[-1]
        mp, key = _norm(s0, sw); ans = mp[fin]
        acc["non_initial_unif"] += u
        if m > 4:
            acc["S4_trunc"] += S[4] == fin
        acc["last2_only"] += apply_swaps(s0, sw[-2:]) == fin
        pr = [b if a == s0 else a for a, b in sw if s0 in (a, b)]; acc["lastS0partner"] += pr[-1] == fin
        c = collections.Counter(x for ab in sw for x in ab); c.pop(s0, None); best = max(c.values()); cands = [x for x in c if c[x] == best]
        acc["mostfreq"] += (fin in cands) / len(cands)
        bk = tuple(sorted(key))
        if bk in bagm:
            acc["bag_MAP"] += bagm[bk] == ans
        else:
            acc["bag_MAP"] += u; uns["bag"] += 1
        k2 = key[-2:]
        if k2 in l2m:
            acc["last2_MAP"] += l2m[k2] == ans
        else:
            acc["last2_MAP"] += u; uns["last2"] += 1
    r = {k: v / len(mv) for k, v in acc.items()}; arg = max(r, key=lambda k: (r[k], k))
    return {"f": r[arg], "arg": arg, "by_shortcut": r, "n_moved": len(mv), "floor_hold": hold, "unseen": {k: v / len(mv) for k, v in uns.items()}}


def set_sha(items):
    return hashlib.sha256("\n".join(x["text"] for x in items).encode()).hexdigest()


if __name__ == "__main__" and "--selfcheck" in sys.argv:
    r = np.random.default_rng(0); bad = 0
    for k in range(20000):
        it = make(r, int(r.choice(NS)), int(r.integers(1, 13)), MODES[k % 2], FRAME_TRAIN + FRAME_TRANSFER, offset=True)
        bad += solve(it["text"]) != it["answer"]
    print("solve mismatches", bad, "of 20000"); it = make(np.random.default_rng(1), 5, 4, "num", offset=True); print(it["text"]); print(cue_si(it, 2), it["R_step"])
    it = make(np.random.default_rng(2), 3, 3, "ord"); print(it["text"]); print(cue_en(it, 2), cue_generic(it, 1))
    sys.exit(1 if bad else 0)
