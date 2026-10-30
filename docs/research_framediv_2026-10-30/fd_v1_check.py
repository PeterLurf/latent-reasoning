import sys, os, re, json
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, "D:/Latent")
import boxes.nested_words as NW
import boxes.ow_words as OW

TRAIN_NEW = {  # frames 8-11 (DIV training), red-team wording fixes applied
"add_c": ["{B} ends up with {c} {u} more than {A}.", "With {c} more {u}, {A} would have as many as {B}.", "{B} possesses {A}'s number of {u} increased by {c}.", "There are {c} more {u} with {B} than with {A}."],
"sub_c": ["{B} ends up with {c} {u} fewer than {A}.", "With {c} fewer {u}, {A} would have as many as {B}.", "{B} possesses {A}'s number of {u} decreased by {c}.", "There are {c} fewer {u} with {B} than with {A}."],
"mul_c": ["{B} ends up with {c} times {A}'s {u}.", "Scale {A}'s {u} by {c} and you have {B}'s {u}.", "{B} possesses {A}'s number of {u} multiplied by {c}.", "There are {c} times as many {u} with {B} as with {A}."],
"add_v": ["{B} ends up with {A}'s {u} and {C}'s {u} put together.", "Pool the {u} of {A} and {C} and you have as many as {B}.", "{B} possesses the combined number of {u} of {A} and {C}.", "There are as many {u} with {B} as with {A} and {C} jointly."],
"sub_v": ["{B} ends up with {A}'s {u} once {C}'s {u} are subtracted from them.", "Subtract {C}'s {u} from {A}'s and you have as many as {B}.", "{B} possesses the number of {u} left when {C}'s are taken from {A}'s.", "There are as many {u} with {B} as {A} has in excess of {C}."],
"mul_v": ["{B} ends up with {A}'s {u} times {C}'s {u}.", "Multiply the {u} of {A} and of {C} and you have as many as {B}.", "{B} possesses the product of the {u} counts of {A} and {C}.", "There are as many {u} with {B} as {A}'s {u} multiplied by {C}'s."],
}
HELD = {  # frames 12-15 (held out, binding): pseudo-cleft / red-team-accepted 13 / passive 'is reached by' / 'stands at'
"add_c": ["What {B} has is {c} {u} past {A}'s tally.", "{A} trails {B} by {c} {u}.", "{B}'s count is reached by adding {c} {u} to {A}'s.", "In {u}, {B} stands {c} above {A}."],
"sub_c": ["What {B} has is {c} {u} short of {A}'s tally.", "{B} trails {A} by {c} {u}.", "{B}'s count is reached by taking {c} {u} off {A}'s.", "In {u}, {B} stands {c} below {A}."],
"mul_c": ["What {B} has is {A}'s tally of {u}, {c} times over.", "{B}'s pile of {u} is {c}-fold {A}'s.", "{B}'s count is reached by counting {A}'s {u} {c} times.", "In {u}, {B} stands at {c} times {A}."],
"add_v": ["What {B} has is {A}'s tally of {u} joined with {C}'s.", "{A} and {C} between them match {B}'s {u}.", "{B}'s count is reached by counting {A}'s {u} and then {C}'s.", "In {u}, {B} stands at the total of {A} with {C}."],
"sub_v": ["What {B} has is {A}'s tally of {u} once {C}'s is removed.", "{A} outnumbers {C} in {u} by exactly {B}'s count.", "{B}'s count is reached by taking {C}'s {u} off {A}'s.", "In {u}, {B} stands at {A} reduced by {C}."],
"mul_v": ["What {B} has is {A}'s tally of {u}, once for each of {C}'s {u}.", "{A}'s {u}, repeated as often as {C} has {u}, give {B}'s {u}.", "{B}'s count is reached by counting {A}'s {u} once per {u} of {C}.", "In {u}, {B} stands at {A} scaled by {C}."],
}
OLD = {k: list(NW.FRAMES[k]) for k in TRAIN_NEW}


def mask(t):
    return re.sub(r"\{(\w)\}", lambda m: "X" if m.group(1) in "ABC" else ("N" if m.group(1) == "c" else "U"), t).lower().replace(".", " .").replace(",", " ,").replace("'s", " 's").split()


def grams(t, n=3):
    w = mask(t); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


worst = []
for op in TRAIN_NEW:
    refs = OLD[op][:8] + TRAIN_NEW[op]          # frames 0-7 and DIV-train 8-11
    for k, t in enumerate(HELD[op]):
        g = grams(t); m = max((jac(g, grams(s)), i) for i, s in enumerate(refs))
        worst.append((m[0], op, 12 + k, m[1], t))
        print(f"{op} {12 + k}: max J={m[0]:.2f} vs frame {m[1]} | {t}")
print("max over held-out:", max(w[0] for w in worst), "gate <= 0.20:", all(w[0] <= 0.20 for w in worst))
# solve / render check with extended FRAMES (in memory only)
for op in TRAIN_NEW:
    NW.FRAMES[op].extend(TRAIN_NEW[op] + HELD[op])
bad = 0; tot = 0
for frs in ((8, 9, 10, 11), (12, 13, 14, 15), tuple(range(8))):
    for law in (1, 2):
        s = OW.Stream(777 + law, frames=frs, order_mode="topo", mul_s_p=law)
        for d in range(1, 7):
            for _ in range(150):
                it = s.draw(d); tot += 1
                if OW.solve(it["text"]) != it["answer"]:
                    bad += 1
                    if bad <= 5:
                        print("MISMATCH", frs, it["text"][:220])
print("solve mismatches", bad, "/", tot)
json.dump({"TRAIN_NEW": TRAIN_NEW, "HELD": HELD}, open("C:/Users/peter/AppData/Local/Temp/claude/D--/6440ebeb-beed-5a82-930f-3ec2470822c0/scratchpad/fd_v1_templates.json", "w"), indent=1)
