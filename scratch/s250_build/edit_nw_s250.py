"""§250 FRAME-DIV Stage 0 (FD_v1 §1): append frames 8-15 to boxes/nested_words.FRAMES from docs/research_framediv_2026-10-30/fd_v1_templates.json, right after the
add_s / mul_s alias line (an in-place list extend, so the aliases follow). FRAME_ALL / FRAME_TRAIN unchanged. Asserted once; the inserted literal is written with repr()."""
import json

T = json.load(open("docs/research_framediv_2026-10-30/fd_v1_templates.json", encoding="utf-8")); OPS6 = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v")
assert set(T) == {"TRAIN_NEW", "HELD"} and all(len(T[k][op]) == 4 for k in T for op in OPS6)
p = "boxes/nested_words.py"; s = open(p, encoding="utf-8").read()
anchor = 'FRAMES["mul_s"] = FRAMES["mul_v"]; FRAMES["add_s"] = FRAMES["add_v"]      # the partner {C} is an earlier spine quantity\'s name\n'
assert s.count(anchor) == 1 and "FRAMES_DIV_TRAIN" not in s
blk = ["# §250 FRAME-DIV (FD_v1 §1; fd_v1_templates.json): frames 8-11 (DIV training) and 12-15 (held out) appended IN PLACE per base op (add_s / mul_s follow via the",
       "# alias above); FRAME_TRAIN / FRAME_ALL unchanged, so every existing stream and set draws exactly as before (G1: all set shas byte-identical).",
       "FRAMES_DIV_TRAIN = {"] + ["    %r: %r," % (op, T["TRAIN_NEW"][op]) for op in OPS6] + ["}", "FRAMES_DIV_HELD = {"] + ["    %r: %r," % (op, T["HELD"][op]) for op in OPS6] + [
       "}", "for _op in %r:" % (OPS6,), "    FRAMES[_op].extend(FRAMES_DIV_TRAIN[_op] + FRAMES_DIV_HELD[_op])",
       "FRAME_DIV_TRAIN = tuple(range(8, 12)); FRAME_DIV_HELD = tuple(range(12, 16))   # §250"]
s = s.replace(anchor, anchor + "\n".join(blk) + "\n"); open(p, "w", encoding="utf-8", newline="\n").write(s); print("nested_words: frames 8-15 appended")
