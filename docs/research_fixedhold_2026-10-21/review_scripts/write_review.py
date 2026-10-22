import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "REVIEW_FH.md")
T = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_body.md"), encoding="utf-8").read()
open(P, "w", encoding="utf-8", newline="\n").write(T)
print("wrote", os.path.abspath(P), len(T))
