"""Replace the line '<marker>' (a PENDING section header) in REDTEAM_LANG.md with the contents of <textfile> (utf-8, LF).
    python -B splice.py "<exact marker line>" <textfile>"""
import sys
p = "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research7/REDTEAM_LANG.md"
s = open(p, encoding="utf-8").read(); marker = sys.argv[1] + "\n"; new = open(sys.argv[2], encoding="utf-8").read()
assert s.count(marker) == 1, "marker not found exactly once"
s = s.replace(marker, new.rstrip("\n") + "\n\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("spliced", len(new), "chars")
