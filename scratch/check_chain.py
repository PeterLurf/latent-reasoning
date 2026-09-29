"""Show how chain.py will split each running chain's command line."""
import shlex
import subprocess

out = subprocess.run(["powershell", "-NoProfile", "-Command",
                      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
                     capture_output=True, text=True).stdout
for line in out.splitlines():
    if "chain.py" not in line:
        continue
    argv = shlex.split(line, posix=False)
    argv = argv[argv.index(next(a for a in argv if a.endswith("chain.py"))):]
    rest = argv[argv.index("--") + 1:]
    cmds, cur = [], []
    for t in rest:
        if t == "::":
            cmds.append(cur); cur = []
        else:
            cur.append(t)
    cmds.append(cur)
    print("chain", argv[1], argv[2])
    for c in cmds:
        print("   ", " ".join(c)[:200])
