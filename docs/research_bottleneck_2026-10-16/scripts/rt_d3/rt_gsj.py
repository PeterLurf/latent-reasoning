"""Red-team D3: per-slot accuracy under GATHER-SPLICE at several J (N2 / N3, pass 1; eval-only, CPU 3 threads, idle; no repo writes).
Question: can the PA-trained blocks copy at slots j >= 33 (the untrained / 'seam' rows) when their input is the plain carry N(V_0)[j]?
Compare per-j accuracy at j >= 33 under gsJ with the carry-input baseline at in-range slots far from the splice boundary (gs10 on T <= 27, j 14..24).
Usage: python rt_gsj.py <run> <out.json> <per_m>"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "D:/Latent")
sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research2/rt_d3")
import numpy as np
import torch
torch.set_num_threads(3)
import rt_pass1 as P
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE
import boxes.deltamem as DM


@torch.no_grad()
def main():
    run, fo, per_m = sys.argv[1], sys.argv[2], int(sys.argv[3])
    m, a = VE.load(run, "cpu"); assert m.mode == "pa"
    rng = np.random.default_rng(90212); items = []
    for mo in range(4, 15):
        for _ in range(per_m):
            r, _t = N.generate(rng, mo); items.append({"root": r, "syms": N.render(r), "m": mo})
    byT = defaultdict(list)
    for it in items:
        byT[len(it["syms"]) + 2].append(it)
    out = {}
    for J in (None, 10, 24, 29, 33):
        PJ = defaultdict(lambda: [0, 0])
        for T in sorted(byT):
            if J == 10 and T > 27:
                continue
            if J in (29, 33) and T < 36:
                continue
            for c0 in range(0, len(byT[T]), 24):
                part = byT[T][c0:c0 + 24]; x, pad, lens = VB.batch_x(part, "cpu")
                V0 = DM.gpt_lower(m.g, m.g.embed(x), VB.CORE[0]); NV0 = m.nrm(V0, V0.pow(2).mean(-1, keepdim=True).sqrt())
                w = torch.softmax(P.gather_w(m, V0, pad, T), -1); U = P.apply_w(w, NV0, m)
                if J is not None:
                    U = torch.cat([U[:, :J], NV0[:, J:]], 1)
                dec = m.rec_logits(P.core_pass1(m, U, V0, pad)).argmax(-1).numpy()
                for k_, it in enumerate(part):
                    E1 = VB.recode_targets(it["root"], 1)[1]
                    for j in range(T - 1):                          # the ' =' slot excluded
                        key = ("T<=27" if T <= 27 else "T36+") + "|j%02d" % min(j, 40)
                        PJ[key][0] += 1; PJ[key][1] += int(int(dec[k_, j]) == E1[j])
        out["gs%s" % J] = {k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(PJ.items())}
        print("J", J, "done", flush=True)
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", flush=True)


if __name__ == "__main__":
    main()
