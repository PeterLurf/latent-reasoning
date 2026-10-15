"""OC of the rev-2 two-part rule at fsq120 prospective sizes (synthetic binormal scores; seedselect.py's own tests).
Writes theory/seedselect/oc_prospective.json.
    python theory/seedselect/oc_prospective.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oc  # noqa: E402  (sets Idle priority on import of seedselect)


def main():
    rng = np.random.default_rng(11); out = {}
    for strata in ([(8, 8)], [(12, 12)]):
        oc.STRATA = strata
        for name, (af, an, rho) in (("confound (0.8, null 0.8, rho 0.7)", (0.8, 0.8, 0.7)), ("C 0.8, null 0.5", (0.8, 0.5, 0.3)),
                                     ("C 0.8, null 0.65", (0.8, 0.65, 0.5)), ("C 0.9, null 0.65", (0.9, 0.65, 0.5))):
            k = 0; n = 150
            for _ in range(n):
                f, _, b = oc.one(rng, af, an, rho, 300); k += (f and b)
            out["%s | %s" % (strata, name)] = k / n
            print(strata, name, "P(USEFUL rev2) = %.3f" % (k / n), flush=True)
    json.dump(out, open(os.path.join(HERE, "oc_prospective.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
