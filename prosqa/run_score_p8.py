"""Runs the registered PROTOCOL8 scorer prosqa/score_p8.py (sha256 8a151ebe..., unchanged) with one serialisation fix:
json.dump gets a default= that converts numpy scalars (the scorer's numpy bools are not JSON-serialisable). The
computation is untouched.  python prosqa/run_score_p8.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_dump = json.dump
json.dump = lambda obj, fp, **kw: _dump(obj, fp, default=lambda o: o.item() if isinstance(o, np.generic) else str(o), **kw)
import score_p8  # noqa: E402

score_p8.main()
