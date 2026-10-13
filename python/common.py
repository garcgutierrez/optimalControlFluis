import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from orrss import turprof_spectral

RT = 2000.0
NCHEB = 128
EDDYFRAC = 1.0
_PROFILE = {}

def profile(N=NCHEB, eddyfrac=EDDYFRAC):
    key = (N, eddyfrac)
    if key not in _PROFILE:
        _PROFILE[key] = turprof_spectral(N, RT, eddyfrac, 'tau')
    return _PROFILE[key]

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA, exist_ok=True)
