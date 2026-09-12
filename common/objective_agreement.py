"""Single objective-comparison policy shared by all result producers."""
from __future__ import annotations

import numpy as np


OBJECTIVE_RTOL = 1e-6
OBJECTIVE_ATOL = 1e-6


def objective_agrees(value, reference):
    """Return NumPy's asymmetric isclose(value, reference) policy as bool."""
    return bool(np.isclose(value, reference, rtol=OBJECTIVE_RTOL, atol=OBJECTIVE_ATOL))
