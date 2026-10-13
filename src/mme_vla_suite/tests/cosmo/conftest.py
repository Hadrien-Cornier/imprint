import os
import sys

import numpy as np
import pytest

# Ensure the mme_vla_suite package root is first on sys.path so "tests.cosmo"
# resolves to THIS package and not the draccus "tests" pkg in site-packages,
# regardless of the CWD pytest is invoked from.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
