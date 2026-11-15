import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "dashboard"))


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def dataset():
    import numpy as np
    p = ROOT / "models" / "dataset.npz"
    if not p.is_file():
        pytest.skip("models/dataset.npz not built; run ml/build_dataset.py")
    return np.load(p, allow_pickle=False)


@pytest.fixture(scope="session")
def harness():
    """A live esp32/host_harness.exe, or skip if no C++ compiler is available."""
    from cpp_bridge import Harness
    try:
        h = Harness()
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip("C++ harness unavailable: %s" % exc)
    yield h
    h.close()
