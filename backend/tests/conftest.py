import sys
from pathlib import Path

import numpy as np
import pytest

# Make `app` importable when pytest is run from the repository root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import registry  # noqa: E402

registry.load_all()


@pytest.fixture(scope="session")
def plate():
    from tests.fixtures.synth import plate as _plate

    return _plate


@pytest.fixture(scope="session")
def clean(plate):
    """A clean, sharp, well-exposed, colour-neutral synthetic photograph."""
    return plate(0)


@pytest.fixture(scope="session")
def gray(clean):
    from app.core.colorspace import luma

    return np.clip(luma(clean), 0, 1).astype(np.float32)


@pytest.fixture(scope="session")
def shapes():
    """Binary-friendly geometry, for morphology and segmentation tests."""
    import cv2

    img = np.full((192, 192, 3), 0.94, np.float32)
    cv2.circle(img, (60, 60), 28, (0.08, 0.08, 0.1), -1)
    cv2.rectangle(img, (110, 95), (165, 155), (0.08, 0.08, 0.1), -1)
    cv2.circle(img, (55, 140), 22, (0.08, 0.08, 0.1), -1)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.5), 0, 1).astype(np.float32)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
