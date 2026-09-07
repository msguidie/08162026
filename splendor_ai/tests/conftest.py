import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running end-to-end tests (real server + worker)")
    config.addinivalue_line("markers", "selfplay_smoke: opt-in multi-minute trainer loop tests")
