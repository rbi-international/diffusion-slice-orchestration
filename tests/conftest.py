import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(1)

from dsorch.config import default_config  # noqa: E402
from dsorch.system import SystemModel  # noqa: E402
from dsorch.workload import controlled_trace  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return default_config()


@pytest.fixture(scope="session")
def sm(cfg):
    return SystemModel.from_config(cfg)


@pytest.fixture(scope="session")
def trace(sm):
    return controlled_trace(sm.slices, 260, 2.0, 3)
