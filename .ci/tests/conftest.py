import sys
from pathlib import Path

# Allow `import run` and `import build` directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from utils import normalize_config


@pytest.fixture
def minimal_config():
    """Minimal platform-centric config, normalized to flat format."""
    raw = {
        "repo": {
            "url": "https://github.com/InfiniTensor/InfiniOps.git",
            "branch": "master",
        },
        "platforms": {
            "nvidia": {
                "image": {
                    "dockerfile": ".ci/images/nvidia/",
                    "build_args": {"BASE_IMAGE": "nvcr.io/nvidia/pytorch:24.10-py3"},
                },
                "setup": "pip install .[dev]",
                "jobs": {
                    "gpu": {
                        "resources": {
                            "gpu_ids": "0",
                            "memory": "32GB",
                            "shm_size": "16g",
                            "timeout": 3600,
                        },
                        "stages": [
                            {
                                "name": "test",
                                "run": "pytest tests/ -v",
                            }
                        ],
                    }
                },
            }
        },
    }
    return normalize_config(raw)
