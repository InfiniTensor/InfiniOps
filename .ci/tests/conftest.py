import sys
from pathlib import Path

# Allow `import run` and `import build` directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def minimal_config():
    return {
        "repo": {
            "url": "https://github.com/InfiniTensor/InfiniOps.git",
            "branch": "master",
        },
        "images": {
            "nvidia": {
                "dockerfile": ".ci/images/nvidia/",
                "build_args": {"BASE_IMAGE": "nvcr.io/nvidia/pytorch:24.10-py3"},
            }
        },
        "jobs": {
            "nvidia_gpu": {
                "image": "latest",
                "platform": "nvidia",
                "resources": {
                    "gpu_ids": "0",
                    "memory": "32GB",
                    "shm_size": "16g",
                    "timeout": 3600,
                },
                "setup": "pip install .[dev]",
                "stages": [
                    {
                        "name": "test",
                        "run": "pytest tests/ -v",
                    }
                ],
            }
        },
    }
