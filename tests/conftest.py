import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return
    skip = pytest.mark.skip(reason="needs >= 2 CUDA GPUs")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
