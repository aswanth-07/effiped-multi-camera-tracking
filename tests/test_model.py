from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from effiped.model import build_jdenet_from_config


@pytest.fixture(scope="module")
def compact_config():
    config = yaml.safe_load(Path("configs/contest/effiped-tier1.yaml").read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["model"]["pretrained"] = False
    config["model"]["use_dcn"] = False
    config["model"]["reid_head_use_dcn"] = False
    config["model"]["p3_refiner_depth"] = 1
    config["data"]["img_size"] = [128, 128]
    return config


def test_model_construction_and_synthetic_forward(compact_config):
    model = build_jdenet_from_config(compact_config, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 128, 128))
    assert {"hm", "wh", "offset", "embedding"}.issubset(output)
    assert output["hm"].shape[-2:] == (32, 32)
    assert output["embedding"].shape[1] == 256
