from pathlib import Path

from tools.validate_media import validate as validate_media
from tools.validate_release import validate as validate_release
from tools.validate_results import EXPECTED
from tools.validate_results import validate as validate_results


def test_canonical_results_and_claim_boundaries():
    assert len(EXPECTED) == 16
    assert validate_results(Path("configs/contest/effiped-tier1.yaml")) == []


def test_media_manifest_and_hashes():
    assert validate_media() == []


def test_public_release_hygiene_and_svg_artifacts():
    assert validate_release() == []
