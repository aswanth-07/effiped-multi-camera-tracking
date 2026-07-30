from dataclasses import dataclass

import numpy as np

from effiped.tracking.cross_camera_associator import CrossCameraAssociator


@dataclass
class TrackStub:
    track_id: int
    smooth_feat: np.ndarray


def test_cross_camera_association_reuses_global_identity():
    associator = CrossCameraAssociator(match_thresh=0.25, temporal_window=30, distinctiveness_ratio=0)
    query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    near = np.asarray([0.99, 0.05, 0.0, 0.0], dtype=np.float32)

    first = associator.update({"C1": [TrackStub(11, query)]}, timestamp=1.0)
    second = associator.update({"C2": [TrackStub(21, near)]}, timestamp=2.0)

    assert first["C1"][11] == second["C2"][21]
