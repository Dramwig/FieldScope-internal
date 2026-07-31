from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.prepare_nyuv2 import _depth_image, _rgb_image, _split_indices


def test_nyuv2_orientation_and_split_helpers(tmp_path: Path) -> None:
    rgb_source = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4)
    rgb = _rgb_image(rgb_source)
    assert rgb.shape == (4, 2, 3)

    depth_source = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    depth = _depth_image(depth_source, (4, 2))
    assert depth.shape == (4, 2)

    splits_path = tmp_path / "splits.mat"
    savemat(
        splits_path,
        {
            "trainNdxs": np.arange(1, 11)[:, None],
            "testNdxs": np.arange(11, 13)[:, None],
        },
    )
    splits = _split_indices(splits_path, seed=4121)
    assert len(splits["train"]) == 9
    assert len(splits["val"]) == 1
    assert splits["test"] == [10, 11]
