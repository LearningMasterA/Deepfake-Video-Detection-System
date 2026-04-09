from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch


def make_default_preprocess(
    image_size: int = 224,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    interpolation: int = cv2.INTER_AREA,
) -> callable:
    """Return OpenCV->tensor preprocessing callable.

    Args:
        image_size: Output square size.
        mean/std: Channel normalization values.
        interpolation: OpenCV interpolation mode.
    """

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=interpolation)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return (tensor - mean_tensor) / std_tensor

    return _preprocess
