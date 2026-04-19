import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detection.gradcam import GradCAM, overlay_heatmap
from model_loader import model


GRADCAM_METHOD = os.getenv("GRADCAM_METHOD", "gradcam++").lower()
GRADCAM_ALPHA = float(os.getenv("GRADCAM_ALPHA", "0.38"))
GRADCAM_BLUR_KERNEL = int(os.getenv("GRADCAM_BLUR_KERNEL", "9"))
ENABLE_GRADCAM_TTA = os.getenv("ENABLE_GRADCAM_TTA", "1") != "0"
GRADCAM_TARGET_LAYER_INDEX = int(os.getenv("GRADCAM_TARGET_LAYER_INDEX", "-2"))
GRADCAM_POWER = float(os.getenv("GRADCAM_POWER", "1.6"))
GRADCAM_PERCENTILE_FLOOR = float(os.getenv("GRADCAM_PERCENTILE_FLOOR", "70"))
GRADCAM_USE_FACE_MASK = os.getenv("GRADCAM_USE_FACE_MASK", "1") != "0"

_cam_extractor = None


def _resolve_target_layer():
    features = list(model.features.children())
    if not features:
        raise RuntimeError("Model does not expose feature blocks for Grad-CAM.")

    index = GRADCAM_TARGET_LAYER_INDEX
    if index < 0:
        index = len(features) + index
    index = max(0, min(index, len(features) - 1))
    return features[index]


def get_cam_extractor() -> GradCAM:
    global _cam_extractor

    if _cam_extractor is None:
        target_layer = _resolve_target_layer()
        method = "gradcam++" if GRADCAM_METHOD == "gradcam++" else "gradcam"
        _cam_extractor = GradCAM(model=model, target_layer=target_layer, method=method)

    return _cam_extractor


def _postprocess_cam_map(cam_map: np.ndarray, raw_image: np.ndarray) -> np.ndarray:
    cam_map = np.clip(cam_map, 0.0, 1.0)

    percentile_floor = float(np.clip(GRADCAM_PERCENTILE_FLOOR, 0.0, 99.0))
    if percentile_floor > 0:
        floor = np.percentile(cam_map, percentile_floor)
        cam_map = np.clip((cam_map - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)

    power = max(0.1, GRADCAM_POWER)
    cam_map = np.power(cam_map, power)

    if GRADCAM_USE_FACE_MASK:
        h, w = raw_image.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        radius_x = max(1.0, w * 0.48)
        radius_y = max(1.0, h * 0.48)
        mask = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
        cam_map = cam_map * mask.astype(np.float32)

    cam_min = float(cam_map.min())
    cam_max = float(cam_map.max())
    if cam_max > cam_min:
        cam_map = (cam_map - cam_min) / (cam_max - cam_min)

    return cam_map.astype(np.float32)


def generate_gradcam(image_tensor, raw_image, target_class=None):
    cam_extractor = get_cam_extractor()
    device = next(model.parameters()).device
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.enable_grad():
        cam_map = cam_extractor(input_tensor=input_tensor, target_class=target_class)[0]

        if ENABLE_GRADCAM_TTA:
            flipped_tensor = torch.flip(input_tensor, dims=[3])
            flipped_cam_map = cam_extractor(
                input_tensor=flipped_tensor,
                target_class=target_class,
            )[0]
            flipped_cam_map = torch.flip(flipped_cam_map, dims=[1])
            cam_map = (cam_map + flipped_cam_map) / 2.0

    cam_map_np = cam_map.detach().cpu().numpy().astype(np.float32)
    cam_map_np = cv2.resize(
        cam_map_np,
        (raw_image.shape[1], raw_image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    cam_map_np = _postprocess_cam_map(cam_map_np, raw_image)

    return overlay_heatmap(
        raw_image,
        cam_map_np,
        alpha=GRADCAM_ALPHA,
        blur_kernel=GRADCAM_BLUR_KERNEL,
    )
