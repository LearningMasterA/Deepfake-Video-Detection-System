from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class _Activations:
    fmap: Optional[torch.Tensor] = None
    grad: Optional[torch.Tensor] = None


class GradCAM:
    """Grad-CAM extractor for CNN-like models.

    Args:
        model: PyTorch model used for inference.
        target_layer: Layer from which activations/gradients are captured.
        method: `gradcam` (default) or `gradcam++`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        method: Literal["gradcam", "gradcam++"] = "gradcam",
    ):
        self.model = model
        self.target_layer = target_layer
        self.method = method
        self._cache = _Activations()
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, _module, _inputs, output) -> None:
        self._cache.fmap = output

    def _backward_hook(self, _module, _grad_in, grad_out) -> None:
        self._cache.grad = grad_out[0]

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _select_targets(self, logits: torch.Tensor, target_class: Optional[int]) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.ndim == 2 and logits.shape[1] == 1:
            if target_class in (None, 1):
                return logits[:, 0].sum()
            if target_class == 0:
                return (-logits[:, 0]).sum()
            raise ValueError("target_class for binary head must be 0 or 1")

        if target_class is None:
            target_idx = logits.argmax(dim=1)
        else:
            target_idx = torch.full((logits.shape[0],), target_class, dtype=torch.long, device=logits.device)

        return logits.gather(1, target_idx.unsqueeze(1)).sum()

    def _compute_cam_weights(self, gradients: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        if self.method == "gradcam++":
            grad2 = gradients.pow(2)
            grad3 = gradients.pow(3)
            denom = 2 * grad2 + (activations * grad3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = grad2 / (denom + 1e-7)
            positive_grad = F.relu(gradients)
            return (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)
        return gradients.mean(dim=(2, 3), keepdim=True)

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """Generate normalized CAM maps for a batch.

        Returns tensor with shape [batch, height, width] and range [0, 1].
        """
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(input_tensor)
            selected = self._select_targets(logits, target_class)
            selected.backward(retain_graph=False)

        if self._cache.fmap is None or self._cache.grad is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self._cache.grad
        activations = self._cache.fmap

        weights = self._compute_cam_weights(gradients, activations)
        cam = F.relu((weights * activations).sum(dim=1))

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Robust normalization: clip outliers first for better localization contrast.
        b = cam.shape[0]
        flat = cam.view(b, -1)
        low = torch.quantile(flat, 0.02, dim=1).view(b, 1, 1)
        high = torch.quantile(flat, 0.98, dim=1).view(b, 1, 1)
        cam = torch.clamp(cam, low, high)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()


def overlay_heatmap(
    image_bgr: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Overlay CAM map on an OpenCV BGR image.

    blur_kernel smooths noisy saliency peaks for visually cleaner explanations.
    """
    if cam_map.ndim != 2:
        raise ValueError("cam_map must be 2D.")

    heatmap = np.clip(cam_map, 0, 1)
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        heatmap = cv2.GaussianBlur(heatmap, (k, k), sigmaX=0)

    heatmap_u8 = np.uint8(heatmap * 255)
    heatmap_resized = cv2.resize(heatmap_u8, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_bgr = cv2.applyColorMap(heatmap_resized, colormap)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
