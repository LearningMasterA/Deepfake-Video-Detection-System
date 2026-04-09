from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch

from .gradcam import GradCAM, overlay_heatmap


@dataclass
class ExplanationResult:
    fake_score: float
    predicted_label: str
    decision_threshold: float
    sampled_frames: int
    total_frames: int
    heatmap_video_path: Optional[Path]


class DeepfakeVideoExplainer:
    """Runs per-frame deepfake scoring and CAM heatmap generation."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        device: str = "cpu",
        cam_method: str = "gradcam++",
        enable_tta: bool = True,
        fake_class_index: int = 1,
        decision_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.enable_tta = enable_tta
        self.fake_class_index = fake_class_index
        self.decision_threshold = decision_threshold
        self.gradcam = GradCAM(self.model, target_layer, method=cam_method)

    def probability_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits into fake-class probabilities."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.ndim == 2 and logits.shape[1] > 1:
            if self.fake_class_index >= logits.shape[1]:
                raise ValueError(
                    f"fake_class_index={self.fake_class_index} out of range for logits shape {tuple(logits.shape)}"
                )
            return torch.softmax(logits, dim=1)[:, self.fake_class_index]

        # Single-logit binary head: logit predicts positive/fake class probability.
        return torch.sigmoid(logits.reshape(-1))

    def _predict_with_tta(self, tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.model(tensor)
            prob = self.probability_from_logits(logits)[0]
            if not self.enable_tta:
                return float(prob.cpu().item())

            flip_tensor = torch.flip(tensor, dims=[3])
            flip_logits = self.model(flip_tensor)
            flip_prob = self.probability_from_logits(flip_logits)[0]
            return float(((prob + flip_prob) / 2.0).cpu().item())

    def explain_video(
        self,
        input_video: str | Path,
        output_video: Optional[str | Path] = None,
        every_nth_frame: int = 5,
        target_class: Optional[int] = None,
    ) -> ExplanationResult:
        input_video = Path(input_video)
        if output_video is not None:
            output_video = Path(output_video)

        if target_class is None:
            target_class = self.fake_class_index

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {input_video}")

        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        scores: List[float] = []
        frame_idx = 0
        total_frames = 0
        sampled = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                total_frames += 1

                if output_video is not None and writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

                if frame_idx % every_nth_frame == 0:
                    tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
                    cam_map = self.gradcam(tensor, target_class=target_class)[0].cpu().numpy()
                    score = self._predict_with_tta(tensor)

                    scores.append(score)
                    sampled += 1

                    if writer is not None:
                        label = "FAKE" if score >= self.decision_threshold else "REAL"
                        overlay = overlay_heatmap(frame, cam_map, alpha=0.45, blur_kernel=7)
                        cv2.putText(
                            overlay,
                            f"{label} p_fake={score:.3f}",
                            (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                        writer.write(overlay)
                elif writer is not None:
                    writer.write(frame)

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if scores:
            scores_np = np.asarray(scores, dtype=np.float32)
            # Balanced aggregation: blend median (robust) + top-tail mean (sensitive to manipulated spans).
            median_score = float(np.median(scores_np))
            sorted_scores = np.sort(scores_np)
            top_k = max(1, int(0.3 * len(sorted_scores)))
            tail_mean = float(sorted_scores[-top_k:].mean())
            video_score = 0.6 * tail_mean + 0.4 * median_score
        else:
            video_score = 0.0

        label = "FAKE" if video_score >= self.decision_threshold else "REAL"
        return ExplanationResult(
            fake_score=video_score,
            predicted_label=label,
            decision_threshold=self.decision_threshold,
            sampled_frames=sampled,
            total_frames=total_frames,
            heatmap_video_path=output_video,
        )
