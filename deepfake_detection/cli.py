from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import models

from .preprocessing import make_default_preprocess
from .video_pipeline import DeepfakeVideoExplainer


def build_model(arch: str, checkpoint: str | None, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        target_layer = model.layer4[-1]
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        target_layer = model.layer4[-1]
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state))

    return model.to(device).eval(), target_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for deepfake video detection.")
    parser.add_argument("--input-video", required=True, help="Path to input video.")
    parser.add_argument("--output-video", required=True, help="Path to output annotated video.")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument(
        "--arch",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "resnet50"],
        help="Backbone architecture.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--image-size", type=int, default=224, help="Input frame size for the model.")
    parser.add_argument("--every-nth-frame", type=int, default=5, help="Sampling interval for Grad-CAM generation.")
    parser.add_argument(
        "--fake-class-index",
        type=int,
        default=1,
        help="Index of the FAKE class in multi-class logits (ignored for single-logit heads).",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Video-level threshold for FAKE vs REAL decision on p_fake.",
    )
    parser.add_argument(
        "--cam-method",
        default="gradcam++",
        choices=["gradcam", "gradcam++"],
        help="CAM formulation. gradcam++ is generally more precise on small artifacts.",
    )
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation for frame scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, target_layer = build_model(args.arch, args.checkpoint, device)
    preprocess = make_default_preprocess(image_size=args.image_size)

    explainer = DeepfakeVideoExplainer(
        model=model,
        target_layer=target_layer,
        preprocess=preprocess,
        device=args.device,
        cam_method=args.cam_method,
        enable_tta=not args.disable_tta,
        fake_class_index=args.fake_class_index,
        decision_threshold=args.decision_threshold,
    )

    result = explainer.explain_video(
        input_video=Path(args.input_video),
        output_video=Path(args.output_video),
        every_nth_frame=args.every_nth_frame,
        target_class=args.fake_class_index,
    )

    print(
        "Done:",
        {
            "architecture": args.arch,
            "predicted_label": result.predicted_label,
            "fake_score": round(result.fake_score, 4),
            "decision_threshold": result.decision_threshold,
            "sampled_frames": result.sampled_frames,
            "total_frames": result.total_frames,
            "cam_method": args.cam_method,
            "tta_enabled": not args.disable_tta,
            "heatmap_video_path": str(result.heatmap_video_path) if result.heatmap_video_path else None,
        },
    )


if __name__ == "__main__":
    main()
