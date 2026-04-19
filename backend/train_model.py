import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an EfficientNet-B0 deepfake detector from video folders.",
    )
    parser.add_argument(
        "--real-dir",
        default="datasets/original_sequences/youtube/c23/videos",
        help="Directory containing real videos.",
    )
    parser.add_argument(
        "--fake-dir",
        default="datasets/manipulated_sequences/Deepfakes/c23/videos",
        help="Directory containing fake videos.",
    )
    parser.add_argument(
        "--output",
        default="models/deepfake_model.pth",
        help="Where to save the best checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=1,
        help="Train classifier head only for this many initial epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=8,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--learning-rate-head",
        type=float,
        default=1e-3,
        help="Learning rate for the classification head.",
    )
    parser.add_argument(
        "--learning-rate-backbone",
        type=float,
        default=1e-4,
        help="Learning rate for the EfficientNet backbone after unfreezing.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=3,
        help="How many training samples to draw from each video per epoch.",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=12,
        help="How many evenly spaced frames to inspect while looking for a face.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input size.",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=0,
        help="Optional cap per class for quick experiments. Use 0 for all videos.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_videos(directory: Path, max_items: int = 0) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")

    videos = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if max_items > 0:
        videos = videos[:max_items]

    if not videos:
        raise FileNotFoundError(f"No supported videos found in {directory}")

    return videos


def build_samples(real_videos: list[Path], fake_videos: list[Path]) -> list[tuple[Path, int]]:
    return [(path, 0) for path in real_videos] + [(path, 1) for path in fake_videos]


def split_samples(
    samples: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")

    by_class: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_samples in by_class.values():
        rng.shuffle(class_samples)
        val_count = max(1, int(len(class_samples) * val_split))
        if val_count >= len(class_samples):
            val_count = len(class_samples) - 1
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    if not train_samples or not val_samples:
        raise RuntimeError("Train/validation split produced an empty subset.")

    return train_samples, val_samples


def detect_face(frame: np.ndarray, face_cascade: cv2.CascadeClassifier) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return frame[y:y + h, x:x + w]

    height, width = frame.shape[:2]
    crop_size = min(height, width)
    offset_y = max(0, (height - crop_size) // 2)
    offset_x = max(0, (width - crop_size) // 2)
    return frame[offset_y:offset_y + crop_size, offset_x:offset_x + crop_size]


class VideoFaceDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int,
        frames_per_sample: int,
        samples_per_video: int,
        training: bool,
    ) -> None:
        self.samples = samples
        self.frames_per_sample = max(1, frames_per_sample)
        self.samples_per_video = max(1, samples_per_video if training else 1)
        self.training = training
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        transform_steps = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]

        if training:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomRotation(6),
                ]
            )

        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_video

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[index % len(self.samples)]
        frame = self._sample_frame(video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        return tensor, torch.tensor(label, dtype=torch.long)

    def _sample_frame(self, video_path: Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError(f"Could not read frames from {video_path}")

            indices = np.linspace(0, max(frame_count - 1, 0), self.frames_per_sample, dtype=int)
            if self.training:
                np.random.shuffle(indices)

            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                cropped = detect_face(frame, self.face_cascade)
                if cropped.size > 0:
                    return cropped

            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if ok and frame is not None:
                return detect_face(frame, self.face_cascade)
        finally:
            capture.release()

        raise RuntimeError(f"Failed to sample a frame from {video_path}")


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def build_model() -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def build_optimizer(
    model: nn.Module,
    learning_rate_head: float,
    learning_rate_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameter_groups = [
        {
            "params": [p for p in model.features.parameters() if p.requires_grad],
            "lr": learning_rate_backbone,
        },
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": learning_rate_head,
        },
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    losses: list[float] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        losses.append(loss.item())
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return EpochMetrics(
        loss=float(np.mean(losses)),
        accuracy=accuracy_score(all_targets, all_predictions),
        precision=precision_score(all_targets, all_predictions, zero_division=0),
        recall=recall_score(all_targets, all_predictions, zero_division=0),
        f1=f1_score(all_targets, all_predictions, zero_division=0),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    real_videos = list_videos(Path(args.real_dir), args.max_videos_per_class)
    fake_videos = list_videos(Path(args.fake_dir), args.max_videos_per_class)

    train_samples, val_samples = split_samples(
        build_samples(real_videos, fake_videos),
        val_split=args.val_split,
        seed=args.seed,
    )

    train_dataset = VideoFaceDataset(
        samples=train_samples,
        image_size=args.image_size,
        frames_per_sample=args.frames_per_sample,
        samples_per_video=args.samples_per_video,
        training=True,
    )
    val_dataset = VideoFaceDataset(
        samples=val_samples,
        image_size=args.image_size,
        frames_per_sample=args.frames_per_sample,
        samples_per_video=1,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(
        model,
        learning_rate_head=args.learning_rate_head,
        learning_rate_backbone=args.learning_rate_backbone,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_f1 = -1.0

    print(
        f"Training on {len(train_samples)} videos, validating on {len(val_samples)} videos "
        f"using device={device}."
    )

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(
                model,
                learning_rate_head=args.learning_rate_head,
                learning_rate_backbone=args.learning_rate_backbone,
                weight_decay=args.weight_decay,
            )

        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_precision={val_metrics.precision:.4f} val_recall={val_metrics.recall:.4f} "
            f"val_f1={val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "fake_class_index": 1,
            "normalization": "imagenet",
            "label_map": ["real", "fake"],
        },
        output_path,
    )

    print(f"Saved best checkpoint to {output_path} with validation F1={best_f1:.4f}")
    print("For inference with this ImageNet-pretrained fine-tuned model, set USE_IMAGENET_NORMALIZATION=1.")


if __name__ == "__main__":
    main()
