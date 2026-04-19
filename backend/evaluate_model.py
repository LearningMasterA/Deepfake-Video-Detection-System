import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from inference import predict
from preprocessing import extract_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deepfake model on labeled real/fake videos.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation_dataset",
        help="Folder containing real/ and fake/ subfolders.",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=0,
        help="Optional cap per class for quick checks. Use 0 for all videos.",
    )
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional path to save per-video predictions as CSV.",
    )
    return parser.parse_args()


def list_videos(folder: Path, max_videos: int) -> list[Path]:
    videos = sorted(path for path in folder.iterdir() if path.is_file())
    if max_videos > 0:
        videos = videos[:max_videos]
    return videos


def label_to_int(label: str) -> int:
    return 1 if label == "fake" else 0


def prediction_to_int(prediction: str) -> int:
    return 1 if prediction == "Fake" else 0


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)

    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Missing evaluation dataset folder: {dataset}. Expected "
            f"{dataset}/real and {dataset}/fake."
        )

    results: list[dict[str, object]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0

    for label in ["real", "fake"]:
        folder = dataset / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for video_path in list_videos(folder, args.max_videos_per_class):
            frames, raw_images, _ = extract_faces(str(video_path), output_prefix="eval")

            try:
                confidence, prediction, _, fake_score, frame_scores = predict(frames, raw_images)
            except RuntimeError as exc:
                raise RuntimeError(f"Cannot evaluate {video_path}: {exc}") from exc

            if prediction not in {"Fake", "Real"}:
                skipped += 1
                print(f"SKIP | {video_path.name:<25} | true={label:<4} | reason={prediction}")
                continue

            true_label = label_to_int(label)
            pred_label = prediction_to_int(prediction)

            y_true.append(true_label)
            y_pred.append(pred_label)

            row = {
                "video": video_path.name,
                "path": str(video_path),
                "true_label": label,
                "predicted_label": prediction.lower(),
                "correct": pred_label == true_label,
                "confidence": round(float(confidence), 4),
                "fake_score": round(float(fake_score), 4),
                "frames_used": len(frame_scores),
            }
            results.append(row)

            status = "OK" if row["correct"] else "ERR"
            print(
                f"{status} | {video_path.name:<25} | true={label:<4} | "
                f"pred={prediction.lower():<4} | fake_score={fake_score:.4f} | "
                f"confidence={confidence:.4f} | frames={len(frame_scores)}"
            )

    if len(y_true) == 0:
        raise RuntimeError("No videos were evaluated.")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print()
    print("Summary")
    print(f"Videos evaluated: {len(y_true)}")
    print(f"Videos skipped: {skipped}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("Confusion Matrix")
    print(f"TN={tn}  FP={fp}")
    print(f"FN={fn}  TP={tp}")

    if args.csv_output:
        output_path = Path(args.csv_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "video",
                    "path",
                    "true_label",
                    "predicted_label",
                    "correct",
                    "confidence",
                    "fake_score",
                    "frames_used",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved per-video results to {output_path}")


if __name__ == "__main__":
    main()
