import argparse
from pathlib import Path

from inference import DECISION_THRESHOLD, FAKE_CLASS_INDEX, predict
from preprocessing import extract_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deepfake model on a single video and print detailed scores.",
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=15,
        help="Maximum number of faces to extract.",
    )
    parser.add_argument(
        "--output-prefix",
        default="inspect",
        help="Prefix for any extracted frame images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames, raw_images, image_paths = extract_faces(
        str(video_path),
        output_prefix=args.output_prefix,
        start_time_seconds=args.start_time,
        max_faces=args.max_faces,
    )

    confidence, prediction, _, fake_score, frame_scores = predict(frames, raw_images)

    print("Single Video Check")
    print(f"Video: {video_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Fake score: {fake_score:.4f}")
    print(f"Decision threshold: {DECISION_THRESHOLD:.4f}")
    print(f"Fake class index: {FAKE_CLASS_INDEX}")
    print(f"Frames used: {len(frame_scores)}")

    if image_paths:
        print("Extracted frames:")
        for image_path in image_paths:
            print(f"  {image_path}")

    if frame_scores:
        print("Frame scores:")
        for item in frame_scores:
            print(f"  frame={item['frame']:>2} fake_score={item['fake_score']:.4f}")


if __name__ == "__main__":
    main()
