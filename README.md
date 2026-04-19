# Deepfake Video Detection System

This project integrates **higher-accuracy Grad-CAM explanations** with improved video-level scoring for deepfake detection.

## Prediction-correction improvements

- Added configurable `fake_class_index` so predictions map to the correct class in your trained head.
- Added configurable `decision_threshold` so final FAKE/REAL labels can be calibrated.
- Improved aggregation by blending median and top-tail confidence mean to reduce unstable video labels.
- Added **Grad-CAM++** (default), robust CAM normalization, and heatmap smoothing.
- Added optional **TTA** (horizontal flip) to stabilize frame-level predictions.
- Added **EfficientNet-B0** support in CLI and made it the default architecture.

---

## Python API usage

```python
import torch
from torchvision import models

from deepfake_detection import DeepfakeVideoExplainer, make_default_preprocess

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)  # [real, fake]

explainer = DeepfakeVideoExplainer(
    model=model,
    target_layer=model.features[-1],
    preprocess=make_default_preprocess(image_size=224),
    device="cpu",
    cam_method="gradcam++",
    enable_tta=True,
    fake_class_index=1,
    decision_threshold=0.55,
)

result = explainer.explain_video(
    input_video="input.mp4",
    output_video="output_with_heatmap.mp4",
    every_nth_frame=5,
)

print(result.predicted_label, result.fake_score)
```

---

## CLI usage

```bash
python -m deepfake_detection \
  --input-video input.mp4 \
  --output-video output_with_heatmap.mp4 \
  --checkpoint model.pth \
  --arch efficientnet_b0 \
  --fake-class-index 1 \
  --decision-threshold 0.55 \
  --cam-method gradcam++ \
  --device cpu \
  --image-size 224 \
  --every-nth-frame 5
```

### CLI notes
- Use `--fake-class-index` to match your model's label mapping.
- Tune `--decision-threshold` on validation data for correct FAKE/REAL decisions.
- Use `--disable-tta` if speed is more important than stability.
- `--checkpoint` accepts either a plain state dict or a dict containing `state_dict`.
- Set `--device cuda:0` when GPU is available.

---

## Fine-tuning the pre-trained model

The repository now includes a training script that fine-tunes an ImageNet-pretrained `EfficientNet-B0` and saves the checkpoint in the same format expected by `backend/model_loader.py`.

It uses the FaceForensics++-style video folders already present in this repo by default:

- Real videos: `datasets/original_sequences/youtube/c23/videos`
- Fake videos: `datasets/manipulated_sequences/Deepfakes/c23/videos`

Run:

```bash
python backend/train_model.py \
  --device cuda \
  --epochs 8 \
  --batch-size 8 \
  --samples-per-video 4 \
  --output models/deepfake_model.pth
```

Useful options:

- `--max-videos-per-class 50` for a quick smoke test.
- `--freeze-backbone-epochs 1` to train the classifier head first, then unfreeze EfficientNet.
- `--frames-per-sample 12` to inspect more candidate frames per video when face detection is inconsistent.

Important:

- The fine-tuning script applies ImageNet normalization because it starts from pretrained EfficientNet weights.
- When serving the resulting checkpoint through the FastAPI backend, set `USE_IMAGENET_NORMALIZATION=1` so inference preprocessing matches training.

---

## Checking if predictions are correct

To evaluate the model on a labeled test set, create:

```text
evaluation_dataset/
  real/
  fake/
```

Then run:

```bash
python backend/evaluate_model.py --dataset evaluation_dataset --csv-output evaluation_results.csv
```

This prints:

- Per-video results with the true label, predicted label, fake score, and confidence.
- Summary metrics: accuracy, precision, recall, and F1.
- A confusion matrix with `TN`, `FP`, `FN`, and `TP`.

To inspect one video in detail:

```bash
python backend/check_video_prediction.py --video path/to/video.mp4
```

That prints the final prediction, confidence, fake score, and the score for each extracted frame.

Important:

- If you fine-tuned from pretrained EfficientNet weights, set `USE_IMAGENET_NORMALIZATION=1` before evaluation.
- If your labels are mapped as `[real, fake]`, set `FAKE_CLASS_INDEX=1`. If the model behaves inverted, this is the first thing to verify.

---

## Running Frontend And Backend Together

From the `frontend` folder, you can start both servers with:

```bash
npm run dev
```

This project is now configured so that the backend started by that command automatically uses:

- `USE_IMAGENET_NORMALIZATION=1`
- `FAKE_CLASS_INDEX=0`
- `DECISION_THRESHOLD=0.5`
- `UNCERTAIN_MARGIN=0.05`
- `GRADCAM_METHOD=gradcam++`
- `GRADCAM_ALPHA=0.34`
- `GRADCAM_BLUR_KERNEL=7`
- `GRADCAM_TARGET_LAYER_INDEX=-2`
- `GRADCAM_POWER=1.6`
- `GRADCAM_PERCENTILE_FLOOR=70`
- `GRADCAM_USE_FACE_MASK=1`
- `ENABLE_GRADCAM_TTA=1`

That means the frontend flow will use the fine-tuned EfficientNet checkpoint settings by default, as long as your model is saved at `models/deepfake_model.pth`.

Notes:

- New checkpoints created by `backend/train_model.py` now store `fake_class_index=1` in checkpoint metadata.
- Older checkpoints may still behave as `fake_class_index=0`.
- Your current checkpoint appears inverted in practice, so the combined frontend/backend dev command now forces `FAKE_CLASS_INDEX=0`.
- You can always override this manually by setting `FAKE_CLASS_INDEX=0` or `FAKE_CLASS_INDEX=1` before starting the backend.
- Grad-CAM sharpness can be tuned further with `GRADCAM_TARGET_LAYER_INDEX`, `GRADCAM_POWER`, and `GRADCAM_PERCENTILE_FLOOR`.
