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
